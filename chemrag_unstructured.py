import os
import io
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

# Unstructured for robust PDF parsing (digital and scanned via hi_res/ocr)
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Element, Table, Title, NarrativeText, ListItem

import pandas as pd

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_community.vectorstores import Chroma

from auxiliary import ChemBERT_ChEMBL_pretrained_embeddings


# ------------------------------
# Configuration
# ------------------------------
DEFAULT_PERSIST_DIR = os.path.join(os.getcwd(), "chroma_chemrag")
DEFAULT_COLLECTION = "chemrag_unstructured"
DEFAULT_PDF_DIR = os.path.join(os.getcwd(), "RAG DB")


@dataclass
class IndexState:
    pdf_dir: str
    persist_dir: str
    collection_name: str
    chunk_size_chars: int = 1500
    chunk_overlap_chars: int = 150
    files: List[str] = field(default_factory=list)
    file_elements: List[Tuple[str, List[Element]]] = field(default_factory=list)
    documents: List[Document] = field(default_factory=list)


def _list_pdf_files(pdf_dir: str) -> List[str]:
    if not os.path.isdir(pdf_dir):
        return []
    out: List[str] = []
    for name in os.listdir(pdf_dir):
        if name.lower().endswith(".pdf"):
            out.append(os.path.join(pdf_dir, name))
    return sorted(out)


def _partition_single_pdf(path: str) -> List[Element]:
    try:
        # hi_res strategy will use OCR for scanned PDFs if available
        return partition_pdf(
            filename=path,
            strategy="hi_res",
            infer_table_structure=True,
            languages=["eng"],
        )
    except Exception:
        # Fallback to fast strategy without OCR if hi_res is unavailable
        return partition_pdf(
            filename=path,
            strategy="fast",
            infer_table_structure=True,
        )


def _table_to_markdown(table: Table) -> str:
    # Prefer HTML if provided by Unstructured; else try plaintext fallback
    html: Optional[str] = None
    try:
        html = getattr(table.metadata, "text_as_html", None)
    except Exception:
        html = None

    if html:
        try:
            dfs = pd.read_html(io.StringIO(html))
            if not dfs:
                raise ValueError("No tables parsed from HTML")
            md_parts: List[str] = []
            for df in dfs:
                md_parts.append(df.to_markdown(index=False))
            return "\n\n".join(md_parts)
        except Exception:
            pass

    # Fallback: use table.text (may be TSV-like); wrap it as code block so it's preserved
    try:
        raw_text = table.text or ""
    except Exception:
        raw_text = ""
    return raw_text.strip()


def _collect_context_around_table(elements: List[Element], idx: int, window: int = 5) -> str:
    # Capture a small window of nearby textual context (titles, paragraphs, list items)
    start = max(0, idx - window)
    end = min(len(elements), idx + window + 1)
    context_bits: List[str] = []
    for j in range(start, end):
        if j == idx:
            continue
        el = elements[j]
        if isinstance(el, (Title, NarrativeText, ListItem)):
            try:
                txt = (el.text or "").strip()
            except Exception:
                txt = ""
            if txt:
                context_bits.append(txt)
    # Limit to reasonable length
    context = "\n".join(context_bits)
    if len(context) > 2000:
        context = context[:2000]
    return context


def _describe_table_with_llm(llm: ChatOpenAI, table_markdown: str, context: str) -> str:
    prompt = (
        "You are a chemistry and materials science assistant. Given a table from a PDF and its nearby "
        "context, write a concise 1-3 sentence description of what the table contains and why it is relevant. "
        "Avoid hallucinations; only use the provided content."
    )
    messages = [
        ("system", prompt),
        ("user", f"Context:\n{context}\n\nTable (markdown):\n{table_markdown}"),
    ]
    try:
        resp = llm.invoke(messages)
        return (getattr(resp, "content", None) or "").strip()
    except Exception:
        return ""


def _flush_text_chunk(buffer: List[Tuple[str, Optional[int]]]) -> Tuple[Optional[str], Optional[List[int]]]:
    if not buffer:
        return None, None
    texts: List[str] = []
    pages: List[int] = []
    for txt, pg in buffer:
        if txt:
            texts.append(txt)
        if isinstance(pg, int):
            pages.append(pg)
    chunk = "\n\n".join(texts).strip()
    if not chunk:
        return None, (sorted(set(pages)) if pages else None)
    return chunk, (sorted(set(pages)) if pages else None)


def _elements_to_documents(
    file_path: str,
    elements: List[Element],
    llm: ChatOpenAI,
    chunk_size_chars: int,
    chunk_overlap_chars: int,
) -> List[Document]:
    docs: List[Document] = []
    text_buffer: List[Tuple[str, Optional[int]]] = []
    current_size = 0

    def add_text_doc(text: str, pages: Optional[List[int]]):
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": file_path,
                    "type": "text",
                    "pages": pages,
                    "page": (pages[0] if isinstance(pages, list) and len(pages) == 1 else None),
                },
            )
        )

    for idx, el in enumerate(elements):
        # If it's a table, flush any buffered text first, then add the table as its own chunk
        if isinstance(el, Table):
            chunk_text, pages = _flush_text_chunk(text_buffer)
            text_buffer.clear()
            current_size = 0
            if chunk_text:
                add_text_doc(chunk_text, pages)

            table_md = _table_to_markdown(el)
            context = _collect_context_around_table(elements, idx, window=5)
            description = _describe_table_with_llm(llm, table_md, context)

            content_parts: List[str] = []
            if description:
                content_parts.append(f"Table description: {description}")
            content_parts.append(table_md)
            content = "\n\n".join(content_parts).strip()
            try:
                page_num = getattr(getattr(el, "metadata", None), "page_number", None)
            except Exception:
                page_num = None
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": file_path,
                        "type": "table",
                        "pages": ([int(page_num)] if isinstance(page_num, int) else None),
                        "page": (int(page_num) if isinstance(page_num, int) else None),
                    },
                )
            )
            continue

        # Otherwise, aggregate textual elements, respecting chunk size
        try:
            txt = (el.text or "").strip()
        except Exception:
            txt = ""

        if not txt:
            continue

        # Titles and list items are informative; include as regular text
        if isinstance(el, (Title, NarrativeText, ListItem)):
            try:
                pg = getattr(getattr(el, "metadata", None), "page_number", None)
            except Exception:
                pg = None
            addition = ("\n" + txt) if current_size > 0 else txt
            if current_size + len(addition) > chunk_size_chars:
                # flush with overlap
                chunk_text, pages = _flush_text_chunk(text_buffer)
                text_buffer.clear()
                if chunk_text:
                    add_text_doc(chunk_text, pages)
                # overlap handling: keep tail of the previous chunk
                if chunk_overlap_chars > 0 and chunk_text:
                    tail = chunk_text[-chunk_overlap_chars:] if chunk_text else ""
                    text_buffer.append((tail, None))
                    current_size = len(tail)
                else:
                    current_size = 0
            text_buffer.append((txt, int(pg) if isinstance(pg, int) else None))
            current_size += len(addition)

    # Flush remaining text buffer
    tail_text, tail_pages = _flush_text_chunk(text_buffer)
    text_buffer.clear()
    if tail_text:
        add_text_doc(tail_text, tail_pages)

    return docs


# ------------------------------
# LangGraph Nodes
# ------------------------------
def _node_list_files(state: IndexState) -> IndexState:
    state.files = _list_pdf_files(state.pdf_dir)
    return state


def _node_partition_pdfs(state: IndexState) -> IndexState:
    file_elements: List[Tuple[str, List[Element]]] = []
    for path in state.files:
        try:
            elements = _partition_single_pdf(path)
        except Exception:
            elements = []
        file_elements.append((path, elements))
    state.file_elements = file_elements
    return state


def _node_chunk_and_describe(state: IndexState) -> IndexState:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)  # uses OPENAI_API_KEY
    docs: List[Document] = []
    for path, elements in state.file_elements:
        if not elements:
            continue
        file_docs = _elements_to_documents(
            file_path=path,
            elements=elements,
            llm=llm,
            chunk_size_chars=state.chunk_size_chars,
            chunk_overlap_chars=state.chunk_overlap_chars,
        )
        docs.extend(file_docs)
    state.documents = docs
    return state


def _node_build_index(state: IndexState) -> IndexState:
    if not state.documents:
        return state
    os.makedirs(state.persist_dir, exist_ok=True)
    embeddings = ChemBERT_ChEMBL_pretrained_embeddings()
    # Use LangChain's Chroma wrapper with persistent storage
    Chroma.from_documents(
        documents=state.documents,
        collection_name=state.collection_name,
        embedding=embeddings,
        persist_directory=state.persist_dir,
    )
    return state


def build_index(
    pdf_dir: str = DEFAULT_PDF_DIR,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    chunk_size_chars: int = 1500,
    chunk_overlap_chars: int = 150,
) -> Dict[str, Any]:
    state = IndexState(
        pdf_dir=pdf_dir,
        persist_dir=persist_dir,
        collection_name=collection_name,
        chunk_size_chars=chunk_size_chars,
        chunk_overlap_chars=chunk_overlap_chars,
    )

    graph = StateGraph(IndexState)
    graph.add_node("list_files", _node_list_files)
    graph.add_node("partition", _node_partition_pdfs)
    graph.add_node("chunk", _node_chunk_and_describe)
    graph.add_node("index", _node_build_index)

    graph.add_edge(START, "list_files")
    graph.add_edge("list_files", "partition")
    graph.add_edge("partition", "chunk")
    graph.add_edge("chunk", "index")
    graph.add_edge("index", END)

    app = graph.compile()
    final_state: IndexState = app.invoke(state)

    return {
        "files_indexed": len(final_state.files),
        "documents_indexed": len(final_state.documents),
        "persist_dir": persist_dir,
        "collection_name": collection_name,
    }


def query_index(
    query: str,
    k: int = 10,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
) -> List[Dict[str, Any]]:
    embeddings = ChemBERT_ChEMBL_pretrained_embeddings()
    vs = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    retriever = vs.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)
    results: List[Dict[str, Any]] = []
    for d in docs:
        try:
            metadata = getattr(d, "metadata", {}) or {}
            results.append(
                {
                    "content": getattr(d, "page_content", ""),
                    "metadata": {
                        "source": metadata.get("source"),
                        "type": metadata.get("type"),
                    },
                }
            )
        except Exception:
            continue
    return results


__all__ = [
    "build_index",
    "query_index",
]


