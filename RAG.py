from langchain_community.document_loaders import ArxivLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from auxiliary import all_mini_l6_v2_pretrained_embeddings, ChemBERT_ChEMBL_pretrained_embeddings
import re


@tool
def retrieve_context(query: str) -> list:
    """
    Retrieve matching documents from arXiv academic repository
    :param query: user input for retrieving information
    :return: a list that contains dictionaries, each with a relevant content chunk (key 'Content'),
     title of the paper (key 'Title') and authors of the paper (key 'Authors').
    """
    try:
        # Check if PyMuPDF is available
        try:
            import fitz
        except ImportError:
            print("Warning: PyMuPDF not available. Installing PyMuPDF for PDF processing...")
            try:
                import subprocess
                import sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "PyMuPDF>=1.23.0"])
                import fitz
            except Exception as install_error:
                print(f"Warning: PyMuPDF installation failed: {install_error}")
                print("Continuing without PDF processing capabilities...")
                return []
        except Exception as fitz_error:
            print(f"Warning: PyMuPDF error: {fitz_error}")
            print("Continuing without PDF processing capabilities...")
            return []
        
        loader = ArxivLoader(query=query, load_max_docs=100, top_k_results=10)
        # text_splitter = TokenTextSplitter(chunk_size=800, chunk_overlap=80, encoding_name="cl100k_base")
        text_splitter = SemanticChunker(embeddings=all_mini_l6_v2_pretrained_embeddings())
        doc_splits = loader.load_and_split(text_splitter)
        # vectorstore = Chroma.from_documents(documents=doc_splits, collection_name='energetic_docs',
        #                                     embedding=NVIDIAEmbeddings())
        # vectorstore = Chroma.from_documents(documents=doc_splits, collection_name='energetic_docs',
        #                                     embedding=OpenAIEmbeddings(model='text-embedding-3-large'))
        # vectorstore = Chroma.from_documents(documents=doc_splits, collection_name='energetic_docs',
        #                                     embedding=HuggingFaceEmbeddings(
        #                                         model_name="sentence-transformers/allenai-specter"))
        # vectorstore = Chroma.from_documents(documents=doc_splits, collection_name='energetic_docs',
        #                                     embedding=HuggingFaceEmbeddings(
        #                                         model_name="sentence-transformers/all-MiniLM-L6-v2"))
        # Build vectorstore with CPU/GPU-aware embeddings (handled in auxiliary)
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name='energetic_docs',
            embedding=ChemBERT_ChEMBL_pretrained_embeddings()
        )

        retriever = vectorstore.as_retriever(search_kwargs={'k': 10})
        retrieved_chunks = retriever.invoke(query)
        results = []
        for i, chunk in enumerate(retrieved_chunks):
            metadata = getattr(chunk, 'metadata', {}) or {}

            # Title (robust to key casing)
            title = metadata.get('Title') or metadata.get('title') or 'Unknown Title'

            # Authors may be list or string and key may vary in casing
            authors_meta = metadata.get('Authors') or metadata.get('authors') or []
            if isinstance(authors_meta, (list, tuple)):
                authors_str = ", ".join(map(str, authors_meta))
            else:
                authors_str = str(authors_meta) if authors_meta else 'Unknown Authors'

            # Extract a 4-digit year from common metadata fields
            year = ''
            for key in [
                'Year', 'year', 'Published', 'published', 'PublicationDate',
                'publication_date', 'UpdateDate', 'update_date', 'Created',
                'created', 'pub_date', 'date'
            ]:
                if key in metadata and metadata[key]:
                    match = re.search(r'(19|20)\d{2}', str(metadata[key]))
                    if match:
                        year = match.group(0)
                        break

            results.append({
                'Content': chunk.page_content,
                'Title': title,
                'Authors': authors_str,
                'Year': year
            })
        return results
        
    except Exception as e:
        print(f"Error in RAG retrieval: {e}")
        # Return empty results instead of failing
        return []
