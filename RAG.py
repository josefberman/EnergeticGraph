from langchain_community.document_loaders import ArxivLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from auxiliary import all_mini_l6_v2_pretrained_embeddings, ChemBERT_ChEMBL_pretrained_embeddings
import re
import os
import pandas as pd
from typing import Dict, Any, List, Optional
try:
    from prediction import predict_properties
    from prediction import convert_name_to_smiles
except Exception:
    predict_properties = None  # type: ignore
    convert_name_to_smiles = None  # type: ignore


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


def _calc_property_error_mape(properties: Dict[str, float], target_properties: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
    weighted_sum = 0.0
    total_weight = 0.0
    epsilon = 1e-9
    weights = weights or {k: 1.0 for k in target_properties.keys()}
    for prop_name, target_value in target_properties.items():
        if prop_name in properties and prop_name in weights:
            current_value = float(properties[prop_name])
            target_val = float(target_value)
            weight = float(weights[prop_name])
            denom = max(epsilon, abs(target_val))
            err = abs(current_value - target_val) / denom
            weighted_sum += weight * err
            total_weight += weight
    return (weighted_sum / total_weight) if total_weight > 0 else weighted_sum


def _collect_source_frames() -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    p = os.path.join(os.getcwd(), 'sample_start_molecules.csv')
    if os.path.exists(p):
        try:
            frames.append(pd.read_csv(p))
        except Exception:
            pass
    return frames


def retrieve_top_molecules_by_properties(target_properties: Dict[str, float], weights: Optional[Dict[str, float]] = None, top_k: int = 10, return_trace: bool = False):
    # 1) Build a retrieval query from target properties
    trace: Dict[str, Any] = {'query': None, 'retrieved_titles': [], 'retrieved_count': 0, 'names_extracted': 0, 'names_converted': 0, 'smiles_extracted': 0, 'candidates_scored': 0, 'fallback_used': False}
    try:
        target_str = ", ".join(f"{k}: {v}" for k, v in target_properties.items())
    except Exception:
        target_str = str(target_properties)
    query = f"energetic materials with properties close to -> {target_str} ; include SMILES or molecule name"
    trace['query'] = query

    # 2) Use Arxiv RAG to retrieve relevant documents
    try:
        chunks = retrieve_context(query)  # type: ignore[name-defined]
    except Exception:
        chunks = []
    try:
        trace['retrieved_count'] = len(chunks)
        trace['retrieved_titles'] = [c.get('Title', 'Unknown') for c in chunks[:10]] if isinstance(chunks, list) else []
    except Exception:
        pass

    # 3) Extract SMILES candidates (and names -> SMILES) from retrieved text
    smiles_candidates: List[str] = []
    names_candidates: List[str] = []
    converted_pairs: List[Dict[str, str]] = []
    try:
        import re as _re
        from rdkit import Chem as _Chem
    except Exception:
        _re = None
        _Chem = None

    def _extract_smiles(text: str) -> List[str]:
        if not _re:
            return []
        cands: List[str] = []
        # Look for explicit SMILES labels first
        for m in _re.finditer(r"SMILES\s*[:=]\s*([A-Za-z0-9@+\-\[\]\(\)=#$/%.]+)", text, _re.IGNORECASE):
            cands.append(m.group(1).strip())
        # Generic fallback: tokens with typical SMILES charset, length 2..200
        for m in _re.finditer(r"(?<![A-Za-z0-9])([A-Za-z0-9@+\-\[\]\(\)=#$/%.]{2,200})(?![A-Za-z0-9])", text):
            tok = m.group(1).strip()
            # Heuristic: must contain at least one bracket or bond or digit or '=' to avoid plain words
            if not _re.search(r"[\[\]#=\d]", tok):
                continue
            cands.append(tok)
        # Validate with RDKit
        uniq: List[str] = []
        for s in cands:
            if s in uniq:
                continue
            if _Chem is not None:
                try:
                    if _Chem.MolFromSmiles(s) is None:
                        continue
                except Exception:
                    continue
            uniq.append(s)
        return uniq[:200]

    def _extract_names(text: str) -> List[str]:
        if not _re:
            return []
        names: List[str] = []
        # Heuristics: capture lines like "Name: ..." or tokens containing typical chemical name patterns
        # 1) Label-based
        for m in _re.finditer(r"(?:Name|Compound|Molecule)\s*[:=]\s*([A-Za-z0-9,\-\s()]+)", text, _re.IGNORECASE):
            cand = m.group(1).strip()
            if 2 <= len(cand) <= 200:
                names.append(cand)
        # 2) Pattern-based: look for words with hyphens, commas, numerals typical in IUPAC names
        for m in _re.finditer(r"([A-Za-z][A-Za-z0-9,\-()]{2,200})", text):
            tok = m.group(1)
            # Heuristics to avoid plain English: require at least one of '-', ',', digit
            if not _re.search(r"[-,\d]", tok):
                continue
            # Exclude tokens that look like pure numbers or section codes
            if _re.fullmatch(r"[\d,\-()]+", tok):
                continue
            names.append(tok)
        # Deduplicate, simple trimming
        seen = set()
        out: List[str] = []
        for n in names:
            n2 = n.strip()
            if n2 and n2 not in seen:
                seen.add(n2)
                out.append(n2)
        return out[:200]

    for ch in (chunks or [])[:20]:
        content = ch.get('Content') if isinstance(ch, dict) else getattr(ch, 'page_content', '')
        if not content:
            continue
        text = str(content)
        # Direct SMILES
        for s in _extract_smiles(text):
            smiles_candidates.append(s)
        # Names -> SMILES conversion
        if convert_name_to_smiles is not None:
            for nm in _extract_names(text):
                names_candidates.append(nm)
                try:
                    smi = convert_name_to_smiles.invoke(nm)
                    if isinstance(smi, str) and smi and smi.lower() != 'did not convert':
                        smi2 = smi.strip()
                        smiles_candidates.append(smi2)
                        converted_pairs.append({'name': nm, 'smiles': smi2})
                except Exception:
                    continue
    trace['names_extracted'] = len(names_candidates)
    trace['names_converted'] = len(converted_pairs)

    # 4) Score SMILES from RAG by property similarity
    results: List[Dict[str, Any]] = []
    if smiles_candidates:
        for smi in list(dict.fromkeys(smiles_candidates))[:200]:
            try:
                if predict_properties is None:
                    continue
                props = predict_properties.invoke(smi)
                score = _calc_property_error_mape(props, target_properties, weights)
                results.append({'smiles': smi, 'properties': props, 'score': float(score)})
            except Exception:
                continue
    trace['smiles_extracted'] = len(set(smiles_candidates))
    trace['candidates_scored'] = len(results)

    # 5) Fallback to local CSV if RAG yielded no valid molecules
    if not results:
        frames = _collect_source_frames()
        if frames:
            trace['fallback_used'] = True
            df = pd.concat(frames, ignore_index=True)
            smiles_col = None
            for c in ['SMILES', 'smiles', 'Smiles']:
                if c in df.columns:
                    smiles_col = c
                    break
            if smiles_col is not None:
                property_columns = list(target_properties.keys())
                for _, row in df.iterrows():
                    smi = str(row.get(smiles_col, '') or '')
                    if not smi:
                        continue
                    props: Dict[str, float] = {}
                    missing = False
                    for p in property_columns:
                        if p in df.columns and pd.notna(row.get(p, None)):
                            try:
                                props[p] = float(row[p])
                            except Exception:
                                missing = True
                        else:
                            missing = True
                    if missing:
                        try:
                            if predict_properties is not None:
                                props = predict_properties.invoke(smi)
                            else:
                                continue
                        except Exception:
                            continue
                    try:
                        score = _calc_property_error_mape(props, target_properties, weights)
                    except Exception:
                        continue
                    results.append({'smiles': smi, 'properties': props, 'score': float(score)})

    results.sort(key=lambda x: float(x.get('score', 1e9)))
    top = results[:top_k]
    if return_trace:
        trace['top_preview'] = [{'smiles': r.get('smiles'), 'score': r.get('score')} for r in top[:5]]
        return top, trace
    return top
