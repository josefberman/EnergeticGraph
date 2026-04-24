"""
RAG (Retrieval-Augmented Generation) module for energetic property lookup.

This module searches scientific literature for known property values before
falling back to ML prediction. It:
1. Converts SMILES to proper chemical names (IUPAC/common)
2. Searches multiple databases (OpenAlex, ArXiv, Crossref, Semantic Scholar) for papers
3. Extracts energetic properties from paper abstracts using regex/LLM
4. Returns found properties, with None for properties not found
"""

import os
import re
import json
import logging
import requests
import tempfile
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from rdkit import Chem

# Try to import PDF parsing library
try:
    import fitz  # PyMuPDF
    PDF_PARSER_AVAILABLE = True
except ImportError as e:
    PDF_PARSER_AVAILABLE = False
    logging.getLogger(__name__).warning(f"PyMuPDF not installed: {e}. Install with: pip install pymupdf")
except Exception as e:
    PDF_PARSER_AVAILABLE = False
    logging.getLogger(__name__).warning(f"PyMuPDF import failed: {e}")

# Try to import sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logging.getLogger(__name__).warning("sentence-transformers not installed. Install with: pip install sentence-transformers")

logger = logging.getLogger(__name__)


class TextChunker:
    """
    Chunks text into smaller pieces for embedding-based retrieval.
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Full text to chunk
            
        Returns:
            List of text chunks
        """
        if not text or len(text) < self.chunk_size:
            return [text] if text else []
        
        chunks = []
        
        # Split by paragraphs first for more natural breaks
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph exceeds chunk size, save current and start new
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Keep overlap from end of previous chunk
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:] + "\n\n" + para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks


class ChunkRetriever:
    """
    Retrieves relevant chunks using embedding-based similarity search.
    """
    
    # Singleton pattern for model (expensive to load)
    _model = None
    _model_name = "all-MiniLM-L6-v2"  # Fast, good quality embeddings
    
    def __init__(self):
        """Initialize retriever with sentence transformer model."""
        if not EMBEDDINGS_AVAILABLE:
            self._model = None
            return
            
        # Load model (singleton)
        if ChunkRetriever._model is None:
            try:
                logger.info(f"Loading embedding model: {self._model_name}")
                ChunkRetriever._model = SentenceTransformer(self._model_name)
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                ChunkRetriever._model = None
    
    def retrieve_relevant_chunks(
        self, 
        chunks: List[str], 
        query: str, 
        top_k: int = 5,
        similarity_threshold: float = 0.3
    ) -> List[Tuple[str, float]]:
        """
        Retrieve most relevant chunks for a query using cosine similarity.
        
        Args:
            chunks: List of text chunks
            query: Search query (e.g., "density detonation velocity TNT")
            top_k: Number of top chunks to return
            similarity_threshold: Minimum similarity score to include
            
        Returns:
            List of (chunk, similarity_score) tuples, sorted by relevance
        """
        if not chunks:
            return []
        
        # If embeddings not available, return all chunks with equal score
        if not EMBEDDINGS_AVAILABLE or ChunkRetriever._model is None:
            return [(chunk, 1.0) for chunk in chunks[:top_k]]
        
        try:
            # Embed query and chunks
            query_embedding = ChunkRetriever._model.encode([query])[0]
            chunk_embeddings = ChunkRetriever._model.encode(chunks)
            
            # Calculate cosine similarities
            similarities = []
            for i, chunk_emb in enumerate(chunk_embeddings):
                # Cosine similarity
                similarity = np.dot(query_embedding, chunk_emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk_emb) + 1e-8
                )
                similarities.append((chunks[i], float(similarity)))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Filter by threshold and return top_k
            relevant = [
                (chunk, score) for chunk, score in similarities 
                if score >= similarity_threshold
            ]
            
            return relevant[:top_k]
            
        except Exception as e:
            logger.warning(f"Embedding retrieval failed: {e}")
            # Fallback: return first chunks
            return [(chunk, 1.0) for chunk in chunks[:top_k]]


@dataclass
class RetrievedProperty:
    """A property value retrieved from literature."""
    value: float
    source: str  # Paper title or DOI
    confidence: float  # 0-1 confidence score


@dataclass
class PaperCitation:
    """Citation information for a paper that provided property data."""
    title: str
    authors: List[str]
    doi: str
    source_db: str  # OpenAlex, Crossref, SemanticScholar
    properties_found: List[str]  # Which properties were extracted from this paper


@dataclass
class RAGResult:
    """Result of RAG property lookup."""
    smiles: str
    chemical_name: Optional[str]
    properties: Dict[str, Optional[RetrievedProperty]]  # Property name -> RetrievedProperty or None
    papers_searched: int
    papers_with_hits: int
    citations: List[PaperCitation] = None  # Papers that provided property data
    
    def __post_init__(self):
        if self.citations is None:
            self.citations = []


class SMILESToNameConverter:
    """
    Converts SMILES strings to chemical names.
    
    Priority:
    1. PubChemPy - queries PubChem database for known compounds (common/IUPAC names)
    2. SMILES2IUPAC - neural network model for generating IUPAC names (novel molecules)
    """
    
    def __init__(self, timeout: int = 10):
        """
        Initialize converter.
        
        Args:
            timeout: API timeout in seconds (not used by pubchempy directly)
        """
        self.timeout = timeout
        self._pubchempy_available = False
        self._smiles2iupac_available = False
        self._iupac_converter = None
        
        # Check if pubchempy is available
        try:
            import pubchempy as pcp
            self._pcp = pcp
            self._pubchempy_available = True
            logger.info("PubChemPy available for SMILES-to-name conversion")
        except ImportError:
            logger.warning("pubchempy not installed. Install with: pip install pubchempy")
            self._pubchempy_available = False
        
        # Check if chemical-converters (SMILES2IUPAC) is available
        try:
            from chemicalconverters import NamesConverter
            self._iupac_converter = NamesConverter(model_name="knowledgator/SMILES2IUPAC-canonical-base")
            self._smiles2iupac_available = True
            logger.info("SMILES2IUPAC model available for IUPAC name generation")
        except ImportError:
            logger.warning("chemical-converters not installed. Install with: pip install chemical-converters")
            self._smiles2iupac_available = False
        except Exception as e:
            logger.warning(f"Failed to load SMILES2IUPAC model: {e}")
            self._smiles2iupac_available = False
    
    def convert(self, smiles: str) -> Optional[str]:
        """
        Convert SMILES to chemical name.
        
        Priority:
        1. Try PubChem (for known compounds - gets common names)
        2. Fall back to SMILES2IUPAC model (generates IUPAC for any molecule)
        
        Args:
            smiles: SMILES string
            
        Returns:
            Chemical name (common or IUPAC) or None if conversion fails
        """
        # Canonicalize SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        canonical_smiles = Chem.MolToSmiles(mol)
        
        # 1. Try PubChemPy first (for known compounds)
        if self._pubchempy_available:
            name = self._query_pubchempy(canonical_smiles)
            if name:
                return name
        
        # 2. Fall back to SMILES2IUPAC neural network model
        if self._smiles2iupac_available:
            name = self._generate_iupac_smiles2iupac(canonical_smiles)
            if name:
                return name
        
        return None
    
    def _generate_iupac_smiles2iupac(self, smiles: str) -> Optional[str]:
        """
        Generate IUPAC name using the SMILES2IUPAC neural network model.
        
        Uses the knowledgator/SMILES2IUPAC-canonical-base model from HuggingFace.
        Model has 86.9% accuracy on IUPAC name generation.
        
        Args:
            smiles: SMILES string
            
        Returns:
            IUPAC name or None if generation fails
        """
        try:
            # Use BASE style for most common/recognizable name
            result = self._iupac_converter.smiles_to_iupac(f"<BASE>{smiles}")
            
            if result and len(result) > 0:
                iupac_name = result[0]
                # Validate that we got a real name (not empty or same as input)
                if iupac_name and iupac_name != smiles and len(iupac_name) > 0:
                    return iupac_name
            
            return None
            
        except Exception as e:
            logger.debug(f"SMILES2IUPAC generation failed for {smiles[:30]}...: {e}")
            return None
    
    def _query_pubchempy(self, smiles: str) -> Optional[str]:
        """
        Query PubChem for chemical name using pubchempy.
        
        Tries to get:
        1. Common/traditional name first
        2. IUPAC name as fallback
        
        Args:
            smiles: Canonical SMILES string
            
        Returns:
            Chemical name or None if not found
        """
        try:
            # Search PubChem by SMILES
            compounds = self._pcp.get_compounds(smiles, 'smiles')
            
            if not compounds:
                logger.debug(f"No PubChem compounds found for {smiles[:30]}...")
                return None
            
            compound = compounds[0]
            
            # Try to get synonyms (includes common names)
            synonyms = compound.synonyms
            
            # Priority: common name > IUPAC name
            # Common names are usually shorter and at the beginning of synonyms list
            if synonyms:
                # Filter out CID references and very long names
                good_names = [
                    s for s in synonyms[:10]  # Check first 10 synonyms
                    if not s.startswith('CID')
                    and not s.startswith('CHEMBL')
                    and not s.startswith('SCHEMBL')
                    and not s.startswith('DTXSID')
                    and not s.startswith('EINECS')
                    and len(s) < 100
                ]
                
                if good_names:
                    # Prefer shorter names (often common names)
                    best_name = min(good_names, key=len)
                    return best_name
            
            # Fallback to IUPAC name
            iupac_name = compound.iupac_name
            if iupac_name:
                return iupac_name
            
            # Last resort: use the first synonym
            if synonyms:
                return synonyms[0]
            
            return None
            
        except Exception as e:
            logger.warning(f"PubChemPy error for {smiles[:30]}...: {e}")
            return None


class LiteratureSearcher:
    """
    Searches multiple scientific literature databases for papers mentioning a chemical compound.
    
    Uses multiple APIs for robustness:
    1. OpenAlex (free, no auth required) - primary source
    2. ArXiv (free, open access) - secondary source for preprints
    3. Crossref (free, polite pool) - tertiary source
    4. Semantic Scholar (free tier) - quaternary source
    """
    
    # Standard headers to avoid 403 errors
    HEADERS = {
        'User-Agent': 'EnergeticMoleculeDesigner/1.0 (https://github.com/energetic-design; mailto:research@example.com)',
        'Accept': 'application/json',
    }
    
    def __init__(self, max_results: int = 10, timeout: int = 15):
        """
        Initialize searcher.
        
        Args:
            max_results: Maximum number of papers to retrieve
            timeout: API timeout in seconds
        """
        self.max_results = max_results
        self.timeout = timeout
    
    # Common abbreviations for search queries
    SEARCH_ALIASES = {
        'trinitrotoluene': 'TNT',
        '2,4,6-trinitrotoluene': 'TNT',
        'cyclotrimethylenetrinitramine': 'RDX',
        'cyclotetramethylenetetranitramine': 'HMX',
        'triaminotrinitrobenzene': 'TATB',
        'pentaerythritol tetranitrate': 'PETN',
        'hexanitrohexaazaisowurtzitane': 'CL-20',
        '1,1-diamino-2,2-dinitroethylene': 'FOX-7',
        'dinitrotoluene': 'DNT',
        '2,4-dinitrotoluene': 'DNT',
        'trinitroazetidine': 'TNAZ',
        '3-nitro-1,2,4-triazol-5-one': 'NTO',
        'dinitroanisole': 'DNAN',
        '2,4-dinitroanisole': 'DNAN',
        'hexanitrostilbene': 'HNS',
        'trinitrophenol': 'picric acid',
        '2,4,6-trinitrophenol': 'picric acid',
        'trinitrophenylmethylnitramine': 'tetryl',
    }
    
    def _get_search_terms(self, chemical_name: str) -> str:
        """
        Get search terms including common abbreviations.
        
        Args:
            chemical_name: Chemical name
            
        Returns:
            Search string with name and abbreviation if known
        """
        name_lower = chemical_name.lower()
        
        # Check if we have a common abbreviation
        for full_name, abbrev in self.SEARCH_ALIASES.items():
            if full_name in name_lower or name_lower in full_name:
                # Return both name and abbreviation for better search
                return f'("{abbrev}" OR "{chemical_name}")'
        
        return f'"{chemical_name}"'
    
    def search(self, chemical_name: str, smiles: str = None) -> List[Dict]:
        """
        Search multiple databases for papers mentioning the chemical.
        
        Args:
            chemical_name: Chemical name to search for
            smiles: Optional SMILES string to include in search
            
        Returns:
            List of paper dictionaries with title, abstract, doi, authors
        """
        papers = []
        
        # Get search terms (including abbreviations)
        search_term = self._get_search_terms(chemical_name)
        
        # 1. Try ArXiv FIRST - has full text PDFs!
        arxiv_papers = self._search_arxiv(chemical_name, search_term)
        papers.extend(arxiv_papers)
        logger.info(f"ArXiv found {len(arxiv_papers)} papers with full text for '{chemical_name}'")
        
        # 2. Try OpenAlex for additional coverage (abstracts only)
        if len(papers) < self.max_results:
            openalex_papers = self._search_openalex(chemical_name, search_term)
            papers.extend(openalex_papers)
            logger.info(f"OpenAlex found {len(openalex_papers)} additional papers")
        
        # 3. Try Crossref for additional coverage
        if len(papers) < self.max_results:
            crossref_papers = self._search_crossref(chemical_name, search_term)
            papers.extend(crossref_papers)
            logger.info(f"Crossref found {len(crossref_papers)} additional papers")
        
        # 4. Try Semantic Scholar as fallback
        if len(papers) < self.max_results // 2:
            semantic_papers = self._search_semantic_scholar(chemical_name, search_term)
            papers.extend(semantic_papers)
            logger.info(f"Semantic Scholar found {len(semantic_papers)} additional papers")
        
        # Deduplicate by DOI
        seen_dois = set()
        unique_papers = []
        for paper in papers:
            doi = paper.get('doi', '')
            if doi and doi not in seen_dois:
                seen_dois.add(doi)
                unique_papers.append(paper)
            elif not doi:
                # Keep papers without DOI but limit them
                if len([p for p in unique_papers if not p.get('doi')]) < 3:
                    unique_papers.append(paper)
        
        logger.info(f"Total unique papers found: {len(unique_papers[:self.max_results])}")
        return unique_papers[:self.max_results]
    
    def _search_openalex(self, chemical_name: str, search_term: str = None) -> List[Dict]:
        """
        Search OpenAlex for papers (free, no auth required).
        
        OpenAlex is a free, open catalog of the world's scholarly works.
        """
        papers = []
        
        try:
            # OpenAlex API - search works
            # Use search_term if provided (includes abbreviations)
            base_term = search_term if search_term else chemical_name
            query = f'{base_term} energetic explosive detonation'
            url = "https://api.openalex.org/works"
            params = {
                'search': query,
                'per_page': min(self.max_results, 25),
                'filter': 'type:article',
                'select': 'id,doi,title,abstract_inverted_index,authorships,publication_date',
            }
            
            response = requests.get(url, params=params, headers=self.HEADERS, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                
                for item in results:
                    # Reconstruct abstract from inverted index
                    abstract = self._reconstruct_abstract(item.get('abstract_inverted_index', {}))
                    
                    # Extract author names
                    authors = []
                    for authorship in item.get('authorships', []):
                        author_info = authorship.get('author', {})
                        name = author_info.get('display_name', '')
                        if name:
                            authors.append(name)
                    
                    paper = {
                        'title': item.get('title', ''),
                        'text': abstract,
                        'doi': item.get('doi', '').replace('https://doi.org/', '') if item.get('doi') else '',
                        'authors': authors,
                        'published_date': item.get('publication_date', ''),
                        'source': 'OpenAlex'
                    }
                    
                    if paper['title']:
                        papers.append(paper)
            else:
                logger.warning(f"OpenAlex API returned status {response.status_code}")
                        
        except requests.exceptions.Timeout:
            logger.warning(f"OpenAlex API timeout for '{chemical_name}'")
        except Exception as e:
            logger.warning(f"OpenAlex API error: {e}")
        
        return papers
    
    def _reconstruct_abstract(self, inverted_index: dict) -> str:
        """Reconstruct abstract from OpenAlex inverted index format."""
        if not inverted_index:
            return ''
        
        try:
            # Create list of (position, word) tuples
            word_positions = []
            for word, positions in inverted_index.items():
                for pos in positions:
                    word_positions.append((pos, word))
            
            # Sort by position and join
            word_positions.sort(key=lambda x: x[0])
            abstract = ' '.join(word for _, word in word_positions)
            return abstract
        except Exception:
            return ''
    
    def _download_arxiv_pdf(self, arxiv_id: str) -> Optional[str]:
        """
        Download ArXiv PDF and extract full text.
        
        Args:
            arxiv_id: ArXiv paper ID (e.g., "2301.12345")
            
        Returns:
            Full text content or None if failed
        """
        if not PDF_PARSER_AVAILABLE:
            return None
            
        try:
            # ArXiv PDF URL format
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            
            response = requests.get(pdf_url, headers=self.HEADERS, timeout=30)
            
            if response.status_code == 200:
                # Save to temp file and parse
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                    tmp_file.write(response.content)
                    tmp_path = tmp_file.name
                
                try:
                    # Extract text using PyMuPDF
                    doc = fitz.open(tmp_path)
                    full_text = ""
                    for page in doc:
                        full_text += page.get_text()
                    doc.close()
                    
                    # Clean up temp file
                    os.unlink(tmp_path)
                    
                    return full_text
                except Exception as e:
                    logger.debug(f"PDF parsing error for {arxiv_id}: {e}")
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    return None
            else:
                logger.debug(f"Failed to download ArXiv PDF {arxiv_id}: status {response.status_code}")
                return None
                
        except Exception as e:
            logger.debug(f"ArXiv PDF download error for {arxiv_id}: {e}")
            return None
    
    def _search_arxiv(self, chemical_name: str, search_term: str = None) -> List[Dict]:
        """
        Search ArXiv for papers and extract FULL TEXT from PDFs.
        
        ArXiv is a repository for physics, chemistry, and materials science preprints.
        Uses the ArXiv API: https://arxiv.org/help/api/
        Downloads full PDFs for text extraction (not just abstracts).
        """
        papers = []
        
        try:
            # ArXiv API - search for papers
            # Use abbreviation if available for better results
            if search_term and 'OR' in search_term:
                abbrev_match = re.search(r'"([^"]+)"', search_term)
                base_term = abbrev_match.group(1) if abbrev_match else chemical_name
            else:
                base_term = chemical_name
            
            # Require a name-variant hit; no generic-fallback clause (which
            # used to dominate results with off-topic energetics papers).
            # Co-requiring an energetic keyword keeps precision high.
            energetic_terms = '(all:energetic OR all:detonation OR all:explosive)'
            if base_term.lower() != chemical_name.lower():
                query = (
                    f'(all:"{base_term}" OR all:"{chemical_name}") '
                    f'AND {energetic_terms}'
                )
            else:
                query = f'all:"{chemical_name}" AND {energetic_terms}'
            url = "http://export.arxiv.org/api/query"
            params = {
                'search_query': query,
                'max_results': min(self.max_results, 10),  # Limit due to PDF downloads
                'sortBy': 'relevance',
                'sortOrder': 'descending',
            }
            
            print(f"         🔍 ArXiv query: {base_term} (PDF parsing: {'✓' if PDF_PARSER_AVAILABLE else '✗'})...", end=" ", flush=True)
            
            response = requests.get(url, params=params, headers=self.HEADERS, timeout=self.timeout)
            
            if response.status_code == 200:
                # Parse XML response (ArXiv uses Atom feed format)
                import xml.etree.ElementTree as ET
                
                root = ET.fromstring(response.content)
                
                # Define namespaces used in ArXiv Atom feed
                ns = {
                    'atom': 'http://www.w3.org/2005/Atom',
                    'arxiv': 'http://arxiv.org/schemas/atom'
                }
                
                entries = root.findall('atom:entry', ns)
                print(f"found {len(entries)} papers")
                
                for entry in entries:
                    # Extract title
                    title_elem = entry.find('atom:title', ns)
                    title = title_elem.text.strip() if title_elem is not None and title_elem.text else ''
                    
                    # Extract abstract (as fallback)
                    abstract_elem = entry.find('atom:summary', ns)
                    abstract = abstract_elem.text.strip() if abstract_elem is not None and abstract_elem.text else ''
                    
                    # Extract authors
                    authors = []
                    for author_elem in entry.findall('atom:author', ns):
                        name_elem = author_elem.find('atom:name', ns)
                        if name_elem is not None and name_elem.text:
                            authors.append(name_elem.text.strip())
                    
                    # Extract ArXiv ID
                    id_elem = entry.find('atom:id', ns)
                    arxiv_id = ''
                    if id_elem is not None and id_elem.text:
                        # ID format: http://arxiv.org/abs/XXXX.XXXXX or http://arxiv.org/abs/cond-mat/XXXXXXX
                        arxiv_id = id_elem.text.replace('http://arxiv.org/abs/', '').replace('https://arxiv.org/abs/', '')
                    
                    # Extract published date
                    published_elem = entry.find('atom:published', ns)
                    published_date = published_elem.text[:10] if published_elem is not None and published_elem.text else ''
                    
                    # Try to get full text from PDF
                    full_text = None
                    if arxiv_id and PDF_PARSER_AVAILABLE:
                        print(f"            📥 Downloading PDF: {arxiv_id}...", end=" ", flush=True)
                        full_text = self._download_arxiv_pdf(arxiv_id)
                        if full_text:
                            print(f"✓ ({len(full_text)} chars)")
                        else:
                            print("✗ failed")
                    elif not PDF_PARSER_AVAILABLE:
                        print(f"            ⚠️ PyMuPDF not available (PDF_PARSER_AVAILABLE={PDF_PARSER_AVAILABLE})")
                    elif not arxiv_id:
                        print(f"            ⚠️ No ArXiv ID extracted - cannot download PDF")
                    
                    # Use full text if available, otherwise fall back to abstract
                    content = full_text if full_text else abstract
                    
                    paper = {
                        'title': title,
                        'text': content,  # Full text from PDF, or abstract as fallback
                        'doi': f'arXiv:{arxiv_id}' if arxiv_id else '',
                        'authors': authors,
                        'published_date': published_date,
                        'source': 'ArXiv-FullText' if full_text else 'ArXiv',
                        'has_full_text': full_text is not None
                    }
                    
                    if paper['title'] and paper['text']:
                        papers.append(paper)
            else:
                print(f"error (status {response.status_code})")
                logger.warning(f"ArXiv API returned status {response.status_code}")
                        
        except requests.exceptions.Timeout:
            print("timeout")
            logger.warning(f"ArXiv API timeout for '{chemical_name}'")
        except Exception as e:
            logger.warning(f"ArXiv API error: {e}")
        
        return papers
    
    def _search_crossref(self, chemical_name: str, search_term: str = None) -> List[Dict]:
        """Search Crossref for papers (free with polite pool)."""
        papers = []
        
        try:
            base_term = search_term.replace('"', '').replace('(', '').replace(')', '') if search_term else chemical_name
            query = f'{base_term} energetic explosive detonation'
            url = "https://api.crossref.org/works"
            params = {
                'query': query,
                'rows': min(10, self.max_results),
                'filter': 'type:journal-article',
            }
            
            response = requests.get(url, params=params, headers=self.HEADERS, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                items = data.get('message', {}).get('items', [])
                
                for item in items:
                    # Clean up abstract (remove HTML tags if present)
                    abstract = item.get('abstract', '')
                    if abstract:
                        abstract = re.sub(r'<[^>]+>', '', abstract)
                    
                    paper = {
                        'title': item.get('title', [''])[0] if item.get('title') else '',
                        'text': abstract,
                        'doi': item.get('DOI', ''),
                        'authors': [f"{a.get('given', '')} {a.get('family', '')}" 
                                   for a in item.get('author', [])],
                        'published_date': str(item.get('published-print', {}).get('date-parts', [['']])[0]),
                        'source': 'Crossref'
                    }
                    if paper['title']:
                        papers.append(paper)
            else:
                logger.debug(f"Crossref API returned status {response.status_code}")
                        
        except requests.exceptions.Timeout:
            logger.debug(f"Crossref API timeout")
        except Exception as e:
            logger.debug(f"Crossref search error: {e}")
        
        return papers
    
    def _search_semantic_scholar(self, chemical_name: str, search_term: str = None) -> List[Dict]:
        """Search Semantic Scholar for papers (free tier)."""
        papers = []
        
        try:
            base_term = search_term.replace('"', '').replace('(', '').replace(')', '') if search_term else chemical_name
            query = f'{base_term} energetic material'
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                'query': query,
                'limit': min(10, self.max_results),
                'fields': 'title,abstract,authors,externalIds,publicationDate',
            }
            
            response = requests.get(url, params=params, headers=self.HEADERS, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                items = data.get('data', [])
                
                for item in items:
                    # Get DOI from external IDs
                    external_ids = item.get('externalIds', {}) or {}
                    doi = external_ids.get('DOI', '')
                    
                    paper = {
                        'title': item.get('title', ''),
                        'text': item.get('abstract', '') or '',
                        'doi': doi,
                        'authors': [a.get('name', '') for a in (item.get('authors') or [])],
                        'published_date': item.get('publicationDate', ''),
                        'source': 'SemanticScholar'
                    }
                    if paper['title'] and paper['text']:
                        papers.append(paper)
            else:
                logger.debug(f"Semantic Scholar API returned status {response.status_code}")
                        
        except requests.exceptions.Timeout:
            logger.debug(f"Semantic Scholar API timeout")
        except Exception as e:
            logger.debug(f"Semantic Scholar search error: {e}")
        
        return papers


class PropertyExtractor:
    """
    Extracts energetic material properties from paper text (abstracts or full text).
    
    Uses regex patterns and optionally LLM for more complex extraction.
    For full-text papers (e.g. ArXiv), uses embedding-based chunk retrieval.
    """
    
    # Common aliases/abbreviations for energetic materials
    # Maps canonical name patterns to list of common names/abbreviations
    ENERGETIC_ALIASES = {
        'trinitrotoluene': ['tnt', '2,4,6-tnt', 'trinitrotoluene'],
        'rdx': ['rdx', 'cyclotrimethylenetrinitramine', 'hexogen', 'cyclonite', 'hexahydro-1,3,5-trinitro-1,3,5-triazine'],
        'hmx': ['hmx', 'cyclotetramethylenetetranitramine', 'octogen', 'octahydro-1,3,5,7-tetranitro-1,3,5,7-tetrazocine'],
        'tatb': ['tatb', 'triaminotrinitrobenzene', '1,3,5-triamino-2,4,6-trinitrobenzene'],
        'petn': ['petn', 'pentaerythritol tetranitrate', 'penthrite', 'nitropenta'],
        'nitroglycerin': ['nitroglycerin', 'nitroglycerine', 'glyceryl trinitrate', 'ng', 'gtn'],
        'picric acid': ['picric acid', 'trinitrophenol', '2,4,6-trinitrophenol', 'melinite', 'lyddite'],
        'tetryl': ['tetryl', 'trinitrophenylmethylnitramine', 'tetryl', 'pyronite'],
        'cl-20': ['cl-20', 'hexanitrohexaazaisowurtzitane', 'hniw', 'china lake 20'],
        'fox-7': ['fox-7', 'dadne', '1,1-diamino-2,2-dinitroethylene', 'diaminodinitroethylene'],
        'dnt': ['dnt', 'dinitrotoluene', '2,4-dinitrotoluene'],
        'tnaz': ['tnaz', 'trinitroazetidine', '1,3,3-trinitroazetidine'],
        'nto': ['nto', 'nitrotriazolone', '3-nitro-1,2,4-triazol-5-one', '5-nitro-2,4-dihydro-3h-1,2,4-triazol-3-one'],
        'dnan': ['dnan', 'dinitroanisole', '2,4-dinitroanisole'],
        'hns': ['hns', 'hexanitrostilbene', '2,2\',4,4\',6,6\'-hexanitrostilbene'],
    }
    
    # Property patterns with units - FLEXIBLE patterns to catch various formats
    PROPERTY_PATTERNS = {
        'Density': [
            # Standard: "density of 1.92 g/cm³", "ρ = 1.85 g cm⁻³"
            r'(?:density|ρ|rho|crystal\s+density|calculated\s+density)\s*(?:of|=|:|is|was)?\s*(\d+\.?\d*)\s*(?:g\s*/?\s*cm|g\s*cm|gcc)',
            r'(\d+\.?\d*)\s*(?:g\s*/?\s*cm|g/cm|gcc|g\s*cm)\s*.*?(?:density)',
            # Just number followed by g/cm3 in context of density discussion
            r'density[^.]{0,50}(\d+\.\d+)\s*(?:g|gcc)',
            # Parenthetical: "(1.92 g/cm³)"
            r'\(\s*(\d+\.\d+)\s*g\s*/?\s*cm',
        ],
        'Det Velocity': [
            # Standard: "detonation velocity of 8500 m/s", "explosion speed 8500 m/s"
            r'(?:detonation\s+velocity|detonation\s+speed|explosion\s+velocity|explosion\s+speed|det\.?\s*vel\.?|vod|velocity\s+of\s+detonation)\s*(?:of|=|:|is|was)?\s*(\d+\.?\d*)\s*(?:m\s*/?\s*s|m/s|ms)',
            r'(\d+\.?\d*)\s*(?:m\s*/?\s*s|m/s)\s*.*?(?:detonation|explosion|velocity|speed)',
            # With km/s units
            r'(?:detonation|explosion|velocity|speed)[^.]{0,30}(\d+\.?\d*)\s*(?:km\s*/?\s*s|km/s)',
            # Just number + m/s near detonation/explosion context
            r'(?:detonation|explosion)[^.]{0,50}(\d{4,5})\s*(?:m/s|m\s*/\s*s)',
        ],
        'Det Pressure': [
            # Standard: "detonation pressure of 39.5 GPa", "explosion pressure 39.5 GPa"
            r'(?:detonation\s+pressure|explosion\s+pressure|det\.?\s*press\.?|pcj|p_cj|chapman.jouguet)\s*(?:of|=|:|is|was)?\s*(\d+\.?\d*)\s*(?:GPa|gpa)',
            r'(\d+\.?\d*)\s*(?:GPa|gpa)\s*.*?(?:detonation|explosion|pressure|pcj)',
            # kbar units
            r'(?:detonation|explosion|pressure)[^.]{0,30}(\d+\.?\d*)\s*(?:kbar)',
            # Just number + GPa near detonation/explosion context  
            r'(?:detonation|explosion)[^.]{0,50}(\d+\.?\d*)\s*(?:GPa|gpa)',
        ],
        'Hf solid': [
            # Standard: "heat of formation of 200 kJ/mol", "enthalpy of formation -50 kJ/mol"
            r'(?:heat\s+of\s+formation|enthalpy\s+of\s+formation|formation\s+enthalpy|hof|Δhf|ΔH)\s*(?:of|=|:|is|was)?\s*([-−+]?\d+\.?\d*)\s*(?:kJ|kj)',
            r'([-−+]?\d+\.?\d*)\s*(?:kJ\s*/?\s*mol|kJ/mol|kj/mol)\s*.*?(?:heat|enthalpy|formation)',
            # kcal/mol units
            r'(?:heat|enthalpy|formation)[^.]{0,30}([-−+]?\d+\.?\d*)\s*(?:kcal)',
            # Just number + kJ/mol near formation context
            r'formation[^.]{0,50}([-−+]?\d+\.?\d*)\s*(?:kJ|kj)',
        ],
    }
    
    # Unit conversion factors
    UNIT_CONVERSIONS = {
        'km/s': 1000,  # km/s to m/s
        'kbar': 0.1,   # kbar to GPa
        'kcal/mol': 4.184,  # kcal/mol to kJ/mol
    }
    
    def __init__(self, use_llm: bool = False, llm_api_key: str = None,
                 use_chunking: bool = True,
                 ollama_base_url: Optional[str] = None,
                 ollama_model: Optional[str] = None):
        self.llm_api_key = llm_api_key or os.getenv('OPENAI_API_KEY')
        self.ollama_base_url = ollama_base_url or os.getenv('OLLAMA_BASE_URL')
        self.ollama_model = ollama_model or os.getenv('OLLAMA_MODEL', 'ALIENTELLIGENCE/chemicalengineer')

        has_backend = bool(self.ollama_base_url or self.llm_api_key)
        self.use_llm = bool(use_llm and has_backend)
        if use_llm and not has_backend:
            logger.warning("LLM extraction requested but no API key or Ollama URL configured; disabling.")
        self.use_chunking = use_chunking
        
        # Initialize chunker and retriever for full-text processing
        if use_chunking:
            self.chunker = TextChunker(chunk_size=500, chunk_overlap=100)
            self.retriever = ChunkRetriever()
    
    def _get_name_variants(self, chemical_name: str) -> List[str]:
        """
        Get all name variants (aliases) for a chemical name.
        
        Args:
            chemical_name: Chemical name to find variants for
            
        Returns:
            List of name variants to search for
        """
        if not chemical_name:
            return []
        
        name_lower = chemical_name.lower()
        variants = [name_lower]
        
        # Check if this name matches any known energetic material aliases
        for key, aliases in self.ENERGETIC_ALIASES.items():
            # Check if the chemical name contains any alias
            for alias in aliases:
                if alias in name_lower or name_lower in alias:
                    # Add all aliases for this compound
                    variants.extend(aliases)
                    break
        
        # Also add the base name without numbers/prefixes
        # E.g., "2,4,6-trinitrotoluene" -> "trinitrotoluene"
        base_name = re.sub(r'^[\d,\'-]+', '', name_lower).strip()
        if base_name and base_name != name_lower:
            variants.append(base_name)
        
        return list(set(variants))  # Remove duplicates
    
    def extract_from_text(self, text: str, chemical_name: str, is_full_text: bool = False) -> Dict[str, Optional[RetrievedProperty]]:
        """
        Extract energetic properties from paper text (abstract or full text).
        
        For full text, uses embedding-based chunk retrieval to find relevant sections,
        then applies both regex and LLM extraction on the retrieved chunks.
        
        Args:
            text: Paper text (abstract or full text)
            chemical_name: Name of the chemical to look for
            is_full_text: Whether this is full text (triggers chunking)
            
        Returns:
            Dictionary mapping property names to RetrievedProperty or None
        """
        properties = {
            'Density': None,
            'Det Velocity': None,
            'Det Pressure': None,
            'Hf solid': None
        }
        
        if not text:
            return properties
        
        # Get all name variants (including common abbreviations like TNT, RDX, etc.)
        name_variants = self._get_name_variants(chemical_name)
        
        # For full text, use chunking and retrieval
        if is_full_text and self.use_chunking and len(text) > 2000:
            properties = self._extract_from_chunks(text, chemical_name, name_variants)
        else:
            # For short text (abstracts), search directly with regex
            properties = self._extract_from_text(text, chemical_name, name_variants)
            
            # Also try LLM on the short text
            if self.use_llm and self.llm_api_key:
                llm_properties = self._extract_with_llm(text, chemical_name)
                for prop_name, prop_value in llm_properties.items():
                    if prop_value is not None:
                        current = properties.get(prop_name)
                        if current is None or prop_value.confidence > current.confidence:
                            properties[prop_name] = prop_value
        
        return properties
    
    def _extract_from_chunks(
        self, 
        full_text: str, 
        chemical_name: str,
        name_variants: List[str]
    ) -> Dict[str, Optional[RetrievedProperty]]:
        """
        Extract properties from full text using chunk retrieval.
        
        Chunks the text, retrieves relevant chunks via embedding similarity,
        then applies both regex and (optionally) LLM extraction on those chunks.
        
        Args:
            full_text: Full paper text
            chemical_name: Chemical name
            name_variants: List of name variants to search for
            
        Returns:
            Dictionary of extracted properties
        """
        properties = {
            'Density': None,
            'Det Velocity': None,
            'Det Pressure': None,
            'Hf solid': None
        }
        
        # Chunk the text
        chunks = self.chunker.chunk_text(full_text)
        logger.debug(f"Split text into {len(chunks)} chunks")
        
        if not chunks:
            return properties
        
        # Build query for retrieval — include chemical name + all property synonyms
        abbreviation = name_variants[0] if name_variants else chemical_name
        query = (
            f"{abbreviation} {chemical_name} "
            f"density "
            f"detonation velocity detonation speed explosion velocity explosion speed "
            f"detonation pressure explosion pressure "
            f"heat of formation enthalpy of formation "
            f"energetic explosive properties"
        )
        
        # Retrieve most relevant chunks
        relevant_chunks = self.retriever.retrieve_relevant_chunks(
            chunks, 
            query, 
            top_k=10,
            similarity_threshold=0.25
        )
        
        logger.debug(f"Retrieved {len(relevant_chunks)} relevant chunks")
        
        # Extract properties from each relevant chunk using regex
        for chunk, similarity in relevant_chunks:
            chunk_properties = self._extract_from_text(chunk, chemical_name, name_variants)
            
            # Merge (keep highest confidence, boost by similarity)
            for prop_name, prop_value in chunk_properties.items():
                if prop_value is not None:
                    # Boost confidence by chunk relevance
                    boosted_confidence = prop_value.confidence * (0.5 + 0.5 * similarity)
                    
                    current = properties.get(prop_name)
                    if current is None or boosted_confidence > current.confidence:
                        properties[prop_name] = RetrievedProperty(
                            value=prop_value.value,
                            source=f"Chunk (sim={similarity:.2f})",
                            confidence=boosted_confidence
                        )
        
        # Apply LLM extraction on the relevant chunks (not raw full text)
        if self.use_llm and self.llm_api_key and relevant_chunks:
            # Concatenate the top relevant chunks for LLM context
            llm_context = "\n\n".join(chunk for chunk, _ in relevant_chunks[:5])
            llm_properties = self._extract_with_llm(llm_context, chemical_name)
            
            # Merge LLM results (higher confidence overrides regex)
            for prop_name, prop_value in llm_properties.items():
                if prop_value is not None:
                    current = properties.get(prop_name)
                    if current is None or prop_value.confidence > current.confidence:
                        properties[prop_name] = prop_value
        
        return properties
    
    def _extract_from_text(
        self, 
        text: str, 
        chemical_name: str,
        name_variants: List[str]
    ) -> Dict[str, Optional[RetrievedProperty]]:
        """
        Extract properties from text using regex patterns.
        
        Args:
            text: Text to search
            chemical_name: Chemical name  
            name_variants: List of name variants
            
        Returns:
            Dictionary of extracted properties
        """
        properties = {
            'Density': None,
            'Det Velocity': None,
            'Det Pressure': None,
            'Hf solid': None
        }
        
        # Normalize text
        text_lower = text.lower()
        
        # Check if any variant of the chemical name is mentioned
        name_found = any(variant in text_lower for variant in name_variants)
        
        # Also check for generic energetic keywords as fallback
        energetic_keywords = ['energetic', 'explosive', 'detonation', 'propellant', 'munition']
        keyword_found = any(kw in text_lower for kw in energetic_keywords)
        
        # If neither name nor keywords found, skip
        if not name_found and not keyword_found:
            return properties
        
        # Extract each property using regex + require the chemical-name variant
        # to appear within a 400-char window of the numeric hit (prevents
        # picking values from unrelated molecules in the same paper).
        proximity_window = 400
        for prop_name, patterns in self.PROPERTY_PATTERNS.items():
            for pattern in patterns:
                for m in re.finditer(pattern, text_lower, re.IGNORECASE):
                    try:
                        value = float(m.group(1).replace('−', '-').replace('–', '-'))

                        # Apply unit conversions if needed
                        if 'km' in pattern and prop_name == 'Det Velocity':
                            value *= 1000  # km/s to m/s
                        elif 'kbar' in pattern and prop_name == 'Det Pressure':
                            value *= 0.1  # kbar to GPa
                        elif 'kcal' in pattern and prop_name == 'Hf solid':
                            value *= 4.184  # kcal/mol to kJ/mol

                        if not self._validate_value(prop_name, value):
                            continue

                        # Require chemical-name mention near the number
                        # (skipped if neither name nor keyword was found,
                        # which already returned early above).
                        if name_found and name_variants:
                            start = max(0, m.start() - proximity_window)
                            end = min(len(text_lower), m.end() + proximity_window)
                            window = text_lower[start:end]
                            if not any(v in window for v in name_variants):
                                continue

                        properties[prop_name] = RetrievedProperty(
                            value=value,
                            source="Regex extraction",
                            confidence=0.7
                        )
                        break
                    except (ValueError, IndexError):
                        continue
                if properties[prop_name] is not None:
                    break

        return properties
    
    def _validate_value(self, prop_name: str, value: float) -> bool:
        """Validate that extracted value is in reasonable range."""
        ranges = {
            'Density': (0.5, 3.0),  # g/cm³
            'Det Velocity': (4000, 12000),  # m/s
            'Det Pressure': (10, 60),  # GPa
            'Hf solid': (-500, 1000),  # kJ/mol (can be negative)
        }
        
        min_val, max_val = ranges.get(prop_name, (float('-inf'), float('inf')))
        return min_val <= value <= max_val
    
    def _extract_with_llm(self, text: str, chemical_name: str) -> Dict[str, Optional[RetrievedProperty]]:
        """Use LLM to extract properties from text (retrieved chunks or abstract)."""
        properties = {
            'Density': None,
            'Det Velocity': None,
            'Det Pressure': None,
            'Hf solid': None
        }
        
        try:
            client, model = _make_llm_client(
                self.llm_api_key, self.ollama_base_url, self.ollama_model)
            if client is None:
                return properties

            prompt = f"""Extract energetic material properties for "{chemical_name}" from the following text.

Text:
{text}

Search for these properties using any of the listed synonyms:
- "density": crystal density, calculated density, ρ (in g/cm³)
- "det_velocity": detonation velocity, detonation speed, explosion velocity, explosion speed, VOD (in m/s)
- "det_pressure": detonation pressure, explosion pressure, Chapman-Jouguet pressure, PCJ (in GPa)
- "heat_of_formation": heat of formation, enthalpy of formation, formation enthalpy, ΔHf (in kJ/mol)

Return ONLY a JSON object with these exact keys (use null if not found):
- "density": value in g/cm³
- "det_velocity": value in m/s
- "det_pressure": value in GPa
- "heat_of_formation": value in kJ/mol

JSON response:"""

            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200
            )
            
            result_text = response.choices[0].message.content or ''
            # Strip <think>...</think> blocks before searching for JSON.
            result_text = re.sub(r'<think>.*?</think>', '', result_text,
                                 flags=re.DOTALL).strip()
            # Match the outermost {...} object (handles nested keys).
            json_match = re.search(r'\{[^{}]*\}', result_text)
            if json_match:
                data = json.loads(json_match.group())
                
                mapping = {
                    'density': 'Density',
                    'det_velocity': 'Det Velocity',
                    'det_pressure': 'Det Pressure',
                    'heat_of_formation': 'Hf solid'
                }
                
                for json_key, prop_name in mapping.items():
                    value = data.get(json_key)
                    if value is not None and self._validate_value(prop_name, float(value)):
                        properties[prop_name] = RetrievedProperty(
                            value=float(value),
                            source="LLM extraction from retrieved text",
                            confidence=0.85
                        )
                        
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")
        
        return properties


def _make_llm_client(openai_api_key: Optional[str],
                     ollama_base_url: Optional[str],
                     ollama_model: Optional[str]):
    """Return (client, model_name) for whichever LLM backend is configured.

    Ollama takes priority if ``ollama_base_url`` is set.
    Ollama exposes an OpenAI-compatible ``/v1`` endpoint so we reuse the
    ``openai`` SDK with a custom ``base_url``.
    Returns (None, None) when neither backend is configured.
    """
    try:
        import openai
    except ImportError:
        return None, None

    if ollama_base_url:
        base = ollama_base_url.strip().rstrip('/')
        # Ensure scheme is present — openai SDK requires an absolute URL.
        if not base.startswith(('http://', 'https://')):
            base = 'http://' + base
        if not base.endswith('/v1'):
            base = f"{base}/v1"
        client = openai.OpenAI(base_url=base, api_key='ollama')
        model = ollama_model or 'ALIENTELLIGENCE/chemicalengineer'
        return client, model

    if openai_api_key:
        return openai.OpenAI(api_key=openai_api_key), 'gpt-4o-mini'

    return None, None


class RAGPropertyRetriever:
    """
    Main RAG module that orchestrates SMILES-to-name conversion,
    literature search, and property extraction.
    """
    
    def __init__(self,
                 use_llm: bool = False,
                 max_papers: int = 10,
                 timeout: int = 15,
                 openai_api_key: Optional[str] = None,
                 cache_path: Optional[str] = None,
                 ollama_base_url: Optional[str] = None,
                 ollama_model: Optional[str] = None):
        """
        Initialize RAG retriever.

        Args:
            use_llm: Whether to use LLM for property extraction
            max_papers: Maximum papers to search
            timeout: API timeout in seconds
            openai_api_key: OpenAI API key (falls back to OPENAI_API_KEY env var)
            cache_path: Optional SQLite cache path; disables caching if None
            ollama_base_url: Base URL of a local Ollama server, e.g.
                             "http://localhost:11434".  Takes priority over
                             openai_api_key when set.
            ollama_model: Model tag to use with Ollama (default: ALIENTELLIGENCE/chemicalengineer)
        """
        self.name_converter = SMILESToNameConverter(timeout=timeout)
        self.searcher = LiteratureSearcher(max_results=max_papers, timeout=timeout)
        # Resolve backends once: explicit args beat env vars.
        self._openai_api_key: Optional[str] = (
            openai_api_key or os.getenv('OPENAI_API_KEY') or None
        )
        self._ollama_base_url: Optional[str] = (
            ollama_base_url or os.getenv('OLLAMA_BASE_URL') or None
        )
        self._ollama_model: str = (
            ollama_model or os.getenv('OLLAMA_MODEL') or 'ALIENTELLIGENCE/chemicalengineer'
        )
        self.extractor = PropertyExtractor(
            use_llm=use_llm,
            llm_api_key=self._openai_api_key,
            ollama_base_url=self._ollama_base_url,
            ollama_model=self._ollama_model,
        )
        self.use_llm = self.extractor.use_llm

        # In-process analogue cache: avoids re-downloading within a single run.
        # { norm_name -> (properties_dict, citations_list, papers_searched) }
        self._analogue_mem: dict = {}

        self.cache = None
        if cache_path:
            try:
                from .rag_cache import RAGCache
                self.cache = RAGCache(cache_path)
                logger.info(f"RAG cache enabled at {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to initialize RAG cache at {cache_path}: {e}")
                self.cache = None
    
    def _get_chemical_name(self, smiles: str) -> Tuple[Optional[str], str]:
        """
        Get chemical name with source tracking.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Tuple of (chemical_name, source) where source is "PubChem" or "SMILES2IUPAC"
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, ""
        canonical_smiles = Chem.MolToSmiles(mol)
        
        # 1. Try PubChemPy first (for known compounds)
        if self.name_converter._pubchempy_available:
            name = self.name_converter._query_pubchempy(canonical_smiles)
            if name:
                return name, "PubChem"
        
        # 2. Fall back to SMILES2IUPAC neural network model
        if self.name_converter._smiles2iupac_available:
            name = self.name_converter._generate_iupac_smiles2iupac(canonical_smiles)
            if name:
                return name, "SMILES2IUPAC"
        
        return None, ""
    
    def retrieve_properties(self, smiles: str) -> RAGResult:
        """
        Retrieve properties for a molecule from literature.
        
        Args:
            smiles: SMILES string of the molecule
            
        Returns:
            RAGResult with found properties
        """
        # Cache hit short-circuits all network calls.
        if self.cache is not None:
            cached = self.cache.get(smiles)
            if cached is not None:
                logger.info(f"RAG cache hit: {smiles[:50]}")
                return cached

        # Initialize result
        properties = {
            'Density': None,
            'Det Velocity': None,
            'Det Pressure': None,
            'Hf solid': None
        }
        
        # Step 1: Convert SMILES to name
        logger.info(f"Converting SMILES to name: {smiles[:50]}...")
        chemical_name, name_source = self._get_chemical_name(smiles)
        
        if not chemical_name:
            # Could not get name from either source
            logger.debug(f"Could not generate name for: {smiles[:30]}...")
            empty = RAGResult(
                smiles=smiles,
                chemical_name=None,
                properties=properties,
                papers_searched=0,
                papers_with_hits=0
            )
            if self.cache is not None:
                self.cache.put(smiles, empty)
            return empty
        
        # Show where the name came from
        logger.info(f"Chemical name: {chemical_name} (via {name_source})")
        if name_source == "PubChem":
            print(f"         📚 Known compound: '{chemical_name}'")
        else:
            print(f"         🧪 Novel compound (IUPAC): '{chemical_name}'")
        
        # Step 2: Search literature
        logger.info(f"Searching literature databases for: {chemical_name}")
        print(f"         📖 Searching literature...", end=" ", flush=True)
        papers = self.searcher.search(chemical_name, smiles)
        
        papers_searched = len(papers)
        papers_with_hits = 0
        citations = []
        
        # Step 3: Extract properties from each paper
        for paper in papers:
            text = paper.get('text', '')
            title = paper.get('title', '')
            is_full_text = paper.get('has_full_text', False)
            
            if not text:
                continue
            
            extracted = self.extractor.extract_from_text(text, chemical_name, is_full_text=is_full_text)
            
            # Track which properties this paper contributed
            props_from_this_paper = []
            
            # Merge extracted properties (keep highest confidence)
            found_any = False
            for prop_name, prop_value in extracted.items():
                if prop_value is not None:
                    found_any = True
                    current = properties.get(prop_name)
                    if current is None or prop_value.confidence > current.confidence:
                        # Update source with paper title
                        prop_value = RetrievedProperty(
                            value=prop_value.value,
                            source=f"{title[:50]}..." if len(title) > 50 else title,
                            confidence=prop_value.confidence
                        )
                        properties[prop_name] = prop_value
                        props_from_this_paper.append(prop_name)
            
            if found_any:
                papers_with_hits += 1
                # Create citation for this paper
                citation = PaperCitation(
                    title=title,
                    authors=paper.get('authors', []),
                    doi=paper.get('doi', ''),
                    source_db=paper.get('source', 'Unknown'),
                    properties_found=props_from_this_paper
                )
                citations.append(citation)
        
        # Step 4: Analogue fallback — if some properties are still missing,
        # search literature for the most similar known energetic compounds.
        missing = [k for k, v in properties.items() if v is None]
        if missing:
            analogue_props, analogue_cites, analogues_searched = \
                self._search_analogues(smiles, chemical_name, missing)
            papers_searched += analogues_searched
            for prop_name, prop_value in analogue_props.items():
                if properties.get(prop_name) is None and prop_value is not None:
                    properties[prop_name] = prop_value
            if analogue_cites:
                citations.extend(analogue_cites)
                papers_with_hits += len(analogue_cites)

        result = RAGResult(
            smiles=smiles,
            chemical_name=chemical_name,
            properties=properties,
            papers_searched=papers_searched,
            papers_with_hits=papers_with_hits,
            citations=citations
        )

        found_props = [k for k, v in properties.items() if v is not None]
        logger.info(f"RAG found {len(found_props)}/4 properties: {found_props}")

        if found_props:
            print(f"found {len(found_props)} properties!")
            print(f"            ✅ Literature values: {', '.join(found_props)}")
        else:
            print(f"no property values found")

        if self.cache is not None:
            self.cache.put(smiles, result)

        return result

    def _suggest_analogues_via_llm(self, smiles: str,
                                    chemical_name: Optional[str],
                                    top_k: int = 3) -> List[Tuple[str, str]]:
        """Ask the LLM to name known energetic compounds structurally similar to
        this molecule. Returns a list of (common_name, rationale) pairs.

        Always uses ``gpt-4o-mini`` and is gated on ``self.extractor.llm_api_key``
        — independent of the ``use_llm`` property-extraction flag so that
        analogue lookup works cheaply even when LLM extraction is off.
        """
        try:
            client, model = _make_llm_client(
                self._openai_api_key, self._ollama_base_url, self._ollama_model)
            if client is None:
                return []

            display_name = chemical_name or smiles

            # Derive a human-readable structural summary for the system prompt
            # so the model can reason about functional groups without re-parsing SMILES.
            system = (
                "You are a world-leading expert in energetic materials chemistry "
                "with deep knowledge of synthesis, crystal structures, and detonation "
                "physics. You can read SMILES fluently and reason about functional "
                "groups, ring systems, and substituent effects on detonation properties."
            )

            prompt = f"""A novel energetic compound has been designed:

  SMILES : {smiles}
  Name   : {display_name}

This compound has no direct literature entry.  Your task is to identify the
{top_k} *most structurally similar* known energetic compounds that:
  1. Have experimentally measured detonation properties (density ρ, detonation
     velocity D, detonation pressure P_CJ, and/or heat of formation ΔHf) published
     in peer-reviewed journals or technical reports.
  2. Share as many of the following features as possible with the target:
       • Same or closely related ring system / cage scaffold
       • Same nitrogen-containing functional groups (N-NO₂, O-NO₂, C-NO₂, N₃, C=N,
         tetrazole, triazole, furazan, tetrazine, oxetane, etc.)
       • Similar N/O balance and oxygen balance
       • Similar molecular size (atom count within ±30 %)
       • Same heteroatom substitution pattern (halogen, azide, amino, etc.)
  3. Are primarily known by a short common name or acronym used in the explosives
     literature (e.g. RDX, HMX, CL-20, TATB, FOX-7, NTO, ADN, PETN).

Ranking rules (most important first):
  a. Maximise the number of shared structural motifs listed above.
  b. Prefer same scaffold/ring over same substituents.
  c. Include exotic or lesser-known compounds (e.g. TNAZ, TEX, BCHMX, DINGU,
     LLM-105, FOX-12, TKX-50, CL-14, MAD-X1, DNTF, BTATz, BTF, DAAF, ADN,
     FEFO, BNFF, NTO, ANTA) if they are genuinely closer than the common RDX/HMX.
  d. Do NOT default to RDX and HMX if a structurally tighter match exists.

For each compound return:
  "name"       — the primary acronym or common name as used in search engines
  "reason"     — one sentence listing the specific shared features (ring size,
                  functional groups, N-count, etc.)
  "similarity" — your estimated structural similarity 0-1 (1 = identical)

Respond with ONLY a JSON array, no prose, no markdown fences:
[
  {{"name": "HMX", "reason": "8-membered nitramine ring with 4 N-NO2 groups, same scaffold as the target", "similarity": 0.72}},
  {{"name": "TNAZ", "reason": "cyclic nitramine with gem-dinitro group and 4-membered ring", "similarity": 0.55}}
]"""

            def _call(messages):
                return client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=500,
                )

            response = _call([
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': prompt},
            ])
            choice = response.choices[0]
            finish = getattr(choice, 'finish_reason', 'unknown')
            raw = (choice.message.content or '').strip()

            if not raw:
                logger.warning(
                    f"LLM analogue suggestion: empty response "
                    f"(finish_reason={finish!r}) — likely content filter. "
                    "Retrying with neutral chemistry prompt.")
                neutral = (
                    f"You are a computational chemist specialising in "
                    f"nitrogen-rich heterocyclic compounds and dense organic crystals.\n\n"
                    f"Molecule SMILES: {smiles}\n"
                    f"IUPAC name: {display_name}\n\n"
                    f"List the {top_k} most structurally similar *well-characterised* "
                    f"reference compounds from academic literature for which crystal "
                    f"density, heat of formation, and Chapman-Jouguet parameters are "
                    f"published. Prioritise shared scaffold (ring size, cage motif), "
                    f"then functional groups (N-NO₂, O-NO₂, C-NO₂, azide, tetrazole, "
                    f"triazole, furazan, tetrazine). Prefer close but less-obvious "
                    f"matches over defaulting to RDX/HMX when a tighter analogue exists.\n\n"
                    f"Reply ONLY with a JSON array, no prose:\n"
                    f'[{{"name":"<acronym>","reason":"<shared features>","similarity":<0-1>}}]'
                )
                response = _call([{'role': 'user', 'content': neutral}])
                choice = response.choices[0]
                finish = getattr(choice, 'finish_reason', 'unknown')
                raw = (choice.message.content or '').strip()
                if not raw:
                    logger.warning(
                        f"LLM analogue suggestion: still empty after retry "
                        f"(finish_reason={finish!r}). Falling back to static library.")
                    return []

            # Strip <think>…</think> reasoning blocks (Qwen, DeepSeek, etc.)
            raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
            arr_match = re.search(r'\[.*?\]', raw, re.DOTALL)
            if not arr_match:
                logger.warning(
                    f"LLM analogue suggestion: no JSON array "
                    f"(finish_reason={finish!r}). "
                    f"Raw (first 400 chars): {raw[:400]!r}")
                return []
            text = arr_match.group(0)
            data = json.loads(text)
            results = []
            for item in data:
                name = str(item.get('name', '')).strip()
                reason = str(item.get('reason', '')).strip()
                # Clamp model-provided similarity to [0.1, 1.0].
                try:
                    sim = float(item.get('similarity', 0.75))
                    sim = max(0.1, min(1.0, sim))
                except (TypeError, ValueError):
                    sim = 0.75
                if name:
                    results.append((name, reason, sim))
            logger.info(f"LLM suggested analogues: "
                        f"{[(n, f'{s:.2f}') for n, _, s in results]}")
            return results[:top_k]
        except Exception as e:
            logger.warning(f"LLM analogue suggestion failed: {e}")
            return []

    def _search_analogues(self, smiles: str, chemical_name: Optional[str],
                          missing_props: List[str],
                          top_k: int = 3,
                          conf_penalty: float = 0.5,
                          ) -> Tuple[Dict[str, 'RetrievedProperty'],
                                     List['PaperCitation'], int]:
        """Analogue-fallback pipeline.

        1. If an OpenAI key is present, ask the LLM to suggest known analogues.
        2. Otherwise fall back to Tanimoto nearest-neighbours from the static
           library in energetic_library.py.

        Property values from analogues get confidence multiplied by
        ``analogue_similarity ** conf_penalty`` so the XGBoost predictor can
        outrank very weak analogue evidence.
        """
        # --- Step 1: resolve analogue list -----------------------------------
        # Each entry: (display_name, similarity_score, source_tag)
        candidate_names: List[Tuple[str, float, str]] = []

        llm_suggestions = self._suggest_analogues_via_llm(smiles, chemical_name, top_k)
        if llm_suggestions:
            print(f"         🤖 LLM suggested analogues: "
                  f"{', '.join(f'{n} ({s:.2f})' for n, _, s in llm_suggestions)}",
                  flush=True)
            for name, reason, sim in llm_suggestions:
                candidate_names.append((name, sim, f"LLM sim={sim:.2f} ({reason[:55]})"))
        else:
            # Static Tanimoto fallback
            try:
                from .energetic_library import find_similar
                for compound, tanimoto in find_similar(smiles, top_k=top_k,
                                                       min_tanimoto=0.30):
                    candidate_names.append(
                        (compound.name, tanimoto,
                         f"Tanimoto={tanimoto:.2f}"))
            except Exception as e:
                logger.debug(f"Static library fallback failed: {e}")

        if not candidate_names:
            return {}, [], 0

        # --- Step 2: search + extract for each analogue ----------------------
        filled: Dict[str, RetrievedProperty] = {}
        analogue_citations: List[PaperCitation] = []
        papers_searched = 0
        still_missing = list(missing_props)

        for analogue_name, similarity, source_tag in candidate_names:
            if not still_missing:
                break

            norm = analogue_name.strip().lower()

            # --- Cache lookup (memory first, then SQLite) --------------------
            cached = self._analogue_mem.get(norm)
            if cached is None and self.cache is not None:
                cached = self.cache.get_analogue(analogue_name)
                if cached is not None:
                    self._analogue_mem[norm] = cached  # promote to memory

            if cached is not None:
                cached_props, cached_cites, cached_n = cached
                print(f"         💾 Analogue cache hit: {analogue_name} "
                      f"({len(cached_props)} props)", flush=True)
                papers_searched += cached_n
                for prop_name in list(still_missing):
                    pv = cached_props.get(prop_name)
                    if pv is None:
                        continue
                    scaled_conf = max(0.01,
                                     pv.confidence * (similarity ** conf_penalty))
                    existing = filled.get(prop_name)
                    if existing is not None and existing.confidence >= scaled_conf:
                        continue
                    filled[prop_name] = RetrievedProperty(
                        value=pv.value,
                        source=pv.source,
                        confidence=scaled_conf,
                    )
                for c in cached_cites:
                    if c not in analogue_citations:
                        analogue_citations.append(c)
                still_missing = [p for p in still_missing if p not in filled]
                if filled:
                    print(f"            ✅ {analogue_name} (cached): "
                          f"{', '.join(sorted(filled.keys()))}")
                continue

            # --- Cache miss: fetch from literature ---------------------------
            print(f"         🔗 Analogue: {analogue_name} [{source_tag}] "
                  f"→ searching literature…", flush=True)
            papers = self.searcher.search(analogue_name, None)
            n_papers = len(papers)
            papers_searched += n_papers

            raw_props: Dict[str, RetrievedProperty] = {}
            raw_cites: List[PaperCitation] = []

            for paper in papers:
                text = paper.get('text', '')
                if not text:
                    continue
                title = paper.get('title', '')
                is_full_text = paper.get('has_full_text', False)
                extracted = self.extractor.extract_from_text(
                    text, analogue_name, is_full_text=is_full_text)

                hits_here: List[str] = []
                for prop_name, pv in extracted.items():
                    if pv is None:
                        continue
                    existing = raw_props.get(prop_name)
                    if existing is None or pv.confidence > existing.confidence:
                        raw_props[prop_name] = RetrievedProperty(
                            value=pv.value,
                            source=f"analogue: {analogue_name} [{source_tag}] — {title[:40]}",
                            confidence=pv.confidence,
                        )
                        hits_here.append(prop_name)

                if hits_here:
                    raw_cites.append(PaperCitation(
                        title=f"[Analogue {analogue_name}] {title}",
                        authors=paper.get('authors', []),
                        doi=paper.get('doi', ''),
                        source_db=paper.get('source', 'Unknown'),
                        properties_found=hits_here,
                    ))

            # Store in both caches so the next candidate skips the download.
            self._analogue_mem[norm] = (raw_props, raw_cites, n_papers)
            if self.cache is not None:
                self.cache.put_analogue(analogue_name, raw_props,
                                        raw_cites, n_papers)

            # Merge into filled with similarity-scaled confidence.
            for prop_name in list(still_missing):
                pv = raw_props.get(prop_name)
                if pv is None:
                    continue
                scaled_conf = max(0.01, pv.confidence * (similarity ** conf_penalty))
                existing = filled.get(prop_name)
                if existing is not None and existing.confidence >= scaled_conf:
                    continue
                filled[prop_name] = RetrievedProperty(
                    value=pv.value,
                    source=pv.source,
                    confidence=scaled_conf,
                )
            analogue_citations.extend(raw_cites)

            still_missing = [p for p in still_missing if p not in filled]
            if filled:
                print(f"            ✅ {analogue_name} filled: "
                      f"{', '.join(sorted(filled.keys()))}")

        return filled, analogue_citations, papers_searched


def get_properties_with_rag(smiles: str, 
                            predictor,
                            use_rag: bool = True,
                            use_llm: bool = False) -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    Get properties for a molecule, using RAG first then ML prediction for missing values.
    
    Args:
        smiles: SMILES string
        predictor: PropertyPredictor instance for ML fallback
        use_rag: Whether to use RAG retrieval
        use_llm: Whether to use LLM for extraction
        
    Returns:
        Tuple of (properties dict, sources dict)
        - properties: {'Density': 1.9, 'Det Velocity': 9000, ...}
        - sources: {'Density': 'literature', 'Det Velocity': 'predicted', ...}
    """
    properties = {}
    sources = {}
    
    if use_rag:
        # Try RAG retrieval first
        rag = RAGPropertyRetriever(
            use_llm=use_llm
        )
        
        rag_result = rag.retrieve_properties(smiles)
        
        # Collect found properties
        for prop_name, prop_value in rag_result.properties.items():
            if prop_value is not None:
                properties[prop_name] = prop_value.value
                sources[prop_name] = f"literature ({prop_value.source})"
    
    # Get missing properties from ML predictor
    missing_props = [p for p in ['Density', 'Det Velocity', 'Det Pressure', 'Hf solid'] 
                    if p not in properties]
    
    if missing_props and predictor is not None:
        predicted = predictor.predict_properties(smiles)
        if predicted:
            for prop_name in missing_props:
                if prop_name in predicted and predicted[prop_name] is not None:
                    properties[prop_name] = predicted[prop_name]
                    sources[prop_name] = "predicted (XGBoost)"
    
    return properties, sources
