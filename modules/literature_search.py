"""
Literature search module for energetic property lookup.

Pipeline:
1. Convert SMILES to chemical name (PubChem / SMILES2IUPAC)
2. Search 4 academic APIs (OpenAlex, ArXiv, Crossref, Semantic Scholar)
3. Download up to 10 relevant PDFs; discard papers with no full text
   AND no candidate-name mention in the abstract
4. Extract name-centered chunks (paragraph before + match + paragraph after)
5. Extract properties via LLM (primary) or regex (fallback)
6. For novel molecules with missing props, LLM suggests structurally
   similar known compounds and the pipeline repeats on those
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

try:
    import fitz  # PyMuPDF
    PDF_PARSER_AVAILABLE = True
except ImportError as e:
    PDF_PARSER_AVAILABLE = False
    logging.getLogger(__name__).warning(f"PyMuPDF not installed: {e}. Install with: pip install pymupdf")
except Exception as e:
    PDF_PARSER_AVAILABLE = False
    logging.getLogger(__name__).warning(f"PyMuPDF import failed: {e}")

logger = logging.getLogger(__name__)

PROP_KEYS = ['Density', 'Det Velocity', 'Det Pressure', 'Hf solid']

# ──────────────────────────────────────────────────────────────── data classes

@dataclass
class RetrievedProperty:
    """A property value retrieved from literature."""
    value: float
    source: str
    confidence: float


@dataclass
class PaperCitation:
    """Citation information for a paper that provided property data."""
    title: str
    authors: List[str]
    doi: str
    source_db: str
    properties_found: List[str]


@dataclass
class LiteratureResult:
    """Result of literature property lookup."""
    smiles: str
    chemical_name: Optional[str]
    properties: Dict[str, Optional[RetrievedProperty]]
    papers_searched: int
    papers_with_hits: int
    citations: List[PaperCitation] = None

    def __post_init__(self):
        if self.citations is None:
            self.citations = []


# ──────────────────────────────────────────────────────── SMILES→Name converter

class SMILESToNameConverter:
    """Converts SMILES to chemical names (PubChem → SMILES2IUPAC fallback)."""

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self._pubchempy_available = False
        self._smiles2iupac_available = False
        self._iupac_converter = None

        try:
            import pubchempy as pcp
            self._pcp = pcp
            self._pubchempy_available = True
            logger.info("PubChemPy available for SMILES-to-name conversion")
        except ImportError:
            logger.warning("pubchempy not installed. Install with: pip install pubchempy")

        try:
            from chemicalconverters import NamesConverter
            self._iupac_converter = NamesConverter(model_name="knowledgator/SMILES2IUPAC-canonical-base")
            self._smiles2iupac_available = True
            logger.info("SMILES2IUPAC model available for IUPAC name generation")
        except ImportError:
            logger.warning("chemical-converters not installed. Install with: pip install chemical-converters")
        except Exception as e:
            logger.warning(f"Failed to load SMILES2IUPAC model: {e}")

    def convert(self, smiles: str) -> Optional[str]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        canonical_smiles = Chem.MolToSmiles(mol)
        if self._pubchempy_available:
            name = self._query_pubchempy(canonical_smiles)
            if name:
                return name
        if self._smiles2iupac_available:
            name = self._generate_iupac(canonical_smiles)
            if name:
                return name
        return None

    def _generate_iupac(self, smiles: str) -> Optional[str]:
        try:
            result = self._iupac_converter.smiles_to_iupac(f"<BASE>{smiles}")
            if result and len(result) > 0:
                name = result[0]
                if name and name != smiles and len(name) > 0:
                    return name
            return None
        except Exception as e:
            logger.debug(f"SMILES2IUPAC failed for {smiles[:30]}...: {e}")
            return None

    def _query_pubchempy(self, smiles: str) -> Optional[str]:
        try:
            compounds = self._pcp.get_compounds(smiles, 'smiles')
            if not compounds:
                return None
            compound = compounds[0]
            synonyms = compound.synonyms
            if synonyms:
                good = [
                    s for s in synonyms[:10]
                    if not s.startswith(('CID', 'CHEMBL', 'SCHEMBL', 'DTXSID', 'EINECS'))
                    and len(s) < 100
                ]
                if good:
                    return min(good, key=len)
            if compound.iupac_name:
                return compound.iupac_name
            if synonyms:
                return synonyms[0]
            return None
        except Exception as e:
            logger.warning(f"PubChemPy error for {smiles[:30]}...: {e}")
            return None


# ──────────────────────────────────────────────── PDF download & text extraction

def _download_pdf(url: str, timeout: int = 30) -> Optional[str]:
    """Download a PDF from *url* and return extracted full text, or None."""
    if not PDF_PARSER_AVAILABLE:
        return None
    try:
        headers = {
            'User-Agent': 'EnergeticMoleculeDesigner/1.0 (mailto:research@example.com)',
            'Accept': 'application/pdf',
        }
        resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        ct = resp.headers.get('Content-Type', '')
        if resp.status_code != 200:
            return None
        # Only parse if content looks like a PDF
        if not ct.startswith('application/pdf') and not resp.content[:5] == b'%PDF-':
            return None
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(resp.content)
            tmp_path = tmp.name
        try:
            doc = fitz.open(tmp_path)
            text = ''.join(page.get_text() for page in doc)
            doc.close()
            os.unlink(tmp_path)
            return text if text.strip() else None
        except Exception as e:
            logger.debug(f"PDF parse error: {e}")
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            return None
    except Exception as e:
        logger.debug(f"PDF download error ({url[:80]}): {e}")
        return None


# ──────────────────────────────────────────────── Academic API searcher

HEADERS = {
    'User-Agent': 'EnergeticMoleculeDesigner/1.0 (mailto:research@example.com)',
    'Accept': 'application/json',
}

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


def _get_search_terms(chemical_name: str) -> str:
    name_lower = chemical_name.lower()
    for full_name, abbrev in SEARCH_ALIASES.items():
        if full_name in name_lower or name_lower in full_name:
            return f'("{abbrev}" OR "{chemical_name}")'
    return f'"{chemical_name}"'


class AcademicSearcher:
    """Searches OpenAlex, ArXiv, Crossref, and Semantic Scholar.

    Each method returns a list of paper dicts with at least:
        title, text, doi, authors, source, pdf_url (may be None),
        has_full_text (bool).
    PDF downloading is handled externally; the searcher only provides
    ``pdf_url`` when the API exposes one.
    """

    def __init__(self, max_results: int = 10, timeout: int = 15):
        self.max_results = max_results
        self.timeout = timeout

    def search(self, chemical_name: str, smiles: str = None) -> List[Dict]:
        """Return up to *max_results* deduplicated papers."""
        papers = []
        search_term = _get_search_terms(chemical_name)

        arxiv = self._search_arxiv(chemical_name, search_term)
        papers.extend(arxiv)
        logger.info(f"ArXiv: {len(arxiv)} papers for '{chemical_name}'")

        if len(papers) < self.max_results:
            oa = self._search_openalex(chemical_name, search_term)
            papers.extend(oa)
            logger.info(f"OpenAlex: {len(oa)} papers")

        if len(papers) < self.max_results:
            cr = self._search_crossref(chemical_name, search_term)
            papers.extend(cr)
            logger.info(f"Crossref: {len(cr)} papers")

        if len(papers) < self.max_results // 2:
            ss = self._search_semantic_scholar(chemical_name, search_term)
            papers.extend(ss)
            logger.info(f"Semantic Scholar: {len(ss)} papers")

        seen_dois = set()
        unique = []
        no_doi_count = 0
        for p in papers:
            doi = p.get('doi', '')
            if doi and doi not in seen_dois:
                seen_dois.add(doi)
                unique.append(p)
            elif not doi and no_doi_count < 3:
                no_doi_count += 1
                unique.append(p)
        return unique[:self.max_results]

    # ── ArXiv ──────────────────────────────────────────────────────────────

    def _search_arxiv(self, chemical_name: str, search_term: str = None) -> List[Dict]:
        papers = []
        try:
            if search_term and 'OR' in search_term:
                m = re.search(r'"([^"]+)"', search_term)
                base = m.group(1) if m else chemical_name
            else:
                base = chemical_name

            energetic = '(all:energetic OR all:detonation OR all:explosive)'
            if base.lower() != chemical_name.lower():
                q = f'(all:"{base}" OR all:"{chemical_name}") AND {energetic}'
            else:
                q = f'all:"{chemical_name}" AND {energetic}'

            print(f"         🔍 ArXiv query: {base} (PDF parsing: {'✓' if PDF_PARSER_AVAILABLE else '✗'})...",
                  end=" ", flush=True)

            resp = requests.get("http://export.arxiv.org/api/query",
                                params={'search_query': q,
                                        'max_results': min(self.max_results, 10),
                                        'sortBy': 'relevance',
                                        'sortOrder': 'descending'},
                                headers=HEADERS, timeout=self.timeout)
            if resp.status_code == 200:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(resp.content)
                ns = {'atom': 'http://www.w3.org/2005/Atom'}
                entries = root.findall('atom:entry', ns)
                print(f"found {len(entries)} papers")
                for entry in entries:
                    t = entry.find('atom:title', ns)
                    title = t.text.strip() if t is not None and t.text else ''
                    s = entry.find('atom:summary', ns)
                    abstract = s.text.strip() if s is not None and s.text else ''
                    authors = [
                        a.find('atom:name', ns).text.strip()
                        for a in entry.findall('atom:author', ns)
                        if a.find('atom:name', ns) is not None
                        and a.find('atom:name', ns).text
                    ]
                    id_el = entry.find('atom:id', ns)
                    arxiv_id = ''
                    if id_el is not None and id_el.text:
                        arxiv_id = id_el.text.replace('http://arxiv.org/abs/', '').replace('https://arxiv.org/abs/', '')
                    pub = entry.find('atom:published', ns)
                    pub_date = pub.text[:10] if pub is not None and pub.text else ''
                    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else None
                    if title and (abstract or pdf_url):
                        papers.append({
                            'title': title, 'text': abstract, 'doi': f'arXiv:{arxiv_id}' if arxiv_id else '',
                            'authors': authors, 'published_date': pub_date,
                            'source': 'ArXiv', 'pdf_url': pdf_url, 'has_full_text': False,
                        })
            else:
                print(f"error (status {resp.status_code})")
        except requests.exceptions.Timeout:
            print("timeout")
        except Exception as e:
            logger.warning(f"ArXiv error: {e}")
        return papers

    # ── OpenAlex ───────────────────────────────────────────────────────────

    def _search_openalex(self, chemical_name: str, search_term: str = None) -> List[Dict]:
        papers = []
        try:
            base = search_term if search_term else chemical_name
            resp = requests.get("https://api.openalex.org/works",
                                params={
                                    'search': f'{base} energetic explosive detonation',
                                    'per_page': min(self.max_results, 25),
                                    'filter': 'type:article',
                                    'select': 'id,doi,title,abstract_inverted_index,authorships,publication_date,open_access',
                                },
                                headers=HEADERS, timeout=self.timeout)
            if resp.status_code == 200:
                for item in resp.json().get('results', []):
                    abstract = self._reconstruct_abstract(item.get('abstract_inverted_index', {}))
                    authors = [
                        a.get('author', {}).get('display_name', '')
                        for a in item.get('authorships', [])
                        if a.get('author', {}).get('display_name')
                    ]
                    oa = item.get('open_access', {}) or {}
                    pdf_url = oa.get('oa_url') or None
                    doi_raw = item.get('doi', '') or ''
                    doi = doi_raw.replace('https://doi.org/', '') if doi_raw else ''
                    if item.get('title'):
                        papers.append({
                            'title': item['title'], 'text': abstract, 'doi': doi,
                            'authors': authors, 'published_date': item.get('publication_date', ''),
                            'source': 'OpenAlex', 'pdf_url': pdf_url, 'has_full_text': False,
                        })
        except requests.exceptions.Timeout:
            logger.warning(f"OpenAlex timeout for '{chemical_name}'")
        except Exception as e:
            logger.warning(f"OpenAlex error: {e}")
        return papers

    @staticmethod
    def _reconstruct_abstract(inv_idx: dict) -> str:
        if not inv_idx:
            return ''
        try:
            positions = []
            for word, poses in inv_idx.items():
                for p in poses:
                    positions.append((p, word))
            positions.sort()
            return ' '.join(w for _, w in positions)
        except Exception:
            return ''

    # ── Crossref ───────────────────────────────────────────────────────────

    def _search_crossref(self, chemical_name: str, search_term: str = None) -> List[Dict]:
        papers = []
        try:
            base = search_term.replace('"', '').replace('(', '').replace(')', '') if search_term else chemical_name
            resp = requests.get("https://api.crossref.org/works",
                                params={
                                    'query': f'{base} energetic explosive detonation',
                                    'rows': min(10, self.max_results),
                                    'filter': 'type:journal-article',
                                },
                                headers=HEADERS, timeout=self.timeout)
            if resp.status_code == 200:
                for item in resp.json().get('message', {}).get('items', []):
                    abstract = re.sub(r'<[^>]+>', '', item.get('abstract', '') or '')
                    links = item.get('link', []) or []
                    pdf_url = None
                    for lnk in links:
                        if lnk.get('content-type', '') == 'application/pdf':
                            pdf_url = lnk.get('URL')
                            break
                    title = item.get('title', [''])[0] if item.get('title') else ''
                    if title:
                        papers.append({
                            'title': title, 'text': abstract,
                            'doi': item.get('DOI', ''),
                            'authors': [f"{a.get('given', '')} {a.get('family', '')}"
                                        for a in item.get('author', [])],
                            'published_date': str(item.get('published-print', {}).get('date-parts', [['']])[0]),
                            'source': 'Crossref', 'pdf_url': pdf_url, 'has_full_text': False,
                        })
        except requests.exceptions.Timeout:
            logger.debug("Crossref timeout")
        except Exception as e:
            logger.debug(f"Crossref error: {e}")
        return papers

    # ── Semantic Scholar ───────────────────────────────────────────────────

    def _search_semantic_scholar(self, chemical_name: str, search_term: str = None) -> List[Dict]:
        papers = []
        try:
            base = search_term.replace('"', '').replace('(', '').replace(')', '') if search_term else chemical_name
            resp = requests.get("https://api.semanticscholar.org/graph/v1/paper/search",
                                params={
                                    'query': f'{base} energetic material',
                                    'limit': min(10, self.max_results),
                                    'fields': 'title,abstract,authors,externalIds,publicationDate,openAccessPdf',
                                },
                                headers=HEADERS, timeout=self.timeout)
            if resp.status_code == 200:
                for item in resp.json().get('data', []):
                    ext = item.get('externalIds', {}) or {}
                    oa_pdf = item.get('openAccessPdf', {}) or {}
                    pdf_url = oa_pdf.get('url') or None
                    title = item.get('title', '')
                    abstract = item.get('abstract', '') or ''
                    if title and abstract:
                        papers.append({
                            'title': title, 'text': abstract,
                            'doi': ext.get('DOI', ''),
                            'authors': [a.get('name', '') for a in (item.get('authors') or [])],
                            'published_date': item.get('publicationDate', ''),
                            'source': 'SemanticScholar', 'pdf_url': pdf_url, 'has_full_text': False,
                        })
        except requests.exceptions.Timeout:
            logger.debug("Semantic Scholar timeout")
        except Exception as e:
            logger.debug(f"Semantic Scholar error: {e}")
        return papers


# ──────────────────────────────────────────────── Name-centered chunking

ENERGETIC_ALIASES = {
    'trinitrotoluene': ['tnt', '2,4,6-tnt', 'trinitrotoluene'],
    'rdx': ['rdx', 'cyclotrimethylenetrinitramine', 'hexogen', 'cyclonite'],
    'hmx': ['hmx', 'cyclotetramethylenetetranitramine', 'octogen'],
    'tatb': ['tatb', 'triaminotrinitrobenzene'],
    'petn': ['petn', 'pentaerythritol tetranitrate', 'nitropenta'],
    'nitroglycerin': ['nitroglycerin', 'nitroglycerine', 'glyceryl trinitrate'],
    'picric acid': ['picric acid', 'trinitrophenol'],
    'tetryl': ['tetryl', 'trinitrophenylmethylnitramine'],
    'cl-20': ['cl-20', 'hexanitrohexaazaisowurtzitane', 'hniw'],
    'fox-7': ['fox-7', 'dadne', '1,1-diamino-2,2-dinitroethylene'],
    'dnt': ['dnt', 'dinitrotoluene', '2,4-dinitrotoluene'],
    'tnaz': ['tnaz', 'trinitroazetidine'],
    'nto': ['nto', 'nitrotriazolone', '3-nitro-1,2,4-triazol-5-one'],
    'dnan': ['dnan', 'dinitroanisole', '2,4-dinitroanisole'],
    'hns': ['hns', 'hexanitrostilbene'],
}


def _get_name_variants(chemical_name: str) -> List[str]:
    """All aliases/abbreviations for *chemical_name*."""
    if not chemical_name:
        return []
    low = chemical_name.lower()
    variants = [low]
    for _key, aliases in ENERGETIC_ALIASES.items():
        for a in aliases:
            if a in low or low in a:
                variants.extend(aliases)
                break
    base = re.sub(r'^[\d,\'-]+', '', low).strip()
    if base and base != low:
        variants.append(base)
    return list(set(variants))


def _name_in_text(text_lower: str, variants: List[str]) -> bool:
    return any(v in text_lower for v in variants)


def _extract_name_chunks(full_text: str, variants: List[str]) -> List[str]:
    """Return 3-paragraph windows centred on paragraphs mentioning *variants*.

    Paragraphs are split on double newlines (``\\n\\n``).  For each matching
    paragraph we take the preceding paragraph, the match itself, and the
    following paragraph, then de-duplicate overlapping windows.
    """
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', full_text) if p.strip()]
    if not paragraphs:
        return []

    hit_indices = set()
    for i, para in enumerate(paragraphs):
        if _name_in_text(para.lower(), variants):
            hit_indices.add(i)

    if not hit_indices:
        return []

    included = set()
    for idx in sorted(hit_indices):
        for offset in (-1, 0, 1):
            j = idx + offset
            if 0 <= j < len(paragraphs):
                included.add(j)

    ordered = sorted(included)
    chunks = []
    current: List[str] = []
    prev = -2
    for i in ordered:
        if current and i != prev + 1:
            chunks.append('\n\n'.join(current))
            current = []
        current.append(paragraphs[i])
        prev = i
    if current:
        chunks.append('\n\n'.join(current))

    return chunks


# ──────────────────────────────────────────────── Property extraction

PROPERTY_PATTERNS = {
    'Density': [
        r'(?:density|ρ|rho|crystal\s+density|calculated\s+density)\s*(?:of|=|:|is|was)?\s*(\d+\.?\d*)\s*(?:g\s*/?\s*cm|g\s*cm|gcc)',
        r'(\d+\.?\d*)\s*(?:g\s*/?\s*cm|g/cm|gcc|g\s*cm)\s*.*?(?:density)',
        r'density[^.]{0,50}(\d+\.\d+)\s*(?:g|gcc)',
        r'\(\s*(\d+\.\d+)\s*g\s*/?\s*cm',
    ],
    'Det Velocity': [
        r'(?:detonation\s+velocity|detonation\s+speed|explosion\s+velocity|explosion\s+speed|det\.?\s*vel\.?|vod|velocity\s+of\s+detonation)\s*(?:of|=|:|is|was)?\s*(\d+\.?\d*)\s*(?:m\s*/?\s*s|m/s|ms)',
        r'(\d+\.?\d*)\s*(?:m\s*/?\s*s|m/s)\s*.*?(?:detonation|explosion|velocity|speed)',
        r'(?:detonation|explosion|velocity|speed)[^.]{0,30}(\d+\.?\d*)\s*(?:km\s*/?\s*s|km/s)',
        r'(?:detonation|explosion)[^.]{0,50}(\d{4,5})\s*(?:m/s|m\s*/\s*s)',
    ],
    'Det Pressure': [
        r'(?:detonation\s+pressure|explosion\s+pressure|det\.?\s*press\.?|pcj|p_cj|chapman.jouguet)\s*(?:of|=|:|is|was)?\s*(\d+\.?\d*)\s*(?:GPa|gpa)',
        r'(\d+\.?\d*)\s*(?:GPa|gpa)\s*.*?(?:detonation|explosion|pressure|pcj)',
        r'(?:detonation|explosion|pressure)[^.]{0,30}(\d+\.?\d*)\s*(?:kbar)',
        r'(?:detonation|explosion)[^.]{0,50}(\d+\.?\d*)\s*(?:GPa|gpa)',
    ],
    'Hf solid': [
        r'(?:heat\s+of\s+formation|enthalpy\s+of\s+formation|formation\s+enthalpy|hof|Δhf|ΔH)\s*(?:of|=|:|is|was)?\s*([-−+]?\d+\.?\d*)\s*(?:kJ|kj)',
        r'([-−+]?\d+\.?\d*)\s*(?:kJ\s*/?\s*mol|kJ/mol|kj/mol)\s*.*?(?:heat|enthalpy|formation)',
        r'(?:heat|enthalpy|formation)[^.]{0,30}([-−+]?\d+\.?\d*)\s*(?:kcal)',
        r'formation[^.]{0,50}([-−+]?\d+\.?\d*)\s*(?:kJ|kj)',
    ],
}

VALUE_RANGES = {
    'Density': (0.5, 3.0),
    'Det Velocity': (4000, 12000),
    'Det Pressure': (10, 60),
    'Hf solid': (-500, 1000),
}


def _validate_value(prop_name: str, value: float) -> bool:
    lo, hi = VALUE_RANGES.get(prop_name, (float('-inf'), float('inf')))
    return lo <= value <= hi


def _extract_regex(text: str, chemical_name: str, variants: List[str]) -> Dict[str, Optional[RetrievedProperty]]:
    """Regex extraction with name-proximity gating."""
    props: Dict[str, Optional[RetrievedProperty]] = {k: None for k in PROP_KEYS}
    if not text:
        return props
    text_lower = text.lower()
    name_found = _name_in_text(text_lower, variants)
    if not name_found and not any(kw in text_lower for kw in ('energetic', 'explosive', 'detonation', 'propellant')):
        return props

    proximity = 400
    for prop_name, patterns in PROPERTY_PATTERNS.items():
        for pat in patterns:
            for m in re.finditer(pat, text_lower, re.IGNORECASE):
                try:
                    val = float(m.group(1).replace('−', '-').replace('–', '-'))
                    if 'km' in pat and prop_name == 'Det Velocity':
                        val *= 1000
                    elif 'kbar' in pat and prop_name == 'Det Pressure':
                        val *= 0.1
                    elif 'kcal' in pat and prop_name == 'Hf solid':
                        val *= 4.184
                    if not _validate_value(prop_name, val):
                        continue
                    if name_found and variants:
                        win = text_lower[max(0, m.start() - proximity):min(len(text_lower), m.end() + proximity)]
                        if not any(v in win for v in variants):
                            continue
                    props[prop_name] = RetrievedProperty(value=val, source="Regex extraction", confidence=0.7)
                    break
                except (ValueError, IndexError):
                    continue
            if props[prop_name] is not None:
                break
    return props


# ──────────────────────────────────────────────── LLM helpers

def _make_llm_client(openai_api_key: Optional[str],
                     ollama_base_url: Optional[str],
                     ollama_model: Optional[str]):
    """Return (client, model_name) for whichever LLM backend is configured."""
    try:
        import openai
    except ImportError:
        return None, None

    if ollama_base_url:
        base = ollama_base_url.strip().rstrip('/')
        if not base.startswith(('http://', 'https://')):
            base = 'http://' + base
        if not base.endswith('/v1'):
            base = f"{base}/v1"
        return openai.OpenAI(base_url=base, api_key='ollama'), (ollama_model or 'ALIENTELLIGENCE/chemicalengineer')

    if openai_api_key:
        return openai.OpenAI(api_key=openai_api_key), 'gpt-4o-mini'

    return None, None


def _extract_with_llm(text: str, chemical_name: str,
                      client, model: str) -> Dict[str, Optional[RetrievedProperty]]:
    """Use LLM to extract properties from *text*."""
    props: Dict[str, Optional[RetrievedProperty]] = {k: None for k in PROP_KEYS}
    try:
        prompt = f"""Extract energetic material properties for "{chemical_name}" from the following text.

Text:
{text[:6000]}

Search for these properties using any of the listed synonyms:
- "density": crystal density, calculated density, ρ (in g/cm³)
- "det_velocity": detonation velocity, detonation speed, explosion velocity, VOD (in m/s)
- "det_pressure": detonation pressure, explosion pressure, Chapman-Jouguet pressure, PCJ (in GPa)
- "heat_of_formation": heat of formation, enthalpy of formation, ΔHf (in kJ/mol)

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
            max_tokens=200,
        )

        raw = response.choices[0].message.content or ''
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        json_match = re.search(r'\{[^{}]*\}', raw)
        if json_match:
            data = json.loads(json_match.group())
            mapping = {
                'density': 'Density',
                'det_velocity': 'Det Velocity',
                'det_pressure': 'Det Pressure',
                'heat_of_formation': 'Hf solid',
            }
            for jk, pk in mapping.items():
                v = data.get(jk)
                if v is not None:
                    try:
                        fv = float(v)
                    except (TypeError, ValueError):
                        continue
                    if _validate_value(pk, fv):
                        props[pk] = RetrievedProperty(value=fv, source="LLM extraction", confidence=0.85)
    except Exception as e:
        logger.warning(f"LLM extraction failed: {e}")
    return props


# ──────────────────────────────────────────────── Property extractor (unified)

class PropertyExtractor:
    """Extracts energetic properties from text.

    When an LLM backend is configured it is the **primary** extraction
    method. Regex patterns serve as fallback when no LLM is available.
    """

    def __init__(self, use_llm: bool = False,
                 llm_api_key: str = None,
                 ollama_base_url: Optional[str] = None,
                 ollama_model: Optional[str] = None):
        self.llm_api_key = llm_api_key or os.getenv('OPENAI_API_KEY')
        self.ollama_base_url = ollama_base_url or os.getenv('OLLAMA_BASE_URL')
        self.ollama_model = ollama_model or os.getenv('OLLAMA_MODEL', 'ALIENTELLIGENCE/chemicalengineer')
        has_backend = bool(self.ollama_base_url or self.llm_api_key)
        self.use_llm = bool(use_llm and has_backend)
        if use_llm and not has_backend:
            logger.warning("LLM extraction requested but no backend configured; using regex only.")

    def extract(self, text: str, chemical_name: str) -> Dict[str, Optional[RetrievedProperty]]:
        """Extract properties from a chunk of text."""
        variants = _get_name_variants(chemical_name)

        if self.use_llm:
            client, model = _make_llm_client(self.llm_api_key, self.ollama_base_url, self.ollama_model)
            if client is not None:
                return _extract_with_llm(text, chemical_name, client, model)

        return _extract_regex(text, chemical_name, variants)


# ──────────────────────────────────────────────── Main retriever

class LiteraturePropertyRetriever:
    """Orchestrates name conversion → search → PDF download → chunk → extract."""

    def __init__(self, *,
                 use_llm: bool = False,
                 max_papers: int = 10,
                 timeout: int = 15,
                 openai_api_key: Optional[str] = None,
                 cache_path: Optional[str] = None,
                 ollama_base_url: Optional[str] = None,
                 ollama_model: Optional[str] = None):
        self.name_converter = SMILESToNameConverter(timeout=timeout)
        self.searcher = AcademicSearcher(max_results=max_papers * 3, timeout=timeout)
        self.max_useful = max_papers  # cap on *useful* papers

        self._openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY') or None
        self._ollama_base_url = ollama_base_url or os.getenv('OLLAMA_BASE_URL') or None
        self._ollama_model = ollama_model or os.getenv('OLLAMA_MODEL') or 'ALIENTELLIGENCE/chemicalengineer'

        self.extractor = PropertyExtractor(
            use_llm=use_llm,
            llm_api_key=self._openai_api_key,
            ollama_base_url=self._ollama_base_url,
            ollama_model=self._ollama_model,
        )
        self.use_llm = self.extractor.use_llm

        self._analogue_mem: dict = {}

        self.cache = None
        if cache_path:
            try:
                from .literature_cache import LiteratureCache
                self.cache = LiteratureCache(cache_path)
                logger.info(f"Literature cache enabled at {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to init literature cache: {e}")

    # ── public API ─────────────────────────────────────────────────────────

    def retrieve_properties(self, smiles: str) -> LiteratureResult:
        if self.cache is not None:
            cached = self.cache.get(smiles)
            if cached is not None:
                logger.info(f"Cache hit: {smiles[:50]}")
                return cached

        empty_props = {k: None for k in PROP_KEYS}

        chemical_name, name_source = self._get_chemical_name(smiles)
        if not chemical_name:
            logger.debug(f"No name for: {smiles[:30]}")
            empty = LiteratureResult(smiles=smiles, chemical_name=None,
                                     properties=dict(empty_props),
                                     papers_searched=0, papers_with_hits=0)
            if self.cache:
                self.cache.put(smiles, empty)
            return empty

        logger.info(f"Chemical name: {chemical_name} (via {name_source})")
        if name_source == "PubChem":
            print(f"         📚 Known compound: '{chemical_name}'")
        else:
            print(f"         🧪 Novel compound (IUPAC): '{chemical_name}'")

        print(f"         📖 Searching literature...", end=" ", flush=True)
        raw_papers = self.searcher.search(chemical_name, smiles)

        variants = _get_name_variants(chemical_name)

        # ── Relevance filter + PDF download ────────────────────────────────
        useful_papers = []
        pdfs_downloaded = 0
        for paper in raw_papers:
            if len(useful_papers) >= self.max_useful:
                break
            abstract = paper.get('text', '')
            pdf_url = paper.get('pdf_url')
            abstract_mentions_name = _name_in_text((abstract or '').lower(), variants)

            # Try PDF download
            if pdf_url and pdfs_downloaded < self.max_useful:
                print(f"\n            📥 PDF: {paper['source']}...", end=" ", flush=True)
                full_text = _download_pdf(pdf_url)
                if full_text:
                    pdfs_downloaded += 1
                    paper['text'] = full_text
                    paper['has_full_text'] = True
                    paper['source'] = paper['source'] + '-FullText'
                    print(f"✓ ({len(full_text)} chars)", end="", flush=True)
                    useful_papers.append(paper)
                    continue
                else:
                    print("✗", end="", flush=True)

            # No full text — keep only if abstract mentions candidate
            if abstract_mentions_name:
                useful_papers.append(paper)
            # else: silently discard — doesn't count toward cap

        papers_searched = len(useful_papers)
        properties = dict(empty_props)
        papers_with_hits = 0
        citations: List[PaperCitation] = []

        # ── Extract from each useful paper ─────────────────────────────────
        for paper in useful_papers:
            text = paper.get('text', '')
            title = paper.get('title', '')
            if not text:
                continue

            # Name-centred chunking for full text
            if paper.get('has_full_text') and len(text) > 2000:
                chunks = _extract_name_chunks(text, variants)
                if not chunks:
                    chunks = [text[:3000]]
            else:
                chunks = [text]

            paper_props: Dict[str, Optional[RetrievedProperty]] = {k: None for k in PROP_KEYS}
            for chunk in chunks:
                extracted = self.extractor.extract(chunk, chemical_name)
                for pn, pv in extracted.items():
                    if pv is not None:
                        cur = paper_props.get(pn)
                        if cur is None or pv.confidence > cur.confidence:
                            paper_props[pn] = RetrievedProperty(
                                value=pv.value,
                                source=f"{title[:50]}..." if len(title) > 50 else title,
                                confidence=pv.confidence,
                            )

            hit_names = []
            found_any = False
            for pn, pv in paper_props.items():
                if pv is not None:
                    found_any = True
                    cur = properties.get(pn)
                    if cur is None or pv.confidence > cur.confidence:
                        properties[pn] = pv
                        hit_names.append(pn)

            if found_any:
                papers_with_hits += 1
                citations.append(PaperCitation(
                    title=title, authors=paper.get('authors', []),
                    doi=paper.get('doi', ''), source_db=paper.get('source', 'Unknown'),
                    properties_found=hit_names,
                ))

        # ── Analogue fallback ──────────────────────────────────────────────
        missing = [k for k, v in properties.items() if v is None]
        if missing:
            a_props, a_cites, a_n = self._search_analogues(smiles, chemical_name, missing)
            papers_searched += a_n
            for pn, pv in a_props.items():
                if properties.get(pn) is None and pv is not None:
                    properties[pn] = pv
            if a_cites:
                citations.extend(a_cites)
                papers_with_hits += len(a_cites)

        result = LiteratureResult(
            smiles=smiles, chemical_name=chemical_name,
            properties=properties, papers_searched=papers_searched,
            papers_with_hits=papers_with_hits, citations=citations,
        )

        found = [k for k, v in properties.items() if v is not None]
        logger.info(f"Literature found {len(found)}/4 properties: {found}")
        if found:
            print(f"\nfound {len(found)} properties!")
            print(f"            ✅ Literature values: {', '.join(found)}")
        else:
            print(f"\nno property values found")

        if self.cache:
            self.cache.put(smiles, result)
        return result

    # ── internals ──────────────────────────────────────────────────────────

    def _get_chemical_name(self, smiles: str) -> Tuple[Optional[str], str]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, ""
        canonical = Chem.MolToSmiles(mol)
        if self.name_converter._pubchempy_available:
            name = self.name_converter._query_pubchempy(canonical)
            if name:
                return name, "PubChem"
        if self.name_converter._smiles2iupac_available:
            name = self.name_converter._generate_iupac(canonical)
            if name:
                return name, "SMILES2IUPAC"
        return None, ""

    # ── Analogue suggestion (LLM) ──────────────────────────────────────────

    def _suggest_analogues_via_llm(self, smiles: str,
                                    chemical_name: Optional[str],
                                    top_k: int = 3) -> List[Tuple[str, str, float]]:
        try:
            client, model = _make_llm_client(
                self._openai_api_key, self._ollama_base_url, self._ollama_model)
            if client is None:
                return []

            display_name = chemical_name or smiles
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
  c. Include exotic or lesser-known compounds if they are genuinely closer.
  d. Do NOT default to RDX and HMX if a structurally tighter match exists.

For each compound return:
  "name"       — the primary acronym or common name as used in search engines
  "reason"     — one sentence listing the specific shared features
  "similarity" — your estimated structural similarity 0-1 (1 = identical)

Respond with ONLY a JSON array, no prose, no markdown fences:
[
  {{"name": "HMX", "reason": "8-membered nitramine ring with 4 N-NO2 groups", "similarity": 0.72}},
  {{"name": "TNAZ", "reason": "cyclic nitramine with gem-dinitro group", "similarity": 0.55}}
]"""

            def _call(msgs):
                return client.chat.completions.create(
                    model=model, messages=msgs, temperature=0.1, max_tokens=500)

            resp = _call([{'role': 'system', 'content': system},
                          {'role': 'user', 'content': prompt}])
            choice = resp.choices[0]
            finish = getattr(choice, 'finish_reason', 'unknown')
            raw = (choice.message.content or '').strip()

            if not raw:
                logger.warning(f"LLM analogue: empty (finish_reason={finish!r}), retrying neutral prompt.")
                neutral = (
                    f"You are a computational chemist specialising in nitrogen-rich "
                    f"heterocyclic compounds and dense organic crystals.\n\n"
                    f"Molecule SMILES: {smiles}\nIUPAC name: {display_name}\n\n"
                    f"List the {top_k} most structurally similar *well-characterised* reference "
                    f"compounds from academic literature for which crystal density, heat of "
                    f"formation, and Chapman-Jouguet parameters are published.\n\n"
                    f"Reply ONLY with a JSON array:\n"
                    f'[{{"name":"<acronym>","reason":"<shared features>","similarity":<0-1>}}]'
                )
                resp = _call([{'role': 'user', 'content': neutral}])
                choice = resp.choices[0]
                finish = getattr(choice, 'finish_reason', 'unknown')
                raw = (choice.message.content or '').strip()
                if not raw:
                    logger.warning(f"LLM analogue: still empty (finish_reason={finish!r}).")
                    return []

            raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
            arr = re.search(r'\[.*?\]', raw, re.DOTALL)
            if not arr:
                logger.warning(f"LLM analogue: no JSON array (finish_reason={finish!r}). "
                               f"Raw (first 400 chars): {raw[:400]!r}")
                return []
            data = json.loads(arr.group(0))
            results = []
            for item in data:
                name = str(item.get('name', '')).strip()
                reason = str(item.get('reason', '')).strip()
                try:
                    sim = max(0.1, min(1.0, float(item.get('similarity', 0.75))))
                except (TypeError, ValueError):
                    sim = 0.75
                if name:
                    results.append((name, reason, sim))
            logger.info(f"LLM analogues: {[(n, f'{s:.2f}') for n, _, s in results]}")
            return results[:top_k]
        except Exception as e:
            logger.warning(f"LLM analogue suggestion failed: {e}")
            return []

    # ── Analogue search pipeline ───────────────────────────────────────────

    def _search_analogues(self, smiles: str, chemical_name: Optional[str],
                          missing_props: List[str],
                          top_k: int = 3,
                          conf_penalty: float = 0.5,
                          ) -> Tuple[Dict[str, RetrievedProperty], List[PaperCitation], int]:
        candidates: List[Tuple[str, float, str]] = []

        llm_suggestions = self._suggest_analogues_via_llm(smiles, chemical_name, top_k)
        if llm_suggestions:
            print(f"         🤖 LLM suggested analogues: "
                  f"{', '.join(f'{n} ({s:.2f})' for n, _, s in llm_suggestions)}", flush=True)
            for name, reason, sim in llm_suggestions:
                candidates.append((name, sim, f"LLM sim={sim:.2f} ({reason[:55]})"))
        else:
            try:
                from .energetic_library import find_similar
                for compound, tanimoto in find_similar(smiles, top_k=top_k, min_tanimoto=0.30):
                    candidates.append((compound.name, tanimoto, f"Tanimoto={tanimoto:.2f}"))
            except Exception as e:
                logger.debug(f"Static library fallback failed: {e}")

        if not candidates:
            return {}, [], 0

        filled: Dict[str, RetrievedProperty] = {}
        analogue_citations: List[PaperCitation] = []
        papers_searched = 0
        still_missing = list(missing_props)

        for analogue_name, similarity, source_tag in candidates:
            if not still_missing:
                break
            norm = analogue_name.strip().lower()

            cached = self._analogue_mem.get(norm)
            if cached is None and self.cache is not None:
                cached = self.cache.get_analogue(analogue_name)
                if cached is not None:
                    self._analogue_mem[norm] = cached

            if cached is not None:
                cached_props, cached_cites, cached_n = cached
                print(f"         💾 Analogue cache hit: {analogue_name} ({len(cached_props)} props)", flush=True)
                papers_searched += cached_n
                for pn in list(still_missing):
                    pv = cached_props.get(pn)
                    if pv is None:
                        continue
                    sc = max(0.01, pv.confidence * (similarity ** conf_penalty))
                    ex = filled.get(pn)
                    if ex is not None and ex.confidence >= sc:
                        continue
                    filled[pn] = RetrievedProperty(value=pv.value, source=pv.source, confidence=sc)
                for c in cached_cites:
                    if c not in analogue_citations:
                        analogue_citations.append(c)
                still_missing = [p for p in still_missing if p not in filled]
                continue

            print(f"         🔗 Analogue: {analogue_name} [{source_tag}] → searching literature…", flush=True)
            papers = self.searcher.search(analogue_name, None)
            n_papers = len(papers)
            papers_searched += n_papers

            raw_props: Dict[str, RetrievedProperty] = {}
            raw_cites: List[PaperCitation] = []
            a_variants = _get_name_variants(analogue_name)

            for paper in papers:
                text = paper.get('text', '')
                if not text:
                    continue
                title = paper.get('title', '')
                # Try PDF for analogue papers too
                if paper.get('pdf_url') and not paper.get('has_full_text'):
                    ft = _download_pdf(paper['pdf_url'])
                    if ft:
                        text = ft
                        paper['has_full_text'] = True

                if paper.get('has_full_text') and len(text) > 2000:
                    chunks = _extract_name_chunks(text, a_variants)
                    if not chunks:
                        chunks = [text[:3000]]
                else:
                    chunks = [text]

                hits_here: List[str] = []
                for chunk in chunks:
                    extracted = self.extractor.extract(chunk, analogue_name)
                    for pn, pv in extracted.items():
                        if pv is None:
                            continue
                        ex = raw_props.get(pn)
                        if ex is None or pv.confidence > ex.confidence:
                            raw_props[pn] = RetrievedProperty(
                                value=pv.value,
                                source=f"analogue: {analogue_name} [{source_tag}] — {title[:40]}",
                                confidence=pv.confidence,
                            )
                            hits_here.append(pn)

                if hits_here:
                    raw_cites.append(PaperCitation(
                        title=f"[Analogue {analogue_name}] {title}",
                        authors=paper.get('authors', []),
                        doi=paper.get('doi', ''),
                        source_db=paper.get('source', 'Unknown'),
                        properties_found=hits_here,
                    ))

            self._analogue_mem[norm] = (raw_props, raw_cites, n_papers)
            if self.cache is not None:
                self.cache.put_analogue(analogue_name, raw_props, raw_cites, n_papers)

            for pn in list(still_missing):
                pv = raw_props.get(pn)
                if pv is None:
                    continue
                sc = max(0.01, pv.confidence * (similarity ** conf_penalty))
                ex = filled.get(pn)
                if ex is not None and ex.confidence >= sc:
                    continue
                filled[pn] = RetrievedProperty(value=pv.value, source=pv.source, confidence=sc)
            analogue_citations.extend(raw_cites)
            still_missing = [p for p in still_missing if p not in filled]
            if filled:
                print(f"            ✅ {analogue_name} filled: {', '.join(sorted(filled.keys()))}")

        return filled, analogue_citations, papers_searched


# ──────────────────────────────────────────────── convenience top-level function

def get_properties_with_literature(smiles: str,
                                   predictor,
                                   use_literature: bool = True,
                                   use_llm: bool = False) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Literature search first, then ML prediction for gaps."""
    properties: Dict[str, float] = {}
    sources: Dict[str, str] = {}

    if use_literature:
        retriever = LiteraturePropertyRetriever(use_llm=use_llm)
        result = retriever.retrieve_properties(smiles)
        for prop_name, prop_value in result.properties.items():
            if prop_value is not None:
                properties[prop_name] = prop_value.value
                sources[prop_name] = f"literature ({prop_value.source})"

    missing = [p for p in PROP_KEYS if p not in properties]
    if missing and predictor is not None:
        predicted = predictor.predict_properties(smiles)
        if predicted:
            for pn in missing:
                if pn in predicted and predicted[pn] is not None:
                    properties[pn] = predicted[pn]
                    sources[pn] = "predicted (XGBoost)"

    return properties, sources
