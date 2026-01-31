"""
RAG (Retrieval-Augmented Generation) module for energetic property lookup.

This module searches scientific literature for known property values before
falling back to ML prediction. It:
1. Converts SMILES to proper chemical names (IUPAC/common)
2. Searches multiple databases (OpenAlex, Crossref, Semantic Scholar) for papers
3. Extracts energetic properties from paper abstracts using regex/LLM
4. Returns found properties, with None for properties not found
"""

import os
import re
import json
import logging
import requests
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import Descriptors

logger = logging.getLogger(__name__)


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
    Converts SMILES strings to chemical names using multiple sources.
    
    Priority:
    1. Common name database lookup (fast, local)
    2. PubChemPy (comprehensive, uses PubChem API)
    3. RDKit systematic name generation (fallback)
    """
    
    # Common energetic materials name database
    COMMON_NAMES = {
        # Nitroaromatics
        "Cc1ccc(cc1[N+](=O)[O-])[N+](=O)[O-]": "TNT (2,4,6-Trinitrotoluene)",
        "c1cc(cc(c1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]": "TNB (1,3,5-Trinitrobenzene)",
        "Cc1c(cc(cc1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]": "TNT",
        
        # RDX/HMX family
        "C1N(CN(CN1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]": "RDX (Cyclotrimethylenetrinitramine)",
        "C1N(CN(CN(CN1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]": "HMX (Cyclotetramethylenetetranitramine)",
        
        # TATB
        "Nc1c([N+](=O)[O-])c(N)c([N+](=O)[O-])c(N)c1[N+](=O)[O-]": "TATB (Triaminotrinitrobenzene)",
        
        # CL-20
        "C12N(C3N(C(N1[N+](=O)[O-])N(C(N2[N+](=O)[O-])N3[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]": "CL-20 (Hexanitrohexaazaisowurtzitane)",
        
        # PETN
        "C(C(CO[N+](=O)[O-])(CO[N+](=O)[O-])CO[N+](=O)[O-])O[N+](=O)[O-]": "PETN (Pentaerythritol tetranitrate)",
        
        # Nitroglycerine
        "C(C(CO[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-]": "Nitroglycerin",
    }
    
    def __init__(self, use_pubchem: bool = True, timeout: int = 10):
        """
        Initialize converter.
        
        Args:
            use_pubchem: Whether to use PubChemPy for name lookup
            timeout: API timeout in seconds (not used by pubchempy directly)
        """
        self.use_pubchem = use_pubchem
        self.timeout = timeout
        self._pubchempy_available = False
        
        # Check if pubchempy is available
        if use_pubchem:
            try:
                import pubchempy as pcp
                self._pcp = pcp
                self._pubchempy_available = True
                logger.info("PubChemPy available for SMILES-to-name conversion")
            except ImportError:
                logger.warning("pubchempy not installed. Install with: pip install pubchempy")
                self._pubchempy_available = False
    
    def convert(self, smiles: str) -> Optional[str]:
        """
        Convert SMILES to chemical name.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Chemical name (IUPAC or common name) or None if not found
        """
        # Canonicalize SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        canonical_smiles = Chem.MolToSmiles(mol)
        
        # 1. Check common names database (fastest)
        name = self._lookup_common_name(canonical_smiles)
        if name:
            logger.debug(f"Found in common names DB: {name}")
            return name
        
        # 2. Try PubChemPy
        if self.use_pubchem and self._pubchempy_available:
            name = self._query_pubchempy(canonical_smiles)
            if name:
                logger.debug(f"Found via PubChemPy: {name}")
                return name
        
        # 3. Generate systematic name from structure (fallback)
        name = self._generate_systematic_name(mol)
        if name:
            logger.debug(f"Generated systematic name: {name}")
            return name
        
        return None
    
    def _lookup_common_name(self, canonical_smiles: str) -> Optional[str]:
        """Look up common name from database."""
        return self.COMMON_NAMES.get(canonical_smiles)
    
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
    
    def _generate_systematic_name(self, mol: Chem.Mol) -> Optional[str]:
        """Generate a descriptive name from molecular structure."""
        try:
            # Create a descriptive name based on molecular features
            formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
            mw = Descriptors.MolWt(mol)
            
            # Count key functional groups
            nitro_count = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[N+](=O)[O-]')))
            amino_count = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[NH2]')))
            azide_count = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[N-]=[N+]=[N-]')))
            ring_count = Descriptors.RingCount(mol)
            
            parts = []
            if nitro_count > 0:
                parts.append(f"{nitro_count}-nitro")
            if amino_count > 0:
                parts.append(f"{amino_count}-amino")
            if azide_count > 0:
                parts.append(f"{azide_count}-azido")
            if ring_count > 0:
                parts.append(f"{ring_count}-ring")
            
            parts.append(f"compound ({formula}, MW={mw:.1f})")
            
            return " ".join(parts) if parts else formula
            
        except Exception:
            return None


class LiteratureSearcher:
    """
    Searches multiple scientific literature databases for papers mentioning a chemical compound.
    
    Uses multiple APIs for robustness:
    1. OpenAlex (free, no auth required) - primary source
    2. Crossref (free, polite pool) - secondary source
    3. Semantic Scholar (free tier) - tertiary source
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
        
        # 1. Try OpenAlex first (most reliable, no auth needed)
        openalex_papers = self._search_openalex(chemical_name)
        papers.extend(openalex_papers)
        logger.info(f"OpenAlex found {len(openalex_papers)} papers for '{chemical_name}'")
        
        # 2. Try Crossref for additional coverage
        if len(papers) < self.max_results:
            crossref_papers = self._search_crossref(chemical_name)
            papers.extend(crossref_papers)
            logger.info(f"Crossref found {len(crossref_papers)} additional papers")
        
        # 3. Try Semantic Scholar as fallback
        if len(papers) < self.max_results // 2:
            semantic_papers = self._search_semantic_scholar(chemical_name)
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
    
    def _search_openalex(self, chemical_name: str) -> List[Dict]:
        """
        Search OpenAlex for papers (free, no auth required).
        
        OpenAlex is a free, open catalog of the world's scholarly works.
        """
        papers = []
        
        try:
            # OpenAlex API - search works
            query = f'{chemical_name} energetic explosive detonation propellant'
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
                        'abstract': abstract,
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
    
    def _search_crossref(self, chemical_name: str) -> List[Dict]:
        """Search Crossref for papers (free with polite pool)."""
        papers = []
        
        try:
            query = f'{chemical_name} energetic explosive detonation'
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
                        'abstract': abstract,
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
    
    def _search_semantic_scholar(self, chemical_name: str) -> List[Dict]:
        """Search Semantic Scholar for papers (free tier)."""
        papers = []
        
        try:
            query = f'{chemical_name} energetic material'
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
                        'abstract': item.get('abstract', '') or '',
                        'doi': doi,
                        'authors': [a.get('name', '') for a in (item.get('authors') or [])],
                        'published_date': item.get('publicationDate', ''),
                        'source': 'SemanticScholar'
                    }
                    if paper['title'] and paper['abstract']:
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
    Extracts energetic material properties from paper abstracts.
    
    Uses regex patterns and optionally LLM for more complex extraction.
    """
    
    # Property patterns with units
    PROPERTY_PATTERNS = {
        'Density': [
            # Density patterns: e.g., "density of 1.92 g/cm³", "ρ = 1.85 g cm⁻³"
            r'(?:density|ρ|rho)\s*(?:of|=|:)?\s*(\d+\.?\d*)\s*(?:g\s*/?\s*cm[³3⁻-]*|g\s*cm\s*[-−]?\s*3)',
            r'(\d+\.?\d*)\s*(?:g\s*/?\s*cm[³3⁻-]*|g\s*cm\s*[-−]?\s*3)\s*(?:density)',
        ],
        'Det Velocity': [
            # Detonation velocity: e.g., "detonation velocity of 8500 m/s", "D = 9000 m s⁻¹"
            r'(?:detonation\s+velocity|det\.?\s*vel\.?|D)\s*(?:of|=|:)?\s*(\d+\.?\d*)\s*(?:m\s*/?\s*s|m\s*s\s*[-−]?\s*1|km\s*/?\s*s)',
            r'(\d+\.?\d*)\s*(?:m\s*/?\s*s|km\s*/?\s*s)\s*(?:detonation)',
        ],
        'Det Pressure': [
            # Detonation pressure: e.g., "detonation pressure of 39.5 GPa", "P_CJ = 40 GPa"
            r'(?:detonation\s+pressure|det\.?\s*press\.?|P\s*(?:CJ|det)?)\s*(?:of|=|:)?\s*(\d+\.?\d*)\s*(?:GPa|kbar)',
            r'(\d+\.?\d*)\s*(?:GPa|kbar)\s*(?:detonation|pressure)',
        ],
        'Hf solid': [
            # Heat of formation: e.g., "heat of formation of 200 kJ/mol", "ΔHf = 150 kJ mol⁻¹"
            r'(?:heat\s+of\s+formation|enthalpy\s+of\s+formation|[ΔΔ]?\s*H\s*f?)\s*(?:of|=|:)?\s*([-−]?\d+\.?\d*)\s*(?:kJ\s*/?\s*mol|kJ\s*mol\s*[-−]?\s*1|kcal\s*/?\s*mol)',
            r'([-−]?\d+\.?\d*)\s*(?:kJ\s*/?\s*mol|kcal\s*/?\s*mol)\s*(?:heat|enthalpy|formation)',
        ],
    }
    
    # Unit conversion factors
    UNIT_CONVERSIONS = {
        'km/s': 1000,  # km/s to m/s
        'kbar': 0.1,   # kbar to GPa
        'kcal/mol': 4.184,  # kcal/mol to kJ/mol
    }
    
    def __init__(self, use_llm: bool = False, llm_api_key: str = None):
        """
        Initialize extractor.
        
        Args:
            use_llm: Whether to use LLM for complex extraction
            llm_api_key: OpenAI API key for LLM extraction
        """
        self.use_llm = use_llm
        self.llm_api_key = llm_api_key or os.getenv('OPENAI_API_KEY')
    
    def extract_from_abstract(self, abstract: str, chemical_name: str) -> Dict[str, Optional[RetrievedProperty]]:
        """
        Extract energetic properties from a paper abstract.
        
        Args:
            abstract: Paper abstract text
            chemical_name: Name of the chemical to look for
            
        Returns:
            Dictionary mapping property names to RetrievedProperty or None
        """
        properties = {
            'Density': None,
            'Det Velocity': None,
            'Det Pressure': None,
            'Hf solid': None
        }
        
        if not abstract:
            return properties
        
        # Normalize text
        text = abstract.lower()
        
        # Check if the chemical is mentioned
        name_lower = chemical_name.lower() if chemical_name else ""
        if name_lower and name_lower not in text:
            # Chemical not mentioned in this abstract
            return properties
        
        # Extract each property using regex
        for prop_name, patterns in self.PROPERTY_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    try:
                        value = float(matches[0].replace('−', '-'))
                        
                        # Apply unit conversions if needed
                        if 'km' in pattern and prop_name == 'Det Velocity':
                            value *= 1000  # km/s to m/s
                        elif 'kbar' in pattern and prop_name == 'Det Pressure':
                            value *= 0.1  # kbar to GPa
                        elif 'kcal' in pattern and prop_name == 'Hf solid':
                            value *= 4.184  # kcal/mol to kJ/mol
                        
                        # Validate reasonable ranges
                        if self._validate_value(prop_name, value):
                            properties[prop_name] = RetrievedProperty(
                                value=value,
                                source=f"Extracted from abstract",
                                confidence=0.7  # Regex extraction confidence
                            )
                            break
                    except (ValueError, IndexError):
                        continue
        
        # Optionally use LLM for more accurate extraction
        if self.use_llm and self.llm_api_key:
            llm_properties = self._extract_with_llm(abstract, chemical_name)
            # Merge LLM results (higher confidence)
            for prop_name, prop_value in llm_properties.items():
                if prop_value is not None:
                    properties[prop_name] = prop_value
        
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
    
    def _extract_with_llm(self, abstract: str, chemical_name: str) -> Dict[str, Optional[RetrievedProperty]]:
        """Use LLM to extract properties from abstract."""
        properties = {
            'Density': None,
            'Det Velocity': None,
            'Det Pressure': None,
            'Hf solid': None
        }
        
        try:
            import openai
            
            client = openai.OpenAI(api_key=self.llm_api_key)
            
            prompt = f"""Extract energetic material properties for "{chemical_name}" from this abstract.
            
Abstract:
{abstract}

Return ONLY a JSON object with these exact keys (use null if not found):
- "density": value in g/cm³
- "det_velocity": value in m/s
- "det_pressure": value in GPa
- "hf_solid": value in kJ/mol

JSON response:"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON from response
            json_match = re.search(r'\{[^}]+\}', result_text)
            if json_match:
                data = json.loads(json_match.group())
                
                mapping = {
                    'density': 'Density',
                    'det_velocity': 'Det Velocity',
                    'det_pressure': 'Det Pressure',
                    'hf_solid': 'Hf solid'
                }
                
                for json_key, prop_name in mapping.items():
                    value = data.get(json_key)
                    if value is not None and self._validate_value(prop_name, float(value)):
                        properties[prop_name] = RetrievedProperty(
                            value=float(value),
                            source="LLM extraction from abstract",
                            confidence=0.85
                        )
                        
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")
        
        return properties


class RAGPropertyRetriever:
    """
    Main RAG module that orchestrates SMILES-to-name conversion,
    literature search, and property extraction.
    """
    
    def __init__(self, 
                 use_pubchem: bool = True,
                 use_llm: bool = False,
                 max_papers: int = 10,
                 timeout: int = 15):
        """
        Initialize RAG retriever.
        
        Args:
            use_pubchem: Whether to use PubChem for name lookup
            use_llm: Whether to use LLM for property extraction
            max_papers: Maximum papers to search
            timeout: API timeout in seconds
        """
        self.name_converter = SMILESToNameConverter(use_pubchem=use_pubchem, timeout=timeout)
        self.searcher = LiteratureSearcher(max_results=max_papers, timeout=timeout)
        self.extractor = PropertyExtractor(use_llm=use_llm)
        
        self.use_pubchem = use_pubchem
        self.use_llm = use_llm
    
    def retrieve_properties(self, smiles: str) -> RAGResult:
        """
        Retrieve properties for a molecule from literature.
        
        Args:
            smiles: SMILES string of the molecule
            
        Returns:
            RAGResult with found properties
        """
        # Initialize result
        properties = {
            'Density': None,
            'Det Velocity': None,
            'Det Pressure': None,
            'Hf solid': None
        }
        
        # Step 1: Convert SMILES to name
        logger.info(f"Converting SMILES to name: {smiles[:50]}...")
        chemical_name = self.name_converter.convert(smiles)
        
        if not chemical_name:
            logger.warning(f"Could not convert SMILES to name: {smiles[:30]}...")
            return RAGResult(
                smiles=smiles,
                chemical_name=None,
                properties=properties,
                papers_searched=0,
                papers_with_hits=0
            )
        
        logger.info(f"Chemical name: {chemical_name}")
        
        # Step 2: Search literature
        logger.info(f"Searching literature databases for: {chemical_name}")
        papers = self.searcher.search(chemical_name, smiles)
        
        papers_searched = len(papers)
        papers_with_hits = 0
        citations = []
        
        # Step 3: Extract properties from each paper
        for paper in papers:
            abstract = paper.get('abstract', '')
            title = paper.get('title', '')
            
            if not abstract:
                continue
            
            extracted = self.extractor.extract_from_abstract(abstract, chemical_name)
            
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
        
        result = RAGResult(
            smiles=smiles,
            chemical_name=chemical_name,
            properties=properties,
            papers_searched=papers_searched,
            papers_with_hits=papers_with_hits,
            citations=citations
        )
        
        # Log summary
        found_props = [k for k, v in properties.items() if v is not None]
        logger.info(f"RAG found {len(found_props)}/4 properties: {found_props}")
        
        return result


def get_properties_with_rag(smiles: str, 
                            predictor,
                            use_rag: bool = True,
                            use_pubchem: bool = True,
                            use_llm: bool = False) -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    Get properties for a molecule, using RAG first then ML prediction for missing values.
    
    Args:
        smiles: SMILES string
        predictor: PropertyPredictor instance for ML fallback
        use_rag: Whether to use RAG retrieval
        use_pubchem: Whether to use PubChem for name lookup
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
            use_pubchem=use_pubchem,
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
