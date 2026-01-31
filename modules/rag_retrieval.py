"""
RAG (Retrieval-Augmented Generation) module for energetic property lookup.

This module searches ChemRXiv papers for known property values before
falling back to ML prediction. It:
1. Converts SMILES to proper chemical names (IUPAC/common)
2. Searches ChemRXiv for papers mentioning the molecule
3. Extracts energetic properties from paper abstracts using LLM
4. Returns found properties, with None for properties not found
"""

import os
import re
import json
import logging
import hashlib
import requests
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from functools import lru_cache

from rdkit import Chem
from rdkit.Chem import Descriptors

logger = logging.getLogger(__name__)

# Cache directory for API responses
CACHE_DIR = ".rag_cache"


@dataclass
class RetrievedProperty:
    """A property value retrieved from literature."""
    value: float
    source: str  # Paper title or DOI
    confidence: float  # 0-1 confidence score


@dataclass
class RAGResult:
    """Result of RAG property lookup."""
    smiles: str
    chemical_name: Optional[str]
    properties: Dict[str, Optional[RetrievedProperty]]  # Property name -> RetrievedProperty or None
    papers_searched: int
    papers_with_hits: int


class SMILESToNameConverter:
    """
    Converts SMILES strings to chemical names using multiple sources.
    
    Priority:
    1. PubChem API (most comprehensive)
    2. RDKit systematic name generation
    3. Common name database lookup
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
            use_pubchem: Whether to use PubChem API
            timeout: API timeout in seconds
        """
        self.use_pubchem = use_pubchem
        self.timeout = timeout
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self):
        """Create cache directory if it doesn't exist."""
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
    
    def _get_cache_path(self, smiles: str) -> str:
        """Get cache file path for a SMILES string."""
        hash_key = hashlib.md5(smiles.encode()).hexdigest()
        return os.path.join(CACHE_DIR, f"name_{hash_key}.json")
    
    def _load_from_cache(self, smiles: str) -> Optional[str]:
        """Load name from cache if available."""
        cache_path = self._get_cache_path(smiles)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    return data.get('name')
            except Exception:
                pass
        return None
    
    def _save_to_cache(self, smiles: str, name: str):
        """Save name to cache."""
        cache_path = self._get_cache_path(smiles)
        try:
            with open(cache_path, 'w') as f:
                json.dump({'smiles': smiles, 'name': name}, f)
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")
    
    def convert(self, smiles: str) -> Optional[str]:
        """
        Convert SMILES to chemical name.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Chemical name or None if not found
        """
        # Check cache first
        cached_name = self._load_from_cache(smiles)
        if cached_name:
            logger.debug(f"Name found in cache: {cached_name}")
            return cached_name
        
        # Canonicalize SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        canonical_smiles = Chem.MolToSmiles(mol)
        
        # 1. Check common names database
        name = self._lookup_common_name(canonical_smiles)
        if name:
            self._save_to_cache(smiles, name)
            return name
        
        # 2. Try PubChem API
        if self.use_pubchem:
            name = self._query_pubchem(canonical_smiles)
            if name:
                self._save_to_cache(smiles, name)
                return name
        
        # 3. Generate systematic name from structure
        name = self._generate_systematic_name(mol)
        if name:
            self._save_to_cache(smiles, name)
            return name
        
        return None
    
    def _lookup_common_name(self, canonical_smiles: str) -> Optional[str]:
        """Look up common name from database."""
        return self.COMMON_NAMES.get(canonical_smiles)
    
    def _query_pubchem(self, smiles: str) -> Optional[str]:
        """Query PubChem for chemical name."""
        try:
            # PubChem REST API endpoint
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{requests.utils.quote(smiles)}/property/IUPACName,Title/JSON"
            
            response = requests.get(url, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                props = data.get('PropertyTable', {}).get('Properties', [{}])[0]
                
                # Prefer common title over IUPAC name if available
                title = props.get('Title')
                iupac = props.get('IUPACName')
                
                if title and not title.startswith('CID'):
                    return title
                elif iupac:
                    return iupac
            
            logger.debug(f"PubChem returned status {response.status_code} for {smiles}")
            return None
            
        except requests.exceptions.Timeout:
            logger.warning(f"PubChem API timeout for {smiles}")
            return None
        except Exception as e:
            logger.warning(f"PubChem API error: {e}")
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


class ChemRXivSearcher:
    """
    Searches ChemRXiv for papers mentioning a chemical compound.
    
    Uses the ChemRXiv API to find papers and extracts relevant abstracts.
    """
    
    CHEMRXIV_API_BASE = "https://chemrxiv.org/engage/chemrxiv/public-api/v1"
    
    def __init__(self, max_results: int = 10, timeout: int = 15):
        """
        Initialize searcher.
        
        Args:
            max_results: Maximum number of papers to retrieve
            timeout: API timeout in seconds
        """
        self.max_results = max_results
        self.timeout = timeout
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self):
        """Create cache directory if it doesn't exist."""
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
    
    def _get_cache_path(self, query: str) -> str:
        """Get cache file path for a search query."""
        hash_key = hashlib.md5(query.encode()).hexdigest()
        return os.path.join(CACHE_DIR, f"search_{hash_key}.json")
    
    def search(self, chemical_name: str, smiles: str = None) -> List[Dict]:
        """
        Search ChemRXiv for papers mentioning the chemical.
        
        Args:
            chemical_name: Chemical name to search for
            smiles: Optional SMILES string to include in search
            
        Returns:
            List of paper dictionaries with title, abstract, doi, authors
        """
        # Build search query
        query_parts = [chemical_name]
        
        # Add energetic material keywords to improve relevance
        query = f'"{chemical_name}" AND (energetic OR explosive OR detonation OR propellant)'
        
        # Check cache
        cache_path = self._get_cache_path(query)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
                    logger.debug(f"Search results loaded from cache for: {chemical_name}")
                    return cached_data.get('papers', [])
            except Exception:
                pass
        
        papers = []
        
        try:
            # ChemRXiv API search endpoint
            url = f"{self.CHEMRXIV_API_BASE}/items"
            params = {
                'term': query,
                'limit': self.max_results,
                'sort': 'RELEVANCE'
            }
            
            response = requests.get(url, params=params, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                items = data.get('itemHits', [])
                
                for item in items:
                    item_data = item.get('item', {})
                    paper = {
                        'title': item_data.get('title', ''),
                        'abstract': item_data.get('abstract', ''),
                        'doi': item_data.get('doi', ''),
                        'authors': [a.get('firstName', '') + ' ' + a.get('lastName', '') 
                                   for a in item_data.get('authors', [])],
                        'published_date': item_data.get('publishedDate', ''),
                    }
                    papers.append(paper)
                
                logger.info(f"Found {len(papers)} papers for '{chemical_name}'")
            else:
                logger.warning(f"ChemRXiv API returned status {response.status_code}")
                
        except requests.exceptions.Timeout:
            logger.warning(f"ChemRXiv API timeout for '{chemical_name}'")
        except Exception as e:
            logger.warning(f"ChemRXiv API error: {e}")
        
        # Also try Crossref for additional coverage
        crossref_papers = self._search_crossref(chemical_name)
        papers.extend(crossref_papers)
        
        # Deduplicate by DOI
        seen_dois = set()
        unique_papers = []
        for paper in papers:
            doi = paper.get('doi', '')
            if doi and doi not in seen_dois:
                seen_dois.add(doi)
                unique_papers.append(paper)
            elif not doi:
                unique_papers.append(paper)
        
        # Cache results
        try:
            with open(cache_path, 'w') as f:
                json.dump({'query': query, 'papers': unique_papers[:self.max_results]}, f)
        except Exception as e:
            logger.warning(f"Failed to cache search results: {e}")
        
        return unique_papers[:self.max_results]
    
    def _search_crossref(self, chemical_name: str) -> List[Dict]:
        """Search Crossref for additional papers."""
        papers = []
        
        try:
            query = f'{chemical_name} energetic explosive detonation'
            url = "https://api.crossref.org/works"
            params = {
                'query': query,
                'rows': min(5, self.max_results),
                'filter': 'type:journal-article',
            }
            headers = {
                'User-Agent': 'EnergeticMoleculeDesigner/1.0 (mailto:research@example.com)'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                items = data.get('message', {}).get('items', [])
                
                for item in items:
                    paper = {
                        'title': item.get('title', [''])[0] if item.get('title') else '',
                        'abstract': item.get('abstract', ''),
                        'doi': item.get('DOI', ''),
                        'authors': [f"{a.get('given', '')} {a.get('family', '')}" 
                                   for a in item.get('author', [])],
                        'published_date': str(item.get('published-print', {}).get('date-parts', [['']])[0]),
                    }
                    if paper['title']:  # Only add if has title
                        papers.append(paper)
                        
        except Exception as e:
            logger.debug(f"Crossref search error: {e}")
        
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
        self.searcher = ChemRXivSearcher(max_results=max_papers, timeout=timeout)
        self.extractor = PropertyExtractor(use_llm=use_llm)
        
        self.use_pubchem = use_pubchem
        self.use_llm = use_llm
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self):
        """Create cache directory if it doesn't exist."""
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
    
    def _get_cache_path(self, smiles: str) -> str:
        """Get cache file path for RAG results."""
        hash_key = hashlib.md5(smiles.encode()).hexdigest()
        return os.path.join(CACHE_DIR, f"rag_{hash_key}.json")
    
    def retrieve_properties(self, smiles: str) -> RAGResult:
        """
        Retrieve properties for a molecule from literature.
        
        Args:
            smiles: SMILES string of the molecule
            
        Returns:
            RAGResult with found properties
        """
        # Check cache first
        cache_path = self._get_cache_path(smiles)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cached = json.load(f)
                    logger.info(f"RAG results loaded from cache for {smiles[:30]}...")
                    
                    # Reconstruct RAGResult from cached data
                    properties = {}
                    for prop_name, prop_data in cached.get('properties', {}).items():
                        if prop_data:
                            properties[prop_name] = RetrievedProperty(
                                value=prop_data['value'],
                                source=prop_data['source'],
                                confidence=prop_data['confidence']
                            )
                        else:
                            properties[prop_name] = None
                    
                    return RAGResult(
                        smiles=smiles,
                        chemical_name=cached.get('chemical_name'),
                        properties=properties,
                        papers_searched=cached.get('papers_searched', 0),
                        papers_with_hits=cached.get('papers_with_hits', 0)
                    )
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}")
        
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
        logger.info(f"Searching ChemRXiv for: {chemical_name}")
        papers = self.searcher.search(chemical_name, smiles)
        
        papers_searched = len(papers)
        papers_with_hits = 0
        
        # Step 3: Extract properties from each paper
        for paper in papers:
            abstract = paper.get('abstract', '')
            title = paper.get('title', '')
            
            if not abstract:
                continue
            
            extracted = self.extractor.extract_from_abstract(abstract, chemical_name)
            
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
            
            if found_any:
                papers_with_hits += 1
        
        result = RAGResult(
            smiles=smiles,
            chemical_name=chemical_name,
            properties=properties,
            papers_searched=papers_searched,
            papers_with_hits=papers_with_hits
        )
        
        # Cache results
        self._cache_result(smiles, result)
        
        # Log summary
        found_props = [k for k, v in properties.items() if v is not None]
        logger.info(f"RAG found {len(found_props)}/4 properties: {found_props}")
        
        return result
    
    def _cache_result(self, smiles: str, result: RAGResult):
        """Cache RAG result to disk."""
        cache_path = self._get_cache_path(smiles)
        try:
            data = {
                'smiles': result.smiles,
                'chemical_name': result.chemical_name,
                'properties': {
                    k: {'value': v.value, 'source': v.source, 'confidence': v.confidence} if v else None
                    for k, v in result.properties.items()
                },
                'papers_searched': result.papers_searched,
                'papers_with_hits': result.papers_with_hits
            }
            with open(cache_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to cache RAG result: {e}")


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
