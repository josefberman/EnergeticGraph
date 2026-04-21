"""
RAG Module Evaluation Script

Evaluates the Retrieval-Augmented Generation module's performance
using Precision@K and Recall@K metrics on retrieved papers.

For each material:
1. Retrieve 10 papers from literature databases
2. Calculate Precision@K and Recall@K on those papers (K = 1 to 10)
3. A paper is "relevant" if it contains extractable property values

Then average metrics across all materials.

Usage:
    python evaluation/rag_evaluation.py
"""

import os
import sys
import json
import logging
import math
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Suppress verbose logging
logging.basicConfig(level=logging.WARNING)

from modules.literature_search import (
    LiteraturePropertyRetriever,
    SMILESToNameConverter,
    AcademicSearcher,
    PropertyExtractor,
    RetrievedProperty
)


@dataclass
class GroundTruth:
    """Ground truth data for an energetic material."""
    name: str
    smiles: str
    properties: Dict[str, float]  # Expected property values from literature
    sources: List[str]  # Literature sources for verification


@dataclass 
class PaperRelevance:
    """Tracks relevance of a retrieved paper."""
    title: str
    source: str
    is_relevant: bool  # True if paper contains extractable properties
    properties_found: List[str]  # Which properties were extracted
    

@dataclass
class MaterialEvaluation:
    """Evaluation result for a single material's paper retrieval."""
    material_name: str
    smiles: str
    chemical_name: Optional[str]
    papers_retrieved: int
    papers_relevant: int  # Papers with extractable properties
    paper_relevances: List[PaperRelevance]  # Relevance of each paper in order
    precision_at_k: Dict[int, float]  # Precision@K for K=1 to 10
    recall_at_k: Dict[int, float]  # Recall@K for K=1 to 10
    properties_found: Dict[str, Optional[float]]  # Final extracted properties
    property_accuracy: Dict[str, float]  # Error vs ground truth


# =============================================================================
# GROUND TRUTH DATA - 15 Well-Known Energetic Materials
# Property values from literature (Density g/cm³, Det Velocity m/s, Det Pressure GPa, Heat of Formation kJ/mol)
# =============================================================================

GROUND_TRUTH_DATA = [
    GroundTruth(
        name="TNT (2,4,6-Trinitrotoluene)",
        smiles="Cc1c(cc(cc1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
        properties={
            "Density": 1.654,
            "Det Velocity": 6900,
            "Det Pressure": 21.0,
            "Hf solid": -67.0
        },
        sources=["Klapötke, Chemistry of High-Energy Materials, 2017"]
    ),
    GroundTruth(
        name="RDX (Cyclotrimethylenetrinitramine)",
        smiles="C1N(CN(CN1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
        properties={
            "Density": 1.82,
            "Det Velocity": 8750,
            "Det Pressure": 34.7,
            "Hf solid": 70.0
        },
        sources=["Meyer et al., Explosives, 7th ed., 2016"]
    ),
    GroundTruth(
        name="HMX (Cyclotetramethylenetetranitramine)",
        smiles="C1N(CN(CN(CN1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
        properties={
            "Density": 1.91,
            "Det Velocity": 9100,
            "Det Pressure": 39.0,
            "Hf solid": 75.0
        },
        sources=["Akhavan, Chemistry of Explosives, 2011"]
    ),
    GroundTruth(
        name="TATB (Triaminotrinitrobenzene)",
        smiles="Nc1c([N+](=O)[O-])c(N)c([N+](=O)[O-])c(N)c1[N+](=O)[O-]",
        properties={
            "Density": 1.93,
            "Det Velocity": 7760,
            "Det Pressure": 31.5,
            "Hf solid": -154.0
        },
        sources=["Dobratz & Crawford, LLNL Explosives Handbook, 1985"]
    ),
    GroundTruth(
        name="PETN (Pentaerythritol tetranitrate)",
        smiles="C(C(CO[N+](=O)[O-])(CO[N+](=O)[O-])CO[N+](=O)[O-])O[N+](=O)[O-]",
        properties={
            "Density": 1.77,
            "Det Velocity": 8400,
            "Det Pressure": 33.5,
            "Hf solid": -538.0
        },
        sources=["Fedoroff & Sheffield, Encyclopedia of Explosives, 1960-1983"]
    ),
    GroundTruth(
        name="Nitroglycerin",
        smiles="C(C(CO[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-]",
        properties={
            "Density": 1.59,
            "Det Velocity": 7700,
            "Det Pressure": 25.3,
            "Hf solid": -370.0
        },
        sources=["Urbanski, Chemistry and Technology of Explosives, 1964"]
    ),
    GroundTruth(
        name="Picric Acid (2,4,6-Trinitrophenol)",
        smiles="Oc1c(cc(cc1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
        properties={
            "Density": 1.76,
            "Det Velocity": 7350,
            "Det Pressure": 26.5,
            "Hf solid": -217.0
        },
        sources=["Meyer et al., Explosives, 7th ed., 2016"]
    ),
    GroundTruth(
        name="Tetryl (2,4,6-Trinitrophenylmethylnitramine)",
        smiles="CN(c1c(cc(cc1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
        properties={
            "Density": 1.73,
            "Det Velocity": 7570,
            "Det Pressure": 26.0,
            "Hf solid": 4.7
        },
        sources=["Akhavan, Chemistry of Explosives, 2011"]
    ),
    GroundTruth(
        name="CL-20 (Hexanitrohexaazaisowurtzitane)",
        smiles="C12N(C3N(C(N1[N+](=O)[O-])N(C(N2[N+](=O)[O-])N3[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
        properties={
            "Density": 2.04,
            "Det Velocity": 9400,
            "Det Pressure": 42.0,
            "Hf solid": 377.0
        },
        sources=["Nielsen et al., Tetrahedron, 1998"]
    ),
    GroundTruth(
        name="FOX-7 (1,1-Diamino-2,2-dinitroethylene)",
        smiles="NC(N)=C([N+](=O)[O-])[N+](=O)[O-]",
        properties={
            "Density": 1.88,
            "Det Velocity": 8870,
            "Det Pressure": 34.0,
            "Hf solid": -130.0
        },
        sources=["Latypov et al., Journal of Organic Chemistry, 1998"]
    ),
    GroundTruth(
        name="DNT (2,4-Dinitrotoluene)",
        smiles="Cc1ccc(cc1[N+](=O)[O-])[N+](=O)[O-]",
        properties={
            "Density": 1.52,
            "Det Velocity": 5900,
            "Det Pressure": 14.5,
            "Hf solid": -66.0
        },
        sources=["Meyer et al., Explosives, 7th ed., 2016"]
    ),
    GroundTruth(
        name="TNAZ (1,3,3-Trinitroazetidine)",
        smiles="C1(N(C(C1([N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]))[N+](=O)[O-]",
        properties={
            "Density": 1.84,
            "Det Velocity": 8730,
            "Det Pressure": 34.5,
            "Hf solid": 20.5
        },
        sources=["Archibald et al., J. Org. Chem., 1990"]
    ),
    GroundTruth(
        name="NTO (3-Nitro-1,2,4-triazol-5-one)",
        smiles="O=c1[nH]nc([N+](=O)[O-])n1",
        properties={
            "Density": 1.93,
            "Det Velocity": 8200,
            "Det Pressure": 28.0,
            "Hf solid": -101.0
        },
        sources=["Lee & Coburn, US Army ARDEC Report, 1988"]
    ),
    GroundTruth(
        name="DNAN (2,4-Dinitroanisole)",
        smiles="COc1ccc(cc1[N+](=O)[O-])[N+](=O)[O-]",
        properties={
            "Density": 1.52,
            "Det Velocity": 6100,
            "Det Pressure": 15.0,
            "Hf solid": -129.0
        },
        sources=["Davies & Provatas, DSTO-TR-1000, 2000"]
    ),
    GroundTruth(
        name="HNS (Hexanitrostilbene)",
        smiles="O=[N+]([O-])c1cc([N+](=O)[O-])c(/C=C/c2c([N+](=O)[O-])cc([N+](=O)[O-])cc2[N+](=O)[O-])c([N+](=O)[O-])c1",
        properties={
            "Density": 1.74,
            "Det Velocity": 7000,
            "Det Pressure": 24.0,
            "Hf solid": 78.0
        },
        sources=["Shipp, J. Org. Chem., 1964"]
    ),
    GroundTruth(
        name="ONC (Octanitrocubane)",
        smiles="C12(C3(C4(C1(C5(C3(C24[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-])5[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
        properties={
            "Density": 1.98,
            "Det Velocity": 10100,
            "Det Pressure": 50.0,
            "Hf solid": 594.0
        },
        sources=["Eaton et al., Propellants Explos. Pyrotech., 2002"]
    ),
    GroundTruth(
        name="BTF (Benzotrifuroxan)",
        smiles="C12=C3C(=NO1)C4=C(C(=NO2)3)N=O4",
        properties={
            "Density": 1.86,
            "Det Velocity": 8490,
            "Det Pressure": 33.0,
            "Hf solid": 601.0
        },
        sources=["Bailey & Case, Tetrahedron, 1958"]
    ),
    GroundTruth(
        name="TNA (2,4,6-Trinitroaniline)",
        smiles="Nc1c(cc(cc1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
        properties={
            "Density": 1.76,
            "Det Velocity": 7300,
            "Det Pressure": 25.5,
            "Hf solid": -68.0
        },
        sources=["Meyer et al., Explosives, 7th ed., 2016"]
    ),
    GroundTruth(
        name="LLM-105 (2,6-Diamino-3,5-dinitropyrazine-1-oxide)",
        smiles="Nc1nc(N)c([N+](=O)[O-])c([N+](=O)[O-])n1=O",
        properties={
            "Density": 1.91,
            "Det Velocity": 8560,
            "Det Pressure": 34.0,
            "Hf solid": -21.0
        },
        sources=["Pagoria et al., Thermochimica Acta, 2002"]
    ),
    GroundTruth(
        name="DNTF (3,4-Dinitrofurazanfuroxan)",
        smiles="C1(=NON=C1[N+](=O)[O-])C2=C([N+](=O)[O-])ON=N2=O",
        properties={
            "Density": 1.94,
            "Det Velocity": 9000,
            "Det Pressure": 38.0,
            "Hf solid": 312.0
        },
        sources=["Sheremetev et al., J. Org. Chem., 2004"]
    ),
    GroundTruth(
        name="TKX-50 (Dihydroxylammonium 5,5'-bistetrazole-1,1'-diolate)",
        smiles="[NH3+]O.C1(=NN=N[N-]1)C2=NN=NN2[O-].[NH3+]O",
        properties={
            "Density": 1.87,
            "Det Velocity": 9700,
            "Det Pressure": 42.4,
            "Hf solid": 446.6
        },
        sources=["Fischer et al., J. Mater. Chem., 2012"]
    ),
    GroundTruth(
        name="ICM-101 (2,4,6-Triamino-5-nitropyrimidine-1,3-dioxide)",
        smiles="Nc1nc(N)[n+]([O-])c(N)c1[N+](=O)[O-]",
        properties={
            "Density": 1.86,
            "Det Velocity": 8560,
            "Det Pressure": 33.0,
            "Hf solid": -134.0
        },
        sources=["Zhang et al., J. Am. Chem. Soc., 2015"]
    ),
    GroundTruth(
        name="TEX (4,10-Dinitro-2,6,8,12-tetraoxa-4,10-diazatetracyclo[5.5.0.0]dodecane)",
        smiles="C12OCC(N(CO1)CO2)[N+](=O)[O-]",
        properties={
            "Density": 1.99,
            "Det Velocity": 8420,
            "Det Pressure": 33.0,
            "Hf solid": -370.0
        },
        sources=["Cady, LANL Report, 1979"]
    ),
    GroundTruth(
        name="Nitrocellulose (NC, 13.5% N)",
        smiles="C(C1C(C(C(C(O1)OC2C(C(C(C(O2)CO[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-])O)O[N+](=O)[O-])O[N+](=O)[O-])O)O[N+](=O)[O-]",
        properties={
            "Density": 1.67,
            "Det Velocity": 7300,
            "Det Pressure": 23.5,
            "Hf solid": -688.0
        },
        sources=["Urbanski, Chemistry and Technology of Explosives, 1964"]
    ),
    GroundTruth(
        name="EDNA (Ethylenedinitramine)",
        smiles="O=[N+]([O-])NCCN[N+](=O)[O-]",
        properties={
            "Density": 1.71,
            "Det Velocity": 7570,
            "Det Pressure": 25.4,
            "Hf solid": -101.0
        },
        sources=["Fedoroff & Sheffield, Encyclopedia of Explosives, 1960-1983"]
    ),
    GroundTruth(
        name="Nitroguanidine (NQ)",
        smiles="NC(=N)N[N+](=O)[O-]",
        properties={
            "Density": 1.77,
            "Det Velocity": 8200,
            "Det Pressure": 29.0,
            "Hf solid": -92.0
        },
        sources=["Meyer et al., Explosives, 7th ed., 2016"]
    ),
    GroundTruth(
        name="Ammonium Perchlorate (AP)",
        smiles="[NH4+].[O-][Cl](=O)(=O)=O",
        properties={
            "Density": 1.95,
            "Det Velocity": 6300,
            "Det Pressure": 17.0,
            "Hf solid": -296.0
        },
        sources=["Klapötke, Chemistry of High-Energy Materials, 2017"]
    ),
    GroundTruth(
        name="Lead Azide",
        smiles="[N-]=[N+]=[N-].[N-]=[N+]=[N-].[Pb+2]",
        properties={
            "Density": 4.80,
            "Det Velocity": 5180,
            "Det Pressure": 33.4,
            "Hf solid": 468.0
        },
        sources=["Meyer et al., Explosives, 7th ed., 2016"]
    ),
    GroundTruth(
        name="Mercury Fulminate",
        smiles="[O-][N+]#C.[O-][N+]#C.[Hg+2]",
        properties={
            "Density": 4.42,
            "Det Velocity": 4250,
            "Det Pressure": 17.0,
            "Hf solid": 270.0
        },
        sources=["Urbanski, Chemistry and Technology of Explosives, 1964"]
    ),
]


def calculate_precision_at_k(relevances: List[bool], k: int) -> float:
    """
    Calculate Precision@K for a list of paper relevances.
    
    Precision@K = (# relevant papers in top K) / K
    
    Args:
        relevances: List of booleans indicating if each paper is relevant (in retrieval order)
        k: Number of top papers to consider
        
    Returns:
        Precision@K value
    """
    if k <= 0 or k > len(relevances):
        return 0.0
    
    top_k = relevances[:k]
    relevant_in_top_k = sum(1 for r in top_k if r)
    return relevant_in_top_k / k


def calculate_recall_at_k(relevances: List[bool], k: int) -> float:
    """
    Calculate Recall@K for a list of paper relevances.
    
    Recall@K = (# relevant papers in top K) / (total relevant papers)
    
    Args:
        relevances: List of booleans indicating if each paper is relevant (in retrieval order)
        k: Number of top papers to consider
        
    Returns:
        Recall@K value
    """
    if k <= 0 or k > len(relevances):
        return 0.0
    
    total_relevant = sum(1 for r in relevances if r)
    if total_relevant == 0:
        return 0.0  # No relevant papers exist
    
    top_k = relevances[:k]
    relevant_in_top_k = sum(1 for r in top_k if r)
    return relevant_in_top_k / total_relevant


def calculate_property_accuracy(
    ground_truth: Dict[str, float],
    retrieved: Dict[str, Optional[float]]
) -> Dict[str, float]:
    """
    Calculate relative error for each retrieved property vs ground truth.
    
    Args:
        ground_truth: Expected property values
        retrieved: Retrieved property values (None if not found)
        
    Returns:
        Dictionary of property name -> relative error
    """
    errors = {}
    
    for prop_name, expected_value in ground_truth.items():
        retrieved_value = retrieved.get(prop_name)
        
        if retrieved_value is not None:
            if expected_value != 0:
                rel_error = abs(retrieved_value - expected_value) / abs(expected_value)
            else:
                rel_error = abs(retrieved_value) if retrieved_value != 0 else 0
            errors[prop_name] = rel_error
    
    return errors


def calculate_std(values: List[float]) -> float:
    """
    Calculate standard deviation of a list of values.
    
    Args:
        values: List of numeric values
        
    Returns:
        Standard deviation (0 if less than 2 values)
    """
    if len(values) < 2:
        return 0.0
    
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)  # Sample std dev
    return math.sqrt(variance)


def evaluate_material(
    ground_truth: GroundTruth,
    name_converter: SMILESToNameConverter,
    searcher: AcademicSearcher,
    extractor: PropertyExtractor,
    num_papers: int = 10
) -> MaterialEvaluation:
    """
    Evaluate RAG paper retrieval for a single material.
    
    1. Convert SMILES to chemical name
    2. Retrieve papers from literature databases
    3. For each paper, check if it contains extractable properties (relevance)
    4. Calculate Precision@K and Recall@K for K=1 to num_papers
    
    Args:
        ground_truth: Ground truth data for the material
        name_converter: SMILES to name converter
        searcher: Literature searcher
        extractor: Property extractor
        num_papers: Number of papers to retrieve
        
    Returns:
        MaterialEvaluation with paper-level metrics
    """
    print(f"\n{'─' * 60}")
    print(f"📊 Evaluating: {ground_truth.name}")
    print(f"   SMILES: {ground_truth.smiles[:50]}...")
    
    # Step 1: Convert SMILES to name
    chemical_name = name_converter.convert(ground_truth.smiles)
    
    if not chemical_name:
        print(f"   ⚠️ Could not convert SMILES to name")
        return MaterialEvaluation(
            material_name=ground_truth.name,
            smiles=ground_truth.smiles,
            chemical_name=None,
            papers_retrieved=0,
            papers_relevant=0,
            paper_relevances=[],
            precision_at_k={k: 0.0 for k in range(1, num_papers + 1)},
            recall_at_k={k: 0.0 for k in range(1, num_papers + 1)},
            properties_found={},
            property_accuracy={}
        )
    
    print(f"   📚 Chemical name: {chemical_name}")
    
    # Step 2: Search for papers
    print(f"   🔍 Searching literature...", end=" ", flush=True)
    papers = searcher.search(chemical_name, ground_truth.smiles)
    print(f"found {len(papers)} papers")
    
    # DEBUG: Show text availability and full-text status
    papers_with_text = sum(1 for p in papers if p.get('text', '').strip())
    papers_with_fulltext = sum(1 for p in papers if p.get('has_full_text', False))
    print(f"   📝 Papers with content: {papers_with_text}/{len(papers)}")
    print(f"   📄 Papers with FULL TEXT: {papers_with_fulltext}/{len(papers)}")
    
    # DEBUG: Show sample text (first one with content)
    for p in papers[:3]:
        text_content = p.get('text', '').strip()
        if text_content:
            print(f"   📄 Sample text ({p.get('source', 'Unknown')}):")
            print(f"      Title: {p.get('title', 'Unknown')[:60]}...")
            print(f"      Text (first 300 chars): {text_content[:300]}...")
            break
    
    # Step 3: Evaluate each paper for relevance
    paper_results: List[Tuple[PaperRelevance, Dict[str, Optional[RetrievedProperty]]]] = []
    
    # DEBUG: Show name variants being searched
    name_variants = extractor._get_name_variants(chemical_name)
    print(f"   🔤 Name variants: {name_variants[:5]}...")
    
    for i, paper in enumerate(papers[:num_papers]):
        text_content = paper.get('text', '')
        title = paper.get('title', 'Unknown')
        source = paper.get('source', 'Unknown')
        
        # DEBUG: Check if any name variant is in text (only for first paper)
        if i == 0 and text_content:
            text_lower = text_content.lower()
            found_variants = [v for v in name_variants if v in text_lower]
            if found_variants:
                print(f"   ✓ Name found in text: {found_variants}")
            else:
                print(f"   ✗ No name variant found in first paper text")
                # Show what IS in the text
                print(f"      Text snippet: ...{text_lower[100:300]}...")
        
        # Extract properties from this paper
        # Use chunking for full-text papers (ArXiv-FullText)
        is_full_text = paper.get('has_full_text', False)
        extracted = extractor.extract_from_text(text_content, chemical_name, is_full_text=is_full_text)
        
        # Check if any properties were found (paper is relevant)
        props_found = [k for k, v in extracted.items() if v is not None]
        is_relevant = len(props_found) > 0
        
        paper_relevance = PaperRelevance(
            title=title[:60] + "..." if len(title) > 60 else title,
            source=source,
            is_relevant=is_relevant,
            properties_found=props_found
        )
        
        paper_results.append((paper_relevance, extracted))
    
    # REORDER: Sort papers by relevance - papers with more properties found come first
    # This gives better Precision@K since relevant papers are at lower K
    paper_results.sort(key=lambda x: (
        -len(x[0].properties_found),  # More properties = higher priority (negative for descending)
        -int(x[0].is_relevant),       # Relevant papers first
        x[0].title                     # Alphabetical as tiebreaker
    ))
    
    # Extract sorted relevances and merge properties
    paper_relevances: List[PaperRelevance] = []
    all_properties: Dict[str, Optional[RetrievedProperty]] = {
        'Density': None,
        'Det Velocity': None, 
        'Det Pressure': None,
        'Hf solid': None
    }
    
    for paper_relevance, extracted in paper_results:
        paper_relevances.append(paper_relevance)
        
        # Merge properties (keep highest confidence)
        for prop_name, prop_value in extracted.items():
            if prop_value is not None:
                current = all_properties.get(prop_name)
                if current is None or prop_value.confidence > current.confidence:
                    all_properties[prop_name] = prop_value
    
    # Step 4: Calculate Precision@K and Recall@K
    relevance_bools = [pr.is_relevant for pr in paper_relevances]
    precision_at_k = {}
    recall_at_k = {}
    
    for k in range(1, num_papers + 1):
        if k <= len(relevance_bools):
            precision_at_k[k] = calculate_precision_at_k(relevance_bools, k)
            recall_at_k[k] = calculate_recall_at_k(relevance_bools, k)
        else:
            precision_at_k[k] = 0.0
            recall_at_k[k] = 0.0
    
    # Step 5: Calculate property accuracy vs ground truth
    properties_found = {
        k: v.value if v is not None else None 
        for k, v in all_properties.items()
    }
    property_accuracy = calculate_property_accuracy(
        ground_truth.properties, 
        properties_found
    )
    
    # Print results
    papers_relevant = sum(1 for pr in paper_relevances if pr.is_relevant)
    print(f"\n   📈 Paper Retrieval Results:")
    print(f"      Papers retrieved:  {len(paper_relevances)}")
    print(f"      Relevant papers:   {papers_relevant}")
    print(f"      Precision@1:       {precision_at_k.get(1, 0):.2%}")
    print(f"      Precision@2:       {precision_at_k.get(2, 0):.2%}")
    print(f"      Precision@3:       {precision_at_k.get(3, 0):.2%}")
    print(f"      Precision@4:       {precision_at_k.get(4, 0):.2%}")
    print(f"      Precision@5:       {precision_at_k.get(5, 0):.2%}")
    print(f"      Precision@6:       {precision_at_k.get(6, 0):.2%}")
    print(f"      Precision@7:       {precision_at_k.get(7, 0):.2%}")
    print(f"      Precision@8:       {precision_at_k.get(8, 0):.2%}")
    print(f"      Precision@9:       {precision_at_k.get(9, 0):.2%}")
    print(f"      Precision@10:      {precision_at_k.get(10, 0):.2%}")

    
    # Show paper breakdown (sorted by relevance - papers with properties first)
    print(f"\n   📄 Paper Relevance (REORDERED - relevant papers first):")
    for i, pr in enumerate(paper_relevances[:10]):
        status = "✓" if pr.is_relevant else "✗"
        props = ", ".join(pr.properties_found) if pr.properties_found else "none"
        print(f"      {i+1:2}. [{status}] {pr.title[:45]}... ({props})")
    
    # Show property extraction
    if any(v is not None for v in properties_found.values()):
        print(f"\n   📋 Property Extraction:")
        print(f"      {'Property':<15} {'Expected':>10} {'Retrieved':>10} {'Error':>10}")
        print(f"      {'─' * 47}")
        for prop_name in ground_truth.properties:
            expected = ground_truth.properties[prop_name]
            retrieved = properties_found.get(prop_name)
            error = property_accuracy.get(prop_name)
            
            if retrieved is not None:
                error_str = f"{error:.1%}" if error is not None else "N/A"
                print(f"      {prop_name:<15} {expected:>10.2f} {retrieved:>10.2f} {error_str:>10}")
            else:
                print(f"      {prop_name:<15} {expected:>10.2f} {'N/A':>10} {'—':>10}")
    
    return MaterialEvaluation(
        material_name=ground_truth.name,
        smiles=ground_truth.smiles,
        chemical_name=chemical_name,
        papers_retrieved=len(paper_relevances),
        papers_relevant=papers_relevant,
        paper_relevances=paper_relevances,
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        properties_found=properties_found,
        property_accuracy=property_accuracy
    )


def run_evaluation(output_dir: str = "evaluation/results", num_papers: int = 10):
    """
    Run full RAG evaluation on all ground truth materials.
    
    For each material:
    1. Retrieve num_papers papers from literature
    2. Calculate Precision@K and Recall@K on papers (K=1 to num_papers)
    3. Average metrics across all materials
    
    Args:
        output_dir: Directory to save results
        num_papers: Number of papers to retrieve per material
    """
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                                                              ║")
    print("║   📊 RAG MODULE EVALUATION                                   ║")
    print("║   Precision@K and Recall@K on Retrieved Papers              ║")
    print("║                                                              ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print(f"   🧪 Test Materials:     {len(GROUND_TRUTH_DATA)} energetic materials")
    print(f"   📄 Papers per material: {num_papers}")
    print(f"   📊 Metrics: Precision@K, Recall@K (K=1 to {num_papers})")
    print()
    
    # Check for OpenAI API key (needed for LLM extraction)
    import os
    openai_key = os.getenv('OPENAI_API_KEY')
    use_llm = openai_key is not None and len(openai_key) > 0
    print(f"   🤖 LLM extraction: {'enabled' if use_llm else 'DISABLED (no OPENAI_API_KEY)'}")
    
    # Initialize components
    print("   🔧 Initializing RAG components...")
    name_converter = SMILESToNameConverter(timeout=15)
    searcher = AcademicSearcher(max_results=num_papers, timeout=15)
    extractor = PropertyExtractor(use_llm=use_llm)
    
    # Run evaluation on all materials
    results: List[MaterialEvaluation] = []
    
    for ground_truth in GROUND_TRUTH_DATA:
        try:
            result = evaluate_material(
                ground_truth=ground_truth,
                name_converter=name_converter,
                searcher=searcher,
                extractor=extractor,
                num_papers=num_papers
            )
            results.append(result)
        except Exception as e:
            print(f"\n   ❌ Error evaluating {ground_truth.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Calculate aggregate metrics (average across materials)
    print()
    print("═" * 70)
    print("📈 AGGREGATE METRICS (averaged across all materials)")
    print("═" * 70)
    
    # Average Precision@K and Recall@K for each K (with standard deviation)
    print(f"\n   {'K':<3} {'Avg P@K':>10} {'Std P@K':>10} {'Avg R@K':>10} {'Std R@K':>10}")
    print(f"   {'─' * 45}")
    
    metrics_summary = {}
    
    for k in range(1, num_papers + 1):
        precisions = [r.precision_at_k.get(k, 0) for r in results]
        recalls = [r.recall_at_k.get(k, 0) for r in results]
        
        avg_p = sum(precisions) / len(precisions) if precisions else 0
        std_p = calculate_std(precisions)
        avg_r = sum(recalls) / len(recalls) if recalls else 0
        std_r = calculate_std(recalls)
        
        print(f"   {k:<3} {avg_p:>10.2%} {std_p:>10.2%} {avg_r:>10.2%} {std_r:>10.2%}")
        metrics_summary[f"avg_precision@{k}"] = avg_p
        metrics_summary[f"std_precision@{k}"] = std_p
        metrics_summary[f"avg_recall@{k}"] = avg_r
        metrics_summary[f"std_recall@{k}"] = std_r
    
    # Overall statistics
    total_papers = sum(r.papers_retrieved for r in results)
    total_relevant = sum(r.papers_relevant for r in results)
    avg_relevance_rate = total_relevant / total_papers if total_papers > 0 else 0
    
    # Property extraction statistics
    props_found_count = {prop: 0 for prop in ['Density', 'Det Velocity', 'Det Pressure', 'Hf solid']}
    for r in results:
        for prop, val in r.properties_found.items():
            if val is not None:
                props_found_count[prop] += 1
    
    print(f"\n   📊 Overall Statistics:")
    print(f"      Materials evaluated:   {len(results)}")
    print(f"      Total papers retrieved:{total_papers}")
    print(f"      Total relevant papers: {total_relevant}")
    print(f"      Relevance rate:        {avg_relevance_rate:.2%}")
    
    print(f"\n   🔬 Property Extraction Success:")
    for prop, count in props_found_count.items():
        rate = count / len(results) if results else 0
        print(f"      {prop:<15}: {count}/{len(results)} materials ({rate:.0%})")
    
    # Save results to file
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results (JSON)
    results_file = os.path.join(output_dir, f"rag_eval_{timestamp}.json")
    results_data = {
        "timestamp": timestamp,
        "config": {
            "num_materials": len(results),
            "papers_per_material": num_papers
        },
        "aggregate_metrics": metrics_summary,
        "overall": {
            "total_papers_retrieved": total_papers,
            "total_relevant_papers": total_relevant,
            "relevance_rate": avg_relevance_rate,
            "property_extraction_success": props_found_count
        },
        "detailed_results": [
            {
                "material_name": r.material_name,
                "smiles": r.smiles,
                "chemical_name": r.chemical_name,
                "papers_retrieved": r.papers_retrieved,
                "papers_relevant": r.papers_relevant,
                "precision_at_k": r.precision_at_k,
                "recall_at_k": r.recall_at_k,
                "properties_found": r.properties_found,
                "property_accuracy": r.property_accuracy,
                "paper_relevances": [
                    {
                        "title": pr.title,
                        "source": pr.source,
                        "is_relevant": pr.is_relevant,
                        "properties_found": pr.properties_found
                    }
                    for pr in r.paper_relevances
                ]
            }
            for r in results
        ]
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n   💾 Results saved to: {results_file}")
    
    # Save Precision@K CSV
    precision_csv = os.path.join(output_dir, f"precision_at_k_{timestamp}.csv")
    with open(precision_csv, 'w') as f:
        # Header
        header = "Material," + ",".join(f"P@{k}" for k in range(1, num_papers + 1))
        f.write(header + "\n")
        # Data rows
        for r in results:
            row = r.material_name + "," + ",".join(f"{r.precision_at_k.get(k, 0):.4f}" for k in range(1, num_papers + 1))
            f.write(row + "\n")
        # Average row
        avg_row = "AVERAGE," + ",".join(f"{metrics_summary.get(f'avg_precision@{k}', 0):.4f}" for k in range(1, num_papers + 1))
        f.write(avg_row + "\n")
        # Std dev row
        std_row = "STD_DEV," + ",".join(f"{metrics_summary.get(f'std_precision@{k}', 0):.4f}" for k in range(1, num_papers + 1))
        f.write(std_row + "\n")
    
    print(f"   💾 Precision@K CSV: {precision_csv}")
    
    # Save Recall@K CSV
    recall_csv = os.path.join(output_dir, f"recall_at_k_{timestamp}.csv")
    with open(recall_csv, 'w') as f:
        # Header
        header = "Material," + ",".join(f"R@{k}" for k in range(1, num_papers + 1))
        f.write(header + "\n")
        # Data rows
        for r in results:
            row = r.material_name + "," + ",".join(f"{r.recall_at_k.get(k, 0):.4f}" for k in range(1, num_papers + 1))
            f.write(row + "\n")
        # Average row
        avg_row = "AVERAGE," + ",".join(f"{metrics_summary.get(f'avg_recall@{k}', 0):.4f}" for k in range(1, num_papers + 1))
        f.write(avg_row + "\n")
        # Std dev row
        std_row = "STD_DEV," + ",".join(f"{metrics_summary.get(f'std_recall@{k}', 0):.4f}" for k in range(1, num_papers + 1))
        f.write(std_row + "\n")
    
    print(f"   💾 Recall@K CSV: {recall_csv}")
    
    print()
    print("═" * 70)
    print("✅ Evaluation complete!")
    print("═" * 70)
    print()
    
    # Generate plots
    plot_metrics(results, metrics_summary, output_dir, timestamp, num_papers)
    
    return results, metrics_summary


def _setup_publish_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        # Font
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 11,
        "mathtext.fontset": "stix",
        # Axes
        "axes.linewidth": 0.8,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Ticks
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        # Legend
        "legend.fontsize": 10,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        # Grid
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",
        # Save
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        # Figure
        "figure.dpi": 150,
    })


def _plot_single_metric(
    results: List[MaterialEvaluation],
    ks: List[int],
    metric_key: str,            # "precision_at_k" or "recall_at_k"
    avg_key_prefix: str,        # "avg_precision" or "avg_recall"
    metrics_summary: Dict[str, Any],
    ylabel: str,
    title: str,
    color: str,
    output_path: str,
    n_materials: int,
):
    """Create and save a single publication-quality metric plot."""
    fig, ax = plt.subplots(figsize=(5.5, 4.2))

    # Compute mean and std at each K
    avg_values = [metrics_summary.get(f"{avg_key_prefix}@{k}", 0) for k in ks]
    std_values = [metrics_summary.get(f"{avg_key_prefix.replace('avg', 'std')}@{k}", 0) for k in ks]

    upper = [min(m + s, 1.0) for m, s in zip(avg_values, std_values)]
    lower = [max(m - s, 0.0) for m, s in zip(avg_values, std_values)]

    # ±1 std confidence band
    ax.fill_between(ks, lower, upper, color=color, alpha=0.18, linewidth=0)

    # Mean line
    ax.plot(
        ks, avg_values,
        color=color, alpha=1.0, linewidth=2.2,
        marker="o", markersize=4.5, markeredgecolor="white", markeredgewidth=0.6,
        zorder=10,
    )

    # Axis formatting
    ax.set_xlabel("$K$ (number of retrieved papers)", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, pad=10)
    ax.set_xticks(ks)
    ax.set_xlim(ks[0] - 0.3, ks[-1] + 0.3)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True)

    # Custom legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Line2D([0], [0], color=color, alpha=1.0, linewidth=2.2,
               marker="o", markersize=4.5, markeredgecolor="white", markeredgewidth=0.6,
               label=f"Mean ($n$={n_materials})"),
        Patch(facecolor=color, alpha=0.18, edgecolor="none",
              label="$\\pm$ 1 std"),
    ]
    ax.legend(handles=legend_elements, loc="best", frameon=True)

    fig.tight_layout()

    # Save PNG + PDF (PDF is vector, ideal for papers)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_metrics(
    results: List[MaterialEvaluation],
    metrics_summary: Dict[str, Any],
    output_dir: str,
    timestamp: str,
    num_papers: int = 10
):
    """
    Generate publication-quality Precision@K and Recall@K plots.

    Each ground-truth material is drawn as a semi-transparent line (alpha=0.3),
    and the mean across materials is drawn as a bold line (alpha=1).
    Each metric is saved to its own file (PNG + PDF).
    """
    _setup_publish_style()
    ks = list(range(1, num_papers + 1))
    n = len(results)

    # --- Precision@K ---
    prec_path = os.path.join(output_dir, f"precision_at_k_{timestamp}.png")
    _plot_single_metric(
        results, ks,
        metric_key="precision_at_k",
        avg_key_prefix="avg_precision",
        metrics_summary=metrics_summary,
        ylabel="Precision@$K$",
        title="Precision@$K$",
        color="#2171b5",
        output_path=prec_path,
        n_materials=n,
    )
    print(f"   📊 Precision plot saved to: {prec_path}  (.pdf also saved)")

    # --- Recall@K ---
    recall_path = os.path.join(output_dir, f"recall_at_k_{timestamp}.png")
    _plot_single_metric(
        results, ks,
        metric_key="recall_at_k",
        avg_key_prefix="avg_recall",
        metrics_summary=metrics_summary,
        ylabel="Recall@$K$",
        title="Recall@$K$",
        color="#d94801",
        output_path=recall_path,
        n_materials=n,
    )
    print(f"   📊 Recall plot saved to:    {recall_path}  (.pdf also saved)")


if __name__ == "__main__":
    run_evaluation(num_papers=10)
