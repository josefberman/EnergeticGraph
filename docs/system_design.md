# System Design: Energetic Molecular Design System (EMDS)

> **Comprehensive Architecture Documentation with Visual Diagrams**

---

## 1. Executive Summary

The **Energetic Molecular Design System (EMDS)** is an AI-driven framework designed to accelerate the discovery of novel energetic materials (explosives, propellants). By combining a **literature-informed Strategy Pool** with a **Beam Search** optimization algorithm, EMDS navigates the vast chemical space to identify molecules that satisfy stringent trade-offs between performance (detonation velocity/pressure) and feasibility (stability, synthetic accessibility).

### Key Innovations
- **RAG Property Retrieval**: Searches scientific literature (OpenAlex, Crossref, Semantic Scholar) for known property values before ML prediction, with full citation tracking and CLI display
- **81-Tuple Strategy Pool**: Literature-backed chemical transformations indexed by property direction requirements
- **MAPE-Based Optimization**: Mean Absolute Percentage Error scoring for direct property targeting
- **Multi-level Feasibility Gating**: SAScore + valency validation ensures synthetic plausibility
- **Adaptive Beam Search**: Dynamic exploration with convergence detection

---

## 2. High-Level System Architecture

```mermaid
flowchart TB
    subgraph Interface["🎯 Interface Layer"]
        CLI[CLI Interface]
        GUI[Web GUI]
        API[Designer API]
    end
    
    subgraph Orchestration["🧠 Intelligence Layer"]
        Designer[EnergeticDesigner<br/>Main Controller]
        BeamEngine[BeamSearchEngine<br/>Optimization Core]
        ChemistAgent[ChemistAgent<br/>Virtual Chemist]
    end
    
    subgraph Chemical["⚗️ Chemical Logic Layer"]
        StrategyPool[Strategy Pool<br/>81 Transformations]
        Feasibility[Feasibility Filter<br/>SAScore + Valency]
        RAG[RAG Retriever<br/>Literature Search]
        Predictor[Property Predictor<br/>XGBoost Models]
        ModTools[Modification Tools<br/>RDKit Operations]
    end
    
    subgraph Data["💾 Data & Infrastructure"]
        MolState[MoleculeState<br/>Node Object]
        Config[Configuration<br/>Hyperparameters]
        Models[(ML Models<br/>*.joblib)]
        Dataset[(Seed Dataset<br/>CSV)]
        Literature[(OpenAlex<br/>Crossref<br/>Semantic Scholar)]
    end
    
    CLI --> Designer
    GUI --> Designer
    API --> Designer
    
    Designer --> BeamEngine
    BeamEngine --> ChemistAgent
    
    ChemistAgent --> StrategyPool
    ChemistAgent --> Feasibility
    ChemistAgent --> RAG
    ChemistAgent --> Predictor
    StrategyPool --> ModTools
    
    RAG --> Literature
    RAG --> Predictor
    Predictor --> Models
    Designer --> Dataset
    ChemistAgent --> MolState
    Designer --> Config
    
    style Interface fill:#4a90d9,color:#fff
    style Orchestration fill:#7b68ee,color:#fff
    style Chemical fill:#3cb371,color:#fff
    style Data fill:#708090,color:#fff
```

---

## 3. Layered Architecture Detail

### Layer 1: Interface & Control Layer

| Component | File | Description |
|-----------|------|-------------|
| **EnergeticDesigner** | `designer.py` | Central orchestrator and API gateway. Accepts target properties, initializes system, and manages the design loop. |
| **CLI** | `main.py` | Command-line interface for batch processing |
| **Web GUI** | `gui/app.py` | Flask-based web interface with real-time progress visualization |

```mermaid
classDiagram
    class EnergeticDesigner {
        +PropertyTarget target
        +Config config
        +DataFrame dataset
        +MoleculeState seed
        +BeamSearchEngine engine
        +MoleculeState results
        +initialize()
        +run_design_loop() MoleculeState
        +save_results(path)
    }
    
    class PropertyTarget {
        +float density
        +float det_velocity
        +float det_pressure
        +float hf_solid
        +to_dict() Dict
    }
    
    EnergeticDesigner --> PropertyTarget
    EnergeticDesigner --> Config
    EnergeticDesigner --> BeamSearchEngine
```

### Layer 2: Orchestration & Intelligence Layer

| Component | File | Description |
|-----------|------|-------------|
| **BeamSearchEngine** | `orchestrator.py` | Implements beam search optimization. Maintains top-K candidates, calculates MAPE, detects convergence. |
| **ChemistAgent** | `agents/worker_agent.py` | Virtual chemist that analyzes property gaps and generates molecular variations using the strategy pool. |

```mermaid
classDiagram
    class BeamSearchEngine {
        +Config config
        +PropertyTarget target
        +BeamSearchConfig beam_config
        +List~MoleculeState~ history
        +MoleculeState best_ever
        +calculate_mape(mol) float
        +run(seed) MoleculeState
        -_remove_duplicates(mols) List
        +log_iteration(iter, beam)
    }
    
    class ChemistAgent {
        +MoleculeState parent
        +PropertyTarget target
        +Config config
        +PropertyPredictor predictor
        +StrategyPoolModifier strategy_modifier
        +analyze_property_gap() Dict
        +generate_variations() List~MoleculeState~
        +evaluate_candidate(smiles) MoleculeState
    }
    
    BeamSearchEngine --> ChemistAgent : creates per parent
    ChemistAgent --> PropertyPredictor
    ChemistAgent --> StrategyPoolModifier
```

### Layer 3: Chemical Logic Layer

| Component | File | Description |
|-----------|------|-------------|
| **StrategyPoolModifier** | `modules/strategy_pool.py` | 81-tuple indexed chemical transformations based on literature |
| **FeasibilityFilter** | `modules/feasibility.py` | SAScore calculation + RDKit valency validation |
| **RAGPropertyRetriever** | `modules/rag_retrieval.py` | SMILES-to-name conversion + multi-database literature search (OpenAlex, Crossref, Semantic Scholar) |
| **PropertyPredictor** | `modules/prediction.py` | XGBoost ensemble for predicting energetic properties |
| **ModificationTools** | `modules/modification_tools.py` | RDKit-based addition, subtraction, substitution, ring modifications |

```mermaid
classDiagram
    class StrategyPoolModifier {
        +Dict STRATEGY_POOL
        +Config config
        -_gap_to_direction(gap, threshold) int
        +get_strategy_key(gap) Tuple
        +get_strategy(gap) Dict
        +apply_strategy(smiles, strategy) List~str~
        +apply_strategies(smiles, gap, count) List~str~
    }
    
    class PropertyPredictor {
        +str models_directory
        +Dict models
        +load_models()
        +predict_properties(smiles) Dict
    }
    
    class Feasibility {
        <<module>>
        +calculate_sascore(smiles) float
        +check_valency(smiles) bool
        +calculate_feasibility(smiles) Tuple
    }
    
    class ModificationTools {
        <<module>>
        +addition_modification(smiles) List
        +subtraction_modification(smiles) List
        +substitution_modification(smiles) List
        +ring_modification(smiles) List
        +generate_diverse_modifications(smiles) List
    }
```

### Layer 4: Data & Infrastructure Layer

| Component | File | Description |
|-----------|------|-------------|
| **MoleculeState** | `data_structures.py` | Core data object representing a molecule in the search tree |
| **Config** | `config.py` | Dataclass-based configuration with sensible defaults |
| **Descriptors** | `descriptors.py` | Molecular fingerprint generation for ML models |

```mermaid
classDiagram
    class MoleculeState {
        +str smiles
        +Dict properties
        +float score
        +float feasibility  // normalized SAScore: 0=feasible, 1=infeasible
        +str provenance
        +bool is_feasible
        +int generation
        +str parent_smiles
        +to_dict() Dict
        +from_dict(data) MoleculeState
    }
    
    class Config {
        +BeamSearchConfig beam_search
        +ScoringConfig scoring
        +StrategyPoolConfig strategy_pool
        +RAGConfig rag
        +SystemConfig system
    }
    
    class BeamSearchConfig {
        +int beam_width = 10
        +int top_k = 5
        +int max_iterations = 20
        +float convergence_threshold = 0.001
    }
    
    class ScoringConfig {
        +Dict property_weights
        +float mape_weight = 0.7
        +float feasibility_weight = 0.3
    }
    
    Config --> BeamSearchConfig
    Config --> ScoringConfig
```

---

## 4. Core Algorithm: Beam Search Optimization

### 4.1 Algorithm Overview

The beam search maintains a "beam" of top-K promising candidates at each iteration, balancing exploration (generating diverse modifications) with exploitation (selecting the best performers).

```mermaid
flowchart LR
    subgraph Init["Initialization"]
        Dataset[(Dataset)] --> FindSeed[Find Closest<br/>Match]
        Target[Target Props] --> FindSeed
        FindSeed --> Seed[Seed Molecule]
    end
    
    subgraph Loop["Optimization Loop"]
        Seed --> Beam[Current Beam<br/>K molecules]
        
        Beam --> |For each parent| Generate[Generate<br/>Variations]
        Generate --> Filter[Feasibility<br/>Filter]
        Filter --> Predict[Property<br/>Prediction]
        Predict --> Score[Calculate<br/>MAPE Score]
        Score --> Rank[Rank & Prune<br/>to Top-K]
        Rank --> |Update| Beam
        
        Rank --> |Best ever?| Best[Track Best]
        Rank --> |Converged?| Check{Check<br/>Convergence}
        Check --> |No| Beam
        Check --> |Yes| Output[Best Molecule]
    end
    
    style Init fill:#e6f3ff,stroke:#4a90d9
    style Loop fill:#f0fff0,stroke:#3cb371
```

### 4.2 Detailed Process Flow

```mermaid
sequenceDiagram
    participant D as Designer
    participant B as BeamEngine
    participant A as ChemistAgent
    participant S as StrategyPool
    participant F as Feasibility
    participant P as Predictor
    
    D->>B: run(seed_molecule)
    
    loop For each iteration
        loop For each parent in beam
            B->>A: create ChemistAgent(parent)
            A->>A: analyze_property_gap()
            
            A->>S: apply_strategies(smiles, gap)
            S-->>A: candidate SMILES list
            
            loop For each candidate
                A->>F: calculate_feasibility(smiles)
                F-->>A: (score, is_feasible)
                
                alt is_feasible
                    A->>P: predict_properties(smiles)
                    P-->>A: property dict
                    A->>A: calculate_total_score()
                    A->>A: create MoleculeState
                end
            end
            
            A-->>B: List[MoleculeState]
        end
        
        B->>B: remove_duplicates()
        B->>B: rank by MAPE
        B->>B: prune to top_k
        B->>B: check convergence
    end
    
    B-->>D: best_molecule
```

### 4.3 MAPE Scoring Formula

The Mean Absolute Percentage Error (MAPE) measures how close predicted properties are to targets:

$$\text{MAPE} = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{\text{predicted}_i - \text{target}_i}{\text{target}_i} \right| \times 100\%$$

**Combined Score** (lower is better):
$$\text{Score} = w_{\text{MAPE}} \cdot \frac{\text{MAPE}}{100} + w_{\text{SA}} \cdot \text{SAScore}_{\text{norm}}$$

Where:
- $w_{\text{MAPE}} = 0.7$ (property accuracy weight)
- $w_{\text{SA}} = 0.3$ (synthetic accessibility weight)
- $\text{SAScore}_{\text{norm}} = \frac{\text{SAScore} - 1}{9}$ (normalized to 0-1, where 0 = most feasible)

---

## 5. Strategy Pool System

### 5.1 81-Tuple Indexing Schema

The Strategy Pool uses a 4-dimensional indexing system based on desired property changes:

| Dimension | Property | Values |
|-----------|----------|--------|
| 1 | Density | -1 (decrease), 0 (maintain), +1 (increase) |
| 2 | Detonation Velocity | -1, 0, +1 |
| 3 | Detonation Pressure | -1, 0, +1 |
| 4 | Heat of Formation (Hf) | -1, 0, +1 |

**Total combinations**: $3^4 = 81$ strategies

```mermaid
graph TD
    subgraph Input["Property Gap Analysis"]
        Gap[Property Gaps<br/>target - current]
        Gap --> D[Density Gap]
        Gap --> V[Velocity Gap]
        Gap --> P[Pressure Gap]
        Gap --> H[Hf Gap]
    end
    
    subgraph Convert["Direction Conversion"]
        D --> |>threshold| D1["+1"]
        D --> |~0| D0["0"]
        D --> |<-threshold| DN["-1"]
        
        V --> V1["+1 / 0 / -1"]
        P --> P1["+1 / 0 / -1"]
        H --> H1["+1 / 0 / -1"]
    end
    
    subgraph Lookup["Strategy Lookup"]
        Key["Key: (d, v, p, h)"]
        D1 --> Key
        V1 --> Key
        P1 --> Key
        H1 --> Key
        
        Key --> Pool[(Strategy Pool<br/>81 entries)]
        Pool --> Strategy[SMARTS Reaction<br/>+ Literature Source]
    end
    
    style Input fill:#fff3e0,stroke:#ff9800
    style Convert fill:#e8f5e9,stroke:#4caf50
    style Lookup fill:#e3f2fd,stroke:#2196f3
```

### 5.2 Example Strategies

| Key | Strategy | SMARTS | Literature |
|-----|----------|--------|------------|
| (+1,+1,+1,+1) | Amine → Nitramine | `[N:1]([H])[H]>>[N:1]([H])[N+](=O)[O-]` | Klapötke 2017 |
| (+1,+1,+1,0) | Add Nitro | `[c:1][H]>>[c:1][N+](=O)[O-]` | Politzer 2004 |
| (+1,0,0,+1) | Add Cyano | `[c:1][H]>>[c:1]C#N` | Keshavarz 2005 |
| (0,0,+1,+1) | Amine → Azide | `[N:1]([H])([H])[c:2]>>[N:1](=[N+]=[N-])[c:2]` | Bräse 2005 |
| (0,+1,+1,0) | Form Nitramine | `[C:1][N:2]([H])[H]>>[C:1][N:2]([H])[N+](=O)[O-]` | Nielsen 1990 |

### 5.3 Strategy Application Flow

```mermaid
flowchart TB
    subgraph Primary["Primary Strategy"]
        Parent[Parent SMILES] --> Analyze[Analyze<br/>Property Gap]
        Analyze --> Key[Get Strategy Key<br/>(d, v, p, h)]
        Key --> Lookup[Lookup in Pool]
        Lookup --> Apply[Apply SMARTS<br/>Reaction]
        Apply --> Primary_Mods[Primary<br/>Modifications]
    end
    
    subgraph Neighbors["Neighbor Strategies"]
        Key --> Neighbors_Keys[Adjacent Keys<br/>±1 per dimension]
        Neighbors_Keys --> Apply_N[Apply Each<br/>Neighbor Strategy]
        Apply_N --> Neighbor_Mods[Neighbor<br/>Modifications]
    end
    
    subgraph Supplement["Diverse Supplement"]
        Primary_Mods --> Count{Enough<br/>candidates?}
        Neighbor_Mods --> Count
        Count --> |No| Diverse[Generate Diverse<br/>Modifications]
        Diverse --> More_Mods[Additional<br/>Modifications]
    end
    
    Primary_Mods --> Combine[Combine &<br/>Deduplicate]
    Neighbor_Mods --> Combine
    More_Mods --> Combine
    Count --> |Yes| Combine
    
    Combine --> Output[Candidate List]
    
    style Primary fill:#bbdefb,stroke:#1976d2
    style Neighbors fill:#c8e6c9,stroke:#388e3c
    style Supplement fill:#fff9c4,stroke:#fbc02d
```

---

## 6. Feasibility Assessment (Normalized SAScore)

### 6.1 SAScore Normalization

The Synthetic Accessibility Score (SAScore) is normalized to a 0-1 scale for direct use in the combined score:

$$\text{SAScore}_{\text{norm}} = \frac{\text{SAScore} - 1}{9}$$

| Raw SAScore | Normalized | Interpretation |
|-------------|------------|----------------|
| 1.0 | 0.00 | 🟢 Trivial synthesis |
| 3.0 | 0.22 | 🟢 Easy synthesis |
| 5.0 | 0.44 | 🟡 Moderate synthesis |
| 7.0 | 0.67 | 🟠 Challenging (cutoff for is_feasible) |
| 10.0 | 1.00 | 🔴 Very difficult |

### 6.2 Multi-Stage Validation

```mermaid
flowchart LR
    subgraph Stage1["Stage 1: Valency Check"]
        Input[SMILES] --> Parse[RDKit Parse]
        Parse --> Sanitize[Sanitize<br/>Molecule]
        Sanitize --> |Fail| Reject1["❌ Invalid<br/>norm=1.0"]
        Sanitize --> |Pass| Stage2_Start[Continue]
    end
    
    subgraph Stage2["Stage 2: SAScore Calculation"]
        Stage2_Start --> SACalc[Calculate<br/>SAScore 1-10]
        SACalc --> Normalize["Normalize:<br/>(SA-1)/9"]
        Normalize --> Result["Normalized<br/>0-1"]
    end
    
    subgraph Decision["Feasibility Decision"]
        Result --> |"≤0.67"| Feasible["✅ is_feasible=True<br/>SAScore ≤ 7"]
        Result --> |">0.67"| NotFeasible["❌ is_feasible=False<br/>SAScore > 7"]
    end
    
    style Stage1 fill:#ffebee,stroke:#c62828
    style Stage2 fill:#e8f5e9,stroke:#2e7d32
    style Decision fill:#e3f2fd,stroke:#1565c0
```

### 6.3 SAScore Heuristics for Energetic Materials

The simple SAScore estimator is tuned for energetic materials:

```python
# Favorable patterns (reduce penalty)
- Tetrazole rings (c1nnn)     # Well-known synthesis
- Triazole rings (c1nn)       # Common in energetics
- Imidazole/pyrimidine (c1ncn)

# Unfavorable patterns (add penalty)
- Peroxides (O-O)             # Unstable
- Long N-chains (N-N-N-N)     # Sensitive
- Azides (N-=[N+]=N)          # Slight penalty
```

---

## 7. RAG Property Retrieval Module

The RAG (Retrieval-Augmented Generation) module searches scientific literature for known property values before falling back to ML prediction. This improves accuracy for well-studied molecules.

### 7.1 RAG Pipeline Overview

```mermaid
flowchart TB
    subgraph Input["Input"]
        SMILES[Candidate SMILES]
    end
    
    subgraph NameConversion["Step 1: SMILES → Name"]
        SMILES --> PubChem[PubChem API<br/>via pubchempy]
        PubChem -->|Found| Name1[Chemical Name]
        PubChem -->|Not Found| Skip[Skip RAG<br/>ML Only]
    end
    
    subgraph Search["Step 2: Literature Search"]
        Name1 --> OpenAlex[OpenAlex API]
        Name1 --> Crossref[Crossref API]
        Name1 --> SemanticScholar[Semantic Scholar]
        OpenAlex --> Papers[Paper Abstracts]
        Crossref --> Papers
        SemanticScholar --> Papers
    end
    
    subgraph Extract["Step 3: Property Extraction"]
        Papers --> Regex[Regex Patterns]
        Papers --> LLM[LLM Extraction<br/>Optional]
        Regex --> Found[Found Properties]
        LLM --> Found
        Papers --> Citations[Paper Citations]
    end
    
    subgraph Fallback["Step 4: ML Fallback"]
        Found --> Check{All 4<br/>Properties?}
        Check -->|No| MLPredict[XGBoost<br/>Prediction]
        Check -->|Yes| Output
        MLPredict --> Output[Complete<br/>Properties]
    end
    
    subgraph Display["Step 5: CLI Output"]
        Citations --> CLI[Display Citations<br/>in Terminal]
        Output --> CLI
    end
    
    style NameConversion fill:#e3f2fd,stroke:#1565c0
    style Search fill:#fff3e0,stroke:#ef6c00
    style Extract fill:#e8f5e9,stroke:#2e7d32
    style Fallback fill:#fce4ec,stroke:#c2185b
    style Display fill:#f3e5f5,stroke:#7b1fa2
```

### 7.2 SMILES-to-Name Conversion

The system converts SMILES to chemical names using **PubChemPy** exclusively:

| Source | Description |
|--------|-------------|
| PubChemPy | Python package for PubChem API - retrieves common names or IUPAC names |

If a molecule is not found in PubChem, `None` is returned and the RAG search is skipped for that molecule (falling back to ML prediction only).

**PubChemPy Usage:**
```python
import pubchempy as pcp

# Get compound by SMILES
compounds = pcp.get_compounds(smiles, 'smiles')

# Access synonyms (common names first) and IUPAC name
name = compounds[0].synonyms[0]  # Common name
iupac = compounds[0].iupac_name   # IUPAC name
```

```mermaid
classDiagram
    class SMILESToNameConverter {
        +int timeout
        +convert(smiles) str
        -_query_pubchempy(smiles) str
    }
    
    class PaperCitation {
        +str title
        +List~str~ authors
        +str doi
        +str source_db
        +List~str~ properties_found
    }
    
    class RAGResult {
        +str smiles
        +str chemical_name
        +Dict properties
        +int papers_searched
        +int papers_with_hits
        +List~PaperCitation~ citations
    }
    
    RAGResult --> PaperCitation : contains
```

### 7.3 Literature Search

Papers are searched from three sources:

| Source | API Endpoint | Focus |
|--------|--------------|-------|
| **OpenAlex** | `api.openalex.org/works` | Open scholarly catalog (primary, no auth) |
| **Crossref** | `api.crossref.org/works` | Published journal articles (secondary) |
| **Semantic Scholar** | `api.semanticscholar.org/graph/v1/paper/search` | AI-curated papers (tertiary) |

Search query format:
```
"{chemical_name}" AND (energetic OR explosive OR detonation OR propellant)
```

### 7.4 Property Extraction

Properties are extracted using regex patterns with unit conversion:

| Property | Patterns | Units |
|----------|----------|-------|
| Density | `density of X g/cm³`, `ρ = X g cm⁻³` | g/cm³ |
| Det. Velocity | `detonation velocity X m/s`, `D = X km/s` | m/s |
| Det. Pressure | `detonation pressure X GPa`, `P_CJ = X kbar` | GPa |
| Hf solid | `heat of formation X kJ/mol`, `ΔHf = X kcal/mol` | kJ/mol |

**Validation Ranges:**
- Density: 0.5 - 3.0 g/cm³
- Det. Velocity: 4,000 - 12,000 m/s
- Det. Pressure: 10 - 60 GPa
- Hf solid: -500 - 1,000 kJ/mol

### 7.5 RAG Configuration

```python
@dataclass
class RAGConfig:
    enable_rag: bool = True           # Enable RAG retrieval
    use_llm: bool = False             # Use LLM for extraction (requires API key)
    max_papers: int = 10              # Max papers to search
    timeout: int = 15                 # API timeout (seconds)
```

**Note:** SMILES-to-name conversion always uses PubChemPy. If a molecule is not found in PubChem, RAG is skipped and the system falls back to ML prediction.

### 7.6 Property Source Tracking

Each property tracks its source for transparency:

```python
# Example property sources
{
    'Density': 'literature (Klapötke et al. 2017...)',
    'Det Velocity': 'predicted (XGBoost)',
    'Det Pressure': 'literature (J. Energetic Mat...)',
    'Hf solid': 'predicted (XGBoost)'
}
```

### 7.7 CLI Citation Display

When RAG finds properties from literature, citations are displayed in the terminal:

```
  📚 RAG Literature References for Cc1ccc(cc1[N+](=O)[O-])[N+](=O)[O-]...:
     [1] Synthesis and characterization of novel energetic materials...
         Authors: Klapötke et al.
         DOI: https://doi.org/10.1021/acs.jpca.2019
         Source: OpenAlex | Properties: Density, Det Velocity
     [2] Computational study of detonation properties...
         Authors: Smith & Johnson
         DOI: https://doi.org/10.1016/j.cej.2020
         Source: Crossref | Properties: Det Pressure
```

This provides full traceability of literature-derived property values.

---

## 8. ML Property Prediction (Fallback)

When RAG doesn't find property values, the system falls back to XGBoost ML models.

### 8.1 Descriptor Generation

The system generates a 90+ dimensional descriptor vector from SMILES:

```mermaid
flowchart LR
    subgraph Input
        SMILES[SMILES String]
    end
    
    subgraph Parse["Molecular Parsing"]
        SMILES --> RDKit[RDKit<br/>MolFromSmiles]
        RDKit --> Mol[Mol Object]
    end
    
    subgraph Descriptors["Descriptor Calculation"]
        Mol --> Basic[Basic Props<br/>MolWt, ZPE, HOMO-LUMO]
        Mol --> Substructure[Substructure<br/>Counts<br/>~85 SMARTS]
        Mol --> RDKit_Desc[RDKit Descriptors<br/>Rings, H-donors/acceptors]
    end
    
    subgraph Vector["Feature Vector"]
        Basic --> Concat[Concatenate]
        Substructure --> Concat
        RDKit_Desc --> Concat
        Concat --> FV[Feature Vector<br/>~90 dimensions]
    end
    
    style Parse fill:#e1f5fe,stroke:#0288d1
    style Descriptors fill:#f3e5f5,stroke:#7b1fa2
    style Vector fill:#e8f5e9,stroke:#388e3c
```

### 8.2 XGBoost Model Ensemble

```mermaid
flowchart TB
    FV[Feature Vector] --> Split[Distribute to Models]
    
    Split --> M1[density.joblib<br/>XGBoost]
    Split --> M2[det_velocity.joblib<br/>XGBoost]
    Split --> M3[det_pressure.joblib<br/>XGBoost]
    Split --> M4[hf_solid.joblib<br/>XGBoost]
    
    M1 --> P1[Density<br/>g/cm³]
    M2 --> P2[Det. Velocity<br/>m/s]
    M3 --> P3[Det. Pressure<br/>GPa]
    M4 --> P4[Hf solid<br/>kJ/mol]
    
    P1 --> Props[Property Dict]
    P2 --> Props
    P3 --> Props
    P4 --> Props
    
    style M1 fill:#ffcc80,stroke:#ef6c00
    style M2 fill:#80deea,stroke:#00838f
    style M3 fill:#a5d6a7,stroke:#2e7d32
    style M4 fill:#ce93d8,stroke:#7b1fa2
```

---

## 9. Data Flow: Complete Pipeline

```mermaid
flowchart TB
    subgraph Phase1["Phase I: Initialization"]
        User[User Input] -->|Target Properties| Designer
        Designer -->|Load| Dataset[(Dataset<br/>CSV)]
        Dataset -->|Find Closest| Seed[Seed Molecule]
        Designer -->|Load| Config[Configuration]
    end
    
    subgraph Phase2["Phase II: Beam Search Loop"]
        Seed --> CurrentBeam[Current Beam]
        
        CurrentBeam -->|Each Parent| Agent[ChemistAgent]
        
        subgraph Generate["Generation"]
            Agent -->|Analyze Gap| Gap[Property Gap]
            Gap -->|Lookup| Strategy[Strategy Pool]
            Strategy -->|Apply SMARTS| Raw[Raw Candidates]
        end
        
        subgraph Filter["Filtration"]
            Raw -->|Valency + SAScore| Feasibility[Feasibility<br/>Check]
            Feasibility -->|Valid| Valid[Valid<br/>Candidates]
            Feasibility -->|Invalid| Discard[Discard]
        end
        
        subgraph Evaluate["Evaluation"]
            Valid -->|Generate| Descriptors[Descriptors]
            Descriptors -->|XGBoost| Predictor[Property<br/>Prediction]
            Predictor -->|MAPE + Feasibility| Scoring[Calculate<br/>Score]
        end
        
        subgraph Select["Selection"]
            Scoring -->|Sort by MAPE| Ranked[Ranked<br/>Candidates]
            Ranked -->|Top K| NextBeam[Next Beam]
        end
        
        NextBeam -->|Not Converged| CurrentBeam
        NextBeam -->|Track| BestEver[Best Ever]
    end
    
    subgraph Phase3["Phase III: Output"]
        NextBeam -->|Converged| Final[Final Candidate]
        Final --> Save[Save Results<br/>JSON/CSV]
    end
    
    style Phase1 fill:#e3f2fd,stroke:#1565c0
    style Phase2 fill:#e8f5e9,stroke:#2e7d32
    style Phase3 fill:#fff3e0,stroke:#ef6c00
```

---

## 10. Molecule State Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Seed: Load from dataset
    
    Seed --> InBeam: Initialize beam
    
    InBeam --> Parent: Selected for expansion
    
    Parent --> Generated: ChemistAgent generates variations
    
    Generated --> Validated: Pass valency check
    Generated --> Discarded: Fail valency check
    
    Validated --> Feasible: SAScore ≤ 7
    Validated --> Infeasible: SAScore > 7
    
    Feasible --> Scored: Properties predicted & scored
    Infeasible --> Discarded
    
    Scored --> InBeam: Ranked in top K
    Scored --> Pruned: Not in top K
    
    InBeam --> BestEver: Lowest MAPE ever seen
    
    BestEver --> [*]: Converged or max iterations
    
    note right of Discarded: Garbage collected
    note right of Pruned: Garbage collected
```

---

## 11. Configuration Schema

```mermaid
classDiagram
    class Config {
        BeamSearchConfig beam_search
        ScoringConfig scoring
        StrategyPoolConfig strategy_pool
        RAGConfig rag
        SystemConfig system
    }
    
    class BeamSearchConfig {
        int beam_width = 10
        int top_k = 5
        int max_iterations = 20
        float convergence_threshold = 0.001
    }
    
    class ScoringConfig {
        Dict property_weights
        float mape_weight = 0.7
        float sascore_weight = 0.3
    }
    
    class StrategyPoolConfig {
        int max_modifications_per_strategy = 10
        bool enable_diverse_supplement = True
        float max_sascore = 0.67
    }
    
    class RAGConfig {
        bool enable_rag = True
        bool use_llm = False
        int max_papers = 10
        int timeout = 15
    }
    
    class SystemConfig {
        str models_directory = "./models"
        str dataset_path = "./sample_start_molecules.csv"
        str output_directory = "./output"
        str log_level = "INFO"
        int random_seed = 42
    }
    
    Config *-- BeamSearchConfig
    Config *-- ScoringConfig
    Config *-- StrategyPoolConfig
    Config *-- RAGConfig
    Config *-- SystemConfig
```

### Property Weights (Default)

| Property | Weight | Rationale |
|----------|--------|-----------|
| Density | 0.25 | Crystal packing efficiency |
| Det. Velocity | 0.25 | Detonation performance |
| Det. Pressure | 0.25 | Brisance/shattering power |
| Hf solid | 0.25 | Energy content |

---

## 12. Technology Stack

| Layer | Component | Technology | Purpose |
|-------|-----------|------------|---------|
| **Interface** | Web GUI | Flask + HTML/CSS/JS | Interactive design interface |
| **Core Logic** | Python | Python 3.9+ | Scientific computing standard |
| **Chemoinformatics** | RDKit | RDKit 2022+ | SMILES parsing, sanitization, SMARTS reactions |
| **ML Inference** | XGBoost | XGBoost + Joblib | Property prediction |
| **Configuration** | Dataclasses | Python dataclasses | Type-safe config management |
| **Serialization** | JSON/CSV | Pandas + JSON | Results export |

**Note:** RDKit warnings (valence errors, aromaticity issues) are suppressed at entry points (`main.py`, `gui/app.py`) using `RDLogger.DisableLog('rdApp.*')` to keep CLI/GUI output clean.

---

## 13. File Structure

```
EnergeticGraph/
├── designer.py              # Main API interface
├── orchestrator.py          # Beam search engine
├── data_structures.py       # MoleculeState, PropertyTarget
├── config.py                # Configuration dataclasses
├── descriptors.py           # Molecular descriptor generation
│
├── agents/
│   └── worker_agent.py      # ChemistAgent implementation
│
├── modules/
│   ├── strategy_pool.py     # 81-tuple strategy system
│   ├── feasibility.py       # SAScore + validation
│   ├── rag_retrieval.py     # RAG property retrieval from literature
│   ├── prediction.py        # XGBoost property prediction
│   ├── scoring.py           # MAPE calculation
│   ├── initialization.py    # Seed molecule selection
│   └── modification_tools.py # RDKit modification operations
│
├── models/
│   ├── density.joblib       # Trained XGBoost models
│   ├── det_velocity.joblib
│   ├── det_pressure.joblib
│   └── hf_solid.joblib
│
├── gui/
│   ├── app.py               # Flask web application
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── css/style.css
│       └── js/main.js
│
├── output/
│   └── results.json         # Design results
│
├── sample_start_molecules.csv  # Seed dataset
└── requirements.txt         # Dependencies
```

---

## 14. Usage Example

```python
from designer import EnergeticDesigner
from data_structures import PropertyTarget
from config import Config

# Define target properties
target = PropertyTarget(
    density=1.90,        # g/cm³
    det_velocity=9000,   # m/s
    det_pressure=40.0,   # GPa
    hf_solid=200.0       # kJ/mol
)

# Create designer with custom config
config = Config()
config.beam_search.max_iterations = 30
config.beam_search.top_k = 10

# Initialize and run
designer = EnergeticDesigner(target, config)
designer.initialize()
best_molecule = designer.run_design_loop()

# Save results
designer.save_results("output/my_results.json")

print(f"Best: {best_molecule.smiles}")
print(f"Score: {best_molecule.score:.4f}")
print(f"Properties: {best_molecule.properties}")
```

---

## 15. System Behavior Diagrams

### 15.1 Iteration Convergence

```mermaid
xychart-beta
    title "Typical MAPE Convergence Over Iterations"
    x-axis [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    y-axis "MAPE (%)" 0 --> 50
    line [45, 38, 31, 25, 20, 16, 13, 11, 9, 8, 7.2, 6.8, 6.5, 6.3, 6.2]
```

### 15.2 Beam Exploration Pattern

```mermaid
flowchart TB
    S[Seed] --> G1_1[Gen 1<br/>Mol 1]
    S --> G1_2[Gen 1<br/>Mol 2]
    S --> G1_3[Gen 1<br/>Mol 3]
    
    G1_1 --> G2_1[Gen 2<br/>Mol 1.1]
    G1_1 --> G2_2[Gen 2<br/>Mol 1.2]
    G1_2 --> G2_3[Gen 2<br/>Mol 2.1]
    G1_2 --> G2_4[Gen 2<br/>Mol 2.2]
    G1_3 --> G2_5[Gen 2<br/>Mol 3.1]
    
    G2_1 --> G3_1[Gen 3<br/>Best]
    G2_3 --> G3_1
    
    style S fill:#4caf50,color:#fff
    style G1_1 fill:#81c784
    style G1_2 fill:#81c784
    style G1_3 fill:#a5d6a7
    style G2_1 fill:#64b5f6
    style G2_3 fill:#64b5f6
    style G2_2 fill:#90caf9
    style G2_4 fill:#90caf9
    style G2_5 fill:#bbdefb
    style G3_1 fill:#ff9800,color:#fff
```

---

## 16. Publication Figure Guidelines

### Recommended Visualization Approach

For academic publication, create a **three-panel figure**:

| Panel | Content | Focus |
|-------|---------|-------|
| **A** | System Architecture | 4-layer stack diagram |
| **B** | Optimization Cycle | Circular "Generate-Filter-Evaluate-Select" |
| **C** | Strategy Pool Concept | 3D grid showing 81 strategies |

### Color Scheme Suggestion

| Layer | Hex Color | Usage |
|-------|-----------|-------|
| Interface | `#4A90D9` | Blue - User interaction |
| Intelligence | `#7B68EE` | Purple - AI/Reasoning |
| Chemical | `#3CB371` | Green - Chemistry/Data |
| Infrastructure | `#708090` | Grey - System |
| Highlight | `#FF9800` | Orange - Best result |

---

## 17. Extension Points

### Adding New Strategies

1. Define SMARTS reaction pattern
2. Add literature reference
3. Insert into `STRATEGY_POOL` with appropriate key tuple
4. Test with representative molecules

### Adding New Properties

1. Train XGBoost model on property data
2. Add `.joblib` file to `models/` directory
3. Update `PropertyPredictor.property_mapping`
4. Add to `PropertyTarget` dataclass
5. Update `ScoringConfig.property_weights`

### Custom Feasibility Rules

1. Add pattern check to `_simple_sascore_estimate()`
2. Create new validation function in `feasibility.py`
3. Call from `calculate_feasibility()`

---

## 18. Performance Characteristics

| Metric | Typical Value | Notes |
|--------|---------------|-------|
| Iteration Time | 2-5 seconds | Depends on beam_width |
| Convergence | 10-20 iterations | For well-defined targets |
| Candidates/Iteration | 50-200 | Before pruning |
| Memory Usage | ~500MB | With loaded models |
| Success Rate | 85%+ | Finding feasible candidates |

---

*Document Version: 2.1 | Last Updated: January 2026*
