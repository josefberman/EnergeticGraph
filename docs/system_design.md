# EnergeticGraph System Design

A modular molecular design system for optimizing energetic materials using beam search, RAG-enhanced modification strategies, and ML-based property prediction.

---

## High-Level Architecture

```mermaid
flowchart TB
    subgraph User_Interface["🖥️ User Interface Layer"]
        GUI["Flask Web GUI<br/>(gui/app.py)"]
        CLI["CLI Entry Point<br/>(main.py)"]
    end

    subgraph Core_Engine["⚙️ Core Engine"]
        Designer["EnergeticDesigner<br/>(designer.py)"]
        Orchestrator["BeamSearchEngine<br/>(orchestrator.py)"]
    end

    subgraph Agents["🤖 Agent Layer"]
        WorkerAgent["ChemistAgent<br/>(agents/worker_agent.py)"]
    end

    subgraph Modules["📦 Modules Layer"]
        direction TB
        RAG["RAGModificationStrategy<br/>(modules/rag_strategy.py)"]
        ModTools["Modification Tools<br/>(modules/modification_tools.py)"]
        Prediction["PropertyPredictor<br/>(modules/prediction.py)"]
        Feasibility["Feasibility Checker<br/>(modules/feasibility.py)"]
        Scoring["Scoring Functions<br/>(modules/scoring.py)"]
        Init["Initialization<br/>(modules/initialization.py)"]
    end

    subgraph External["🌐 External Services"]
        OpenAI["OpenAI API<br/>(GPT-4o-mini + Embeddings)"]
        Arxiv["Arxiv API<br/>(Paper Search)"]
    end

    subgraph Data["💾 Data Layer"]
        Models["XGBoost Models<br/>(models/*.joblib)"]
        Dataset["Molecule Dataset<br/>(sample_start_molecules.csv)"]
        ChromaDB["FAISS VectorStore<br/>(chroma_db/)"]
    end

    subgraph DataStructures["📋 Data Structures"]
        MolState["MoleculeState"]
        PropTarget["PropertyTarget"]
        Config["Config"]
    end

    %% User Interface connections
    GUI --> Designer
    CLI --> Designer

    %% Core Engine connections
    Designer --> Init
    Designer --> Orchestrator
    Orchestrator --> WorkerAgent

    %% Agent connections
    WorkerAgent --> RAG
    WorkerAgent --> ModTools
    WorkerAgent --> Prediction
    WorkerAgent --> Feasibility
    WorkerAgent --> Scoring

    %% RAG connections
    RAG --> OpenAI
    RAG --> Arxiv
    RAG --> ChromaDB
    RAG --> ModTools

    %% Module connections to data
    Prediction --> Models
    Init --> Dataset

    %% Data structures usage
    Designer -.-> MolState
    Designer -.-> PropTarget
    Designer -.-> Config
    Orchestrator -.-> MolState
    WorkerAgent -.-> MolState

    classDef interface fill:#4CAF50,stroke:#2E7D32,color:white
    classDef core fill:#2196F3,stroke:#1565C0,color:white
    classDef agent fill:#FF9800,stroke:#EF6C00,color:white
    classDef module fill:#9C27B0,stroke:#6A1B9A,color:white
    classDef external fill:#F44336,stroke:#C62828,color:white
    classDef data fill:#607D8B,stroke:#37474F,color:white
    classDef structure fill:#00BCD4,stroke:#00838F,color:white

    class GUI,CLI interface
    class Designer,Orchestrator core
    class WorkerAgent agent
    class RAG,ModTools,Prediction,Feasibility,Scoring,Init module
    class OpenAI,Arxiv external
    class Models,Dataset,ChromaDB data
    class MolState,PropTarget,Config structure
```

---

## Component Details

### 1. User Interface Layer

| Component | File | Description |
|-----------|------|-------------|
| **Flask Web GUI** | `gui/app.py` | Modern web interface with real-time progress updates via SSE, molecule visualization, and interactive parameter controls |
| **CLI Entry** | `main.py` | Command-line interface for batch processing and scripted execution |

---

### 2. Core Engine

```mermaid
flowchart LR
    subgraph Designer["EnergeticDesigner"]
        D1["initialize()"] --> D2["run_design_loop()"]
        D2 --> D3["get_results()"]
        D3 --> D4["save_results()"]
    end

    subgraph Orchestrator["BeamSearchEngine"]
        O1["run(seed)"] --> O2["For each iteration"]
        O2 --> O3["Create ChemistAgent"]
        O3 --> O4["Generate variations"]
        O4 --> O5["Filter feasible"]
        O5 --> O6["Rank by MAPE"]
        O6 --> O7["Prune to top_k"]
        O7 --> O2
    end

    Designer --> Orchestrator
```

#### EnergeticDesigner (`designer.py`)
The high-level orchestrator that:
- Loads molecular datasets
- Finds the best seed molecule using MAPE distance
- Runs the beam search optimization
- Saves results in JSON/CSV format

#### BeamSearchEngine (`orchestrator.py`)
Manages the beam search algorithm:
- Maintains a beam of top-K candidate molecules
- Creates worker agents for each parent molecule
- Evaluates and ranks candidates by MAPE
- Tracks best molecule ever found
- Detects convergence

---

### 3. Agent Layer

```mermaid
flowchart TB
    subgraph ChemistAgent["ChemistAgent Workflow"]
        A1["Analyze Property Gap<br/>(target - current)"] --> A2{"RAG Enabled?"}
        A2 -->|Yes| A3["RAG Modification Strategy"]
        A2 -->|No| A4["Default Heuristic Strategy"]
        A3 --> A5["Fill with Default Strategy<br/>(if needed)"]
        A4 --> A5
        A5 --> A6["Evaluate each candidate"]
        A6 --> A7["Predict properties"]
        A7 --> A8["Check feasibility"]
        A8 --> A9["Calculate score"]
        A9 --> A10["Return MoleculeState objects"]
    end
```

#### ChemistAgent (`agents/worker_agent.py`)
Worker agent that generates molecular variations:
- Analyzes property gaps between current and target
- Uses RAG or heuristic modification strategies
- Evaluates candidates with property prediction + feasibility

---

### 4. Modules Layer

#### 4.1 RAG Modification Strategy (`modules/rag_strategy.py`)

```mermaid
flowchart TB
    subgraph RAG["RAGModificationStrategy"]
        R1["Generate Arxiv Query<br/>(based on property gaps)"] --> R2["Search Arxiv Papers"]
        R2 --> R3["Chunk & Embed Papers<br/>(FAISS + OpenAI)"]
        R3 --> R4["Query RAG System<br/>(retrieve relevant context)"]
        R4 --> R5["Extract Modification Strategies<br/>(LLM analysis)"]
        R5 --> R6["Apply Modifications<br/>(SMARTS reactions)"]
        R6 --> R7["Return modified SMILES"]
    end
```

**Key Features:**
- Uses LangChain with ChatOpenAI (GPT-4o-mini)
- Searches Arxiv for relevant chemistry papers
- Chunks documents and creates FAISS vector store
- Extracts actionable modification strategies from literature
- Applies SMARTS-based chemical reactions

#### 4.2 Modification Tools (`modules/modification_tools.py`)

| Function | Description |
|----------|-------------|
| `addition_modification()` | Adds functional groups (-NO2, -NH2, -OH, etc.) |
| `subtraction_modification()` | Removes terminal atoms/groups |
| `substitution_modification()` | Substitutes atoms (C→N, H→F, etc.) |
| `ring_modification()` | Opens/closes rings, cyclization |
| `generate_diverse_modifications()` | Combines all strategies for diversity |

#### 4.3 Property Prediction (`modules/prediction.py`)

```mermaid
flowchart LR
    SMILES["SMILES"] --> Desc["RDKit Descriptors<br/>(descriptors.py)"]
    Desc --> XGB1["XGBoost<br/>density.joblib"]
    Desc --> XGB2["XGBoost<br/>det_velocity.joblib"]
    Desc --> XGB3["XGBoost<br/>det_pressure.joblib"]
    Desc --> XGB4["XGBoost<br/>hf_solid.joblib"]
    XGB1 --> Props["Predicted Properties"]
    XGB2 --> Props
    XGB3 --> Props
    XGB4 --> Props
```

**Predicted Properties:**
- **Density** (g/cm³)
- **Detonation Velocity** (m/s)
- **Detonation Pressure** (GPa)
- **Heat of Formation** (kJ/mol)

#### 4.4 Feasibility Checker (`modules/feasibility.py`)

```mermaid
flowchart TB
    SMILES["SMILES"] --> V["check_valency()<br/>(RDKit sanitization)"]
    V -->|Invalid| F1["Return (0.0, False)"]
    V -->|Valid| SA["calculate_sascore()<br/>(Synthetic Accessibility)"]
    SA --> Conv["Convert SAScore 1-10<br/>to Feasibility 0-1"]
    Conv --> F2["Return (feasibility, is_feasible)"]
```

**SAScore Interpretation:**
| SAScore | Feasibility | Interpretation |
|---------|-------------|----------------|
| 1-3 | 90-100% | Easy synthesis |
| 3-5 | 70-90% | Moderate |
| 5-7 | 50-70% | Challenging but doable |
| 7-10 | 0-50% | Very difficult |

#### 4.5 Scoring Functions (`modules/scoring.py`)

```
Total Score = 0.7 × (MAPE/100) + 0.3 × (1 - Feasibility)
```

Where **MAPE** (Mean Absolute Percentage Error) measures distance from target properties.

---

### 5. Data Structures (`data_structures.py`)

```mermaid
classDiagram
    class MoleculeState {
        +str smiles
        +Dict properties
        +float score
        +float feasibility
        +str provenance
        +bool is_feasible
        +int generation
        +str parent_smiles
        +to_dict()
        +from_dict()
    }

    class PropertyTarget {
        +float density
        +float det_velocity
        +float det_pressure
        +float hf_solid
        +to_dict()
    }

    class Config {
        +BeamSearchConfig beam_search
        +ScoringConfig scoring
        +RAGConfig rag
        +SystemConfig system
    }

    class BeamSearchConfig {
        +int beam_width = 10
        +int top_k = 5
        +int max_iterations = 20
        +float convergence_threshold = 0.001
    }

    class RAGConfig {
        +bool enable_rag = True
        +int arxiv_max_results = 20
        +str openai_api_key
        +str llm_model = "gpt-4o-mini"
    }

    Config --> BeamSearchConfig
    Config --> RAGConfig
```

---

## Data Flow

```mermaid
sequenceDiagram
    participant User
    participant GUI as Flask GUI
    participant Designer as EnergeticDesigner
    participant Init as Initialization
    participant Engine as BeamSearchEngine
    participant Agent as ChemistAgent
    participant RAG as RAGStrategy
    participant Pred as PropertyPredictor
    participant Feas as Feasibility

    User->>GUI: Set target properties
    GUI->>Designer: initialize()
    Designer->>Init: load_dataset()
    Init-->>Designer: DataFrame
    Designer->>Init: find_closest_match()
    Init-->>Designer: Seed MoleculeState

    User->>GUI: Start optimization
    GUI->>Designer: run_design_loop()
    Designer->>Engine: run(seed)

    loop Each Iteration
        loop Each parent in beam
            Engine->>Agent: generate_variations()
            Agent->>Agent: analyze_property_gap()
            Agent->>RAG: rag_modification_strategy()
            RAG->>RAG: Search Arxiv → Embed → Query → Extract
            RAG-->>Agent: Modified SMILES list

            loop Each candidate SMILES
                Agent->>Pred: predict_properties()
                Pred-->>Agent: Properties dict
                Agent->>Feas: calculate_feasibility()
                Feas-->>Agent: (score, is_feasible)
            end
            Agent-->>Engine: List[MoleculeState]
        end
        Engine->>Engine: Filter → Rank by MAPE → Prune to top_k
    end

    Engine-->>Designer: Best molecule
    Designer-->>GUI: Results
    GUI-->>User: Display best molecule + properties
```

---

## Directory Structure

```
EnergeticGraph/
├── 📄 main.py                 # CLI entry point
├── 📄 designer.py             # High-level EnergeticDesigner class
├── 📄 orchestrator.py         # BeamSearchEngine
├── 📄 data_structures.py      # MoleculeState, PropertyTarget, Config defs
├── 📄 config.py               # Configuration dataclasses
├── 📄 descriptors.py          # RDKit molecular descriptor generation
│
├── 📁 agents/
│   └── 📄 worker_agent.py     # ChemistAgent implementation
│
├── 📁 modules/
│   ├── 📄 rag_strategy.py     # RAG-based modification with Arxiv + LLM
│   ├── 📄 modification_tools.py # RDKit molecular modifications
│   ├── 📄 prediction.py       # XGBoost property prediction
│   ├── 📄 feasibility.py      # SAScore + valency checking
│   ├── 📄 scoring.py          # MAPE + feasibility scoring
│   └── 📄 initialization.py   # Dataset loading + seed selection
│
├── 📁 gui/
│   ├── 📄 app.py              # Flask web application
│   ├── 📁 templates/          # HTML templates
│   └── 📁 static/             # CSS, JS, images
│
├── 📁 models/                  # Trained XGBoost models
│   ├── 📄 density.joblib
│   ├── 📄 det_velocity.joblib
│   ├── 📄 det_pressure.joblib
│   └── 📄 hf_solid.joblib
│
├── 📁 chroma_db/               # FAISS vector store data
├── 📄 sample_start_molecules.csv # Seed molecule dataset
└── 📄 requirements.txt
```

---

## Key Algorithms

### Beam Search Optimization

```
Algorithm: Beam Search for Molecular Design
Input: seed_molecule, target_properties, beam_width, top_k, max_iterations
Output: best_molecule

1. beam ← [seed_molecule]
2. best_ever ← seed_molecule

3. for iteration = 1 to max_iterations:
    a. candidates ← []
    b. for each parent in beam:
        i.   agent ← ChemistAgent(parent, target_properties)
        ii.  variations ← agent.generate_variations()
        iii. candidates.extend(variations)
    
    c. feasible ← filter(candidates, λm: m.is_feasible)
    d. unique ← remove_duplicates(feasible)
    e. ranked ← sort(unique, by=MAPE, ascending=True)
    f. beam ← ranked[0:top_k]
    
    g. if MAPE(beam[0]) < MAPE(best_ever):
        best_ever ← beam[0]
    
    h. if converged(improvement < threshold):
        break

4. return best_ever
```

### MAPE Distance Calculation

```
MAPE = (100/n) × Σ |predicted_i - target_i| / |target_i|
```

---

## External Dependencies

| Dependency | Purpose |
|------------|---------|
| **RDKit** | Molecular structure handling, SMARTS reactions, descriptors |
| **XGBoost** | Property prediction models |
| **LangChain** | RAG pipeline orchestration |
| **OpenAI API** | GPT-4o-mini for LLM, text-embedding-3-small for embeddings |
| **FAISS** | Vector similarity search for RAG |
| **Flask** | Web application framework |
| **Arxiv API** | Scientific paper retrieval |

---

## Configuration Options

```python
# Beam Search Parameters
beam_width = 10          # Candidates to keep per iteration
top_k = 5                # Top candidates to select
max_iterations = 20      # Maximum search iterations
convergence_threshold = 0.001

# Scoring Weights
mape_weight = 0.7        # Property accuracy importance
feasibility_weight = 0.3 # Synthetic feasibility importance

# RAG Settings
enable_rag = True
arxiv_max_results = 20
llm_model = "gpt-4o-mini"
llm_temperature = 0.3
```
