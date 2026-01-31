# Energetic Molecular Design System (EMDS)
## High-Level System Architecture

---

## System Overview

An AI-powered platform for discovering novel energetic materials (explosives, propellants) using intelligent search algorithms combined with literature-informed chemical transformations.

---

## Architecture Diagram Description

### Main Components (4 Layers)

```
┌─────────────────────────────────────────────────────────────────┐
│                      USER INTERFACE                              │
│                  (Web GUI / CLI / API)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   OPTIMIZATION ENGINE                            │
│                                                                  │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│   │    Beam     │───▶│   Chemist   │───▶│   Scoring   │        │
│   │   Search    │    │    Agent    │    │   Function  │        │
│   └─────────────┘    └─────────────┘    └─────────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   CHEMICAL INTELLIGENCE                          │
│                                                                  │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│   │  Strategy   │    │     RAG     │    │     ML      │        │
│   │    Pool     │    │  Retrieval  │    │  Predictor  │        │
│   │ (81 Rules)  │    │ (Literature)│    │  (XGBoost)  │        │
│   └─────────────┘    └─────────────┘    └─────────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EXTERNAL RESOURCES                            │
│                                                                  │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│   │   PubChem   │    │  OpenAlex   │    │  Crossref   │        │
│   │  Database   │    │   Papers    │    │   Papers    │        │
│   └─────────────┘    └─────────────┘    └─────────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Workflow

```
     ┌──────────────┐
     │ User Input:  │
     │ Target Props │
     └──────┬───────┘
            │
            ▼
     ┌──────────────┐
     │  Find Seed   │
     │   Molecule   │
     └──────┬───────┘
            │
            ▼
    ┌───────────────────────────────────────┐
    │           BEAM SEARCH LOOP            │
    │                                       │
    │  ┌─────────┐      ┌─────────────┐    │
    │  │ Generate│      │   Evaluate  │    │
    │  │ Variants│─────▶│  & Filter   │    │
    │  └─────────┘      └──────┬──────┘    │
    │       ▲                  │           │
    │       │                  ▼           │
    │       │           ┌─────────────┐    │
    │       └───────────│  Select     │    │
    │                   │  Top-K      │    │
    │                   └─────────────┘    │
    │                                       │
    └───────────────────────────────────────┘
            │
            ▼
     ┌──────────────┐
     │    Best      │
     │   Molecule   │
     └──────────────┘
```

---

## Key Components

| Component | Function |
|-----------|----------|
| **Beam Search** | Explores chemical space by maintaining top-K candidates |
| **Strategy Pool** | 81 literature-backed transformation rules |
| **RAG Retrieval** | Searches scientific papers for known property values |
| **ML Predictor** | XGBoost models predict 4 energetic properties |
| **Feasibility Filter** | SAScore ensures synthetic accessibility |

---

## Property Targets

| Property | Unit | Description |
|----------|------|-------------|
| Density | g/cm³ | Crystal packing efficiency |
| Detonation Velocity | m/s | Speed of detonation wave |
| Detonation Pressure | GPa | Explosive power/brisance |
| Heat of Formation | kJ/mol | Energy content |

---

## Data Flow

```
SMILES ──▶ Name Lookup ──▶ Literature Search ──▶ Property Extraction
   │            │                  │                     │
   │            ▼                  ▼                     ▼
   │       [PubChem]         [OpenAlex]            [Regex/LLM]
   │       [SMILES2IUPAC]    [Crossref]
   │                         [Semantic Scholar]
   │
   └──▶ ML Prediction (fallback for missing properties)
              │
              ▼
         [XGBoost Models]
              │
              ▼
         Final Properties
```

---

## Scoring Formula

**Combined Score** (minimize):

```
Score = 0.7 × MAPE + 0.3 × SAScore_normalized

Where:
- MAPE = Mean Absolute Percentage Error (vs targets)
- SAScore_normalized = (SAScore - 1) / 9  [0 = easy, 1 = hard]
```

---

## Technology Stack

- **Core**: Python 3.11, RDKit (chemoinformatics)
- **ML**: XGBoost, scikit-learn
- **RAG**: PubChemPy, SMILES2IUPAC (HuggingFace), OpenAlex/Crossref APIs
- **Interface**: Flask (Web GUI), CLI
- **Search**: Beam Search with 81-rule Strategy Pool

---

## Visual Concept for Presentation

**Suggested image prompt for AI generation:**

"A modern scientific workflow diagram showing an AI-powered molecular design system. Four horizontal layers connected by arrows: 1) User Interface at top (web browser icon), 2) Optimization Engine (gears and search icons), 3) Chemical Intelligence (molecule structures, books, brain icon), 4) External Databases at bottom (cloud/database icons). Clean tech aesthetic, blue and green color scheme, white background, vector style, professional presentation quality."

---

## One-Liner Summary

> **EMDS uses beam search optimization with literature-backed chemical transformations and RAG-enhanced property retrieval to design novel energetic materials matching user-specified targets.**
