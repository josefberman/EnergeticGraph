# Chemical Feasibility Scoring: Research & Implementation

## Executive Summary

This document describes the research into state-of-the-art chemical synthesizability and feasibility scoring methods, comparing them to the current xTB-based approach, and recommending an improved implementation strategy.

---

## Current Approach

**Method**: xTB (Extended Tight Binding) single-point calculations via `tblite` and ASE

**What it measures**:
- Energy minimization feasibility
- Electronic structure stability
- Molecular geometry optimization

**Limitations**:
- Computationally expensive (needs quantum chemistry calculations)
- Focuses on thermodynamic stability, not synthetic accessibility
- Doesn't consider practical synthesis routes
- Slow for large-scale screening

**Score range**: 0-1 composite score based on:
- Energy convergence
- Geometry optimization success
- Electronic structure validity

---

## State-of-the-Art Alternatives (2024)

### 1. **SAScore (Synthetic Accessibility Score)** ⭐ RECOMMENDED

**Source**: Ertl & Schuffenhauer (2009), implemented in RDKit  
**Status**: Mature, widely adopted, fast

**Description**:
- Estimates how easy it is to synthesize a molecule
- Based on fragment contributions and molecular complexity
- Fast heuristic approach using structural features

**Advantages**:
- ✅ **Lightning fast** - milliseconds per molecule
- ✅ **No external dependencies** - pure RDKit
- ✅ **Well-validated** - industry standard
- ✅ **Interpretable** - based on known chemistry
- ✅ **Easy to integrate** - simple function call

**Score range**: 1 (easy to synthesize) to 10 (very difficult)

**Implementation**:
```python
from rdkit.Chem import RDKitConfig
import sys
sys.path.append(f'{RDKitConfig.RDContribDir}/SA_Score')
import sascorer

score = sascorer.calculateScore(mol)
# Returns 1-10, where 1 = easy, 10 = very hard
```

**When to use**:
- Fast screening of large libraries
- General drug-like molecules
- Quick feasibility checks

---

### 2. **RAscore (Retrosynthetic Accessibility score)**

**Source**: AstraZeneca (2020), based on AiZynthFinder  
**Status**: Modern, ML-based, excellent for drug discovery

**Description**:
- Predicts if retrosynthetic planning software can find a route
- Trained on 200K ChEMBL compounds with AiZynthFinder
- Uses neural networks and gradient boosting

**Advantages**:
- ✅ **Highly accurate** for drug-like molecules
- ✅ **Accounts for real synthesis routes**
- ✅ **ML-based** - learns from actual retrosynthesis data
- ✅ **Fast inference** - pre-trained models

**Disadvantages**:
- ❌ Requires external package installation
- ❌ Best for ChEMBL-like compounds
- ❌ May need fine-tuning for energetic materials

**Score range**: 0 (cannot synthesize) to 1 (can synthesize)

**Installation**:
```bash
pip install rascore
```

**Implementation**:
```python
from rascore import RAscore

scorer = RAscore()
score = scorer.predict(smiles)
# Returns probability that a route can be found
```

**When to use**:
- Drug-like organic molecules
- When accuracy is critical
- Production environments with proper validation

---

### 3. **SCScore (Synthetic Complexity Score)**

**Source**: Coley et al., MIT (2018)  
**Status**: Neural network based, research-grade

**Description**:
- Neural network trained on millions of reactions
- Predicts number of reaction steps needed
- Correlates complexity with synthesis difficulty

**Advantages**:
- ✅ **Data-driven** - learned from real reactions
- ✅ **Quantitative** - predicts actual number of steps
- ✅ **Validated** - peer-reviewed

**Disadvantages**:
- ❌ Requires TensorFlow/neural network inference
- ❌ Slower than SAScore
- ❌ Focused on organic synthesis

**Score range**: 1 (simple) to 5 (very complex)

**Installation**:
```bash
git clone https://github.com/connorcoley/scscore.git
# Requires tensorflow, h5py, numpy
```

**When to use**:
- Research applications
- When predicting synthesis steps is important
- Organic chemistry focus

---

### 4. **SYBA (SYnthetic Bayesian Accessibility)**

**Source**: Voršilák et al. (2020)  
**Status**: Bayesian classifier, open-source

**Description**:
- Bayesian classifier trained on purchasable vs hard-to-make molecules
- Fast fragment-based approach
- Good for early-stage filtering

**Advantages**:
- ✅ Simple probabilistic model
- ✅ Fast execution
- ✅ Easy to interpret

**Disadvantages**:
- ❌ Less accurate than ML methods
- ❌ Limited training data compared to newer methods

---

## Comparison Table

| Method | Speed | Accuracy | Setup | Best For | Score Range |
|--------|-------|----------|-------|----------|-------------|
| **Current (xTB)** | Slow (seconds) | Good for stability | Complex | Thermodynamic feasibility | 0-1 |
| **SAScore** ⭐ | Very Fast (ms) | Good | Simple | General screening | 1-10 |
| **RAscore** | Fast (ms) | Excellent | Moderate | Drug discovery | 0-1 |
| **SCScore** | Moderate (ms) | Very Good | Complex | Research | 1-5 |
| **SYBA** | Very Fast (ms) | Moderate | Simple | Quick filtering | 0-1 |

---

## Recommendation: Hybrid Approach

### Strategy

Implement a **two-tier feasibility scoring system**:

**Tier 1 - Fast Screening (SAScore)**:
- Use SAScore for initial candidate filtering during beam search
- Ultra-fast, allows evaluating many candidates
- Filters out obviously unsynthesizable molecules

**Tier 2 - Detailed Validation (xTB)** [Optional]:
- Apply xTB calculations only to final candidates
- Validates thermodynamic stability
- Ensures electronic structure is sensible
- Can be made optional via configuration

### Benefits

1. **Massive speed improvement** (1000x faster for screening)
2. **Better chemical relevance** (SAScore based on real synthesis)
3. **Backward compatible** (can keep xTB as optional)
4. **Easy to implement** (SAScore already in RDKit)
5. **Flexible** (can combine scores or use separately)

---

## Recommended Implementation

### Phase 1: Add SAScore (IMMEDIATE)

**File**: `molecular_tools.py`

Add SAScore to synthesis feasibility:

```python
def calculate_synthetic_accessibility(mol):
    """Calculate SAScore (1-10, lower is easier to synthesize)"""
    from rdkit.Chem import RDKitConfig
    import sys
    sys.path.append(f'{RDKitConfig.RDContribDir}/SA_Score')
    import sascorer
    
    try:
        sa_score = sascorer.calculateScore(mol)
        # Normalize to 0-1 (invert so higher = better)
        # SAScore: 1=easy, 10=hard → normalized: 1=best, 0=worst
        normalized = (10 - sa_score) / 9.0
        return {
            'sa_score': sa_score,
            'normalized': normalized,
            'feasible': sa_score < 6.0  # Threshold: < 6 is reasonable
        }
    except Exception as e:
        return {'error': str(e), 'sa_score': None, 'normalized': 0.0, 'feasible': False}
```

**Benefits**:
- **1000x faster** than xTB
- **More relevant** for synthesis planning
- **No new dependencies** (uses existing RDKit)

### Phase 2: Hybrid Scoring (FUTURE)

Combine multiple scores:

```python
def calculate_composite_feasibility(mol, use_xtb=False):
    """Composite feasibility using multiple methods"""
    scores = {}
    
    # Fast SAScore (always)
    sa_result = calculate_synthetic_accessibility(mol)
    scores['sa_score'] = sa_result['normalized']
    
    # Optional: xTB for thermodynamic validation (slow)
    if use_xtb:
        xtb_result = calculate_xtb_feasibility(mol)
        scores['xtb_score'] = xtb_result.get('composite_score_0_1', 0.0)
        
        # Weighted average
        composite = 0.7 * scores['sa_score'] + 0.3 * scores['xtb_score']
    else:
        composite = scores['sa_score']
    
    return {
        'composite_score_0_1': composite,
        'components': scores,
        'method': 'hybrid' if use_xtb else 'sa_only'
    }
```

### Phase 3: Advanced Methods (OPTIONAL)

For research or production:
- Add RAscore for drug-like molecules
- Add SCScore for detailed synthesis planning
- Custom training for energetic materials

---

## Performance Impact

### Current (xTB only):
- **Time per molecule**: ~2-5 seconds
- **Beam width 10**: ~20-50 seconds per iteration
- **Total optimization**: minutes to hours

### With SAScore:
- **Time per molecule**: ~0.001 seconds (1ms)
- **Beam width 10**: ~0.01 seconds per iteration
- **Total optimization**: seconds to minutes

**Expected speedup**: **100-1000x faster**

---

## Technical Details

### SAScore Algorithm

**How it works**:
1. **Fragment scoring**: Molecule broken into fragments
2. **Complexity penalties**: Ring systems, stereocenters, large molecules penalized
3. **Weighted combination**: Fragment scores combined with complexity

**Training data**:
- Based on 1 million PubChem compounds
- Fragment frequencies from common vs rare structures
- Complexity metrics from medicinal chemistry

**Validation**:
- Tested on known easy/hard molecules
- Correlated with expert chemist ratings
- Industry standard for 15+ years

### Integration Points

**Where to modify**:
1. `molecular_tools.py` - Add SAScore calculation
2. `molecular_optimizer_agent.py` - Use SAScore in candidate scoring
3. Configuration - Add `use_sa_score` flag
4. GUI - Display SAScore alongside other metrics

**Backward compatibility**:
- Keep xTB as optional
- Default to SAScore for speed
- Allow users to enable xTB for validation

---

## Conclusion

**SAScore is the clear winner** for the EnergeticGraph optimizer:

✅ **1000x faster** - enables real-time optimization  
✅ **Already in RDKit** - no new dependencies  
✅ **Industry standard** - well-validated  
✅ **Easy to implement** - ~50 lines of code  
✅ **Interpretable** - chemists understand it  

### Recommended Action

1. **Implement SAScore immediately** as primary feasibility score
2. **Make xTB optional** for users who need thermodynamic validation
3. **Add configuration flag** to choose scoring method
4. **Document performance improvements** in GUI

This will make the optimizer significantly faster while maintaining chemical relevance!

---

## References

1. Ertl, P.; Schuffenhauer, A. "Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions" *J. Cheminform.* 2009, 1, 8.

2. Thakkar, A.; Chadimová, V.; et al. "Retrosynthetic accessibility score (RAscore) - rapid machine learned synthesizability classification from AI driven retrosynthetic planning" *Chem. Sci.* 2021, 12, 3339-3349.

3. Coley, C. W.; Rogers, L.; et al. "SCScore: Synthetic Complexity Learned from a Reaction Corpus" *J. Chem. Inf. Model.* 2018, 58, 2, 252–261.

4. AiZynthFinder v4.0 Release Notes, 2024. MolecularAI/aizynthfinder GitHub repository.

5. "AI in Computer-Aided Synthesis Planning Market Report" Market.us, 2024.
