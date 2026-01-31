"""
Core data structures for the beam search molecular design system.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class MoleculeState:
    """
    Represents a single molecule in the beam search tree.
    
    This is the fundamental node object passed between agents and modules.
    """
    smiles: str  # SMILES representation
    properties: Dict[str, float] = field(default_factory=dict)  # Predicted properties
    score: float = float('inf')  # Total score (lower is better)
    feasibility: float = 1.0  # Normalized SAScore (0-1, 0 = most feasible, 1 = least feasible)
    provenance: str = ""  # History of modifications
    is_feasible: bool = False  # Whether molecule passes feasibility checks (SAScore <= 7)
    generation: int = 0  # Which iteration this molecule was created in
    parent_smiles: Optional[str] = None  # Parent molecule SMILES
    
    def __repr__(self) -> str:
        return (f"MoleculeState(smiles='{self.smiles}', "
                f"score={self.score:.4f}, "
                f"feasibility={self.feasibility:.2f}, "
                f"generation={self.generation})")
    
    def __lt__(self, other: 'MoleculeState') -> bool:
        """Enable sorting by score (lower is better)."""
        return self.score < other.score
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'smiles': self.smiles,
            'properties': self.properties,
            'score': self.score,
            'feasibility': self.feasibility,
            'provenance': self.provenance,
            'is_feasible': self.is_feasible,
            'generation': self.generation,
            'parent_smiles': self.parent_smiles
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'MoleculeState':
        """Create MoleculeState from dictionary."""
        return cls(**data)


@dataclass
class PropertyTarget:
    """Target properties for molecular design."""
    density: float  # g/cm³
    det_velocity: float  # m/s
    det_pressure: float  # GPa
    hf_solid: float  # kJ/mol
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'Density': self.density,
            'Det Velocity': self.det_velocity,
            'Det Pressure': self.det_pressure,
            'Hf solid': self.hf_solid
        }
    
    def __repr__(self) -> str:
        return (f"PropertyTarget(Density={self.density:.2f}, "
                f"Det_Velocity={self.det_velocity:.0f}, "
                f"Det_Pressure={self.det_pressure:.1f}, "
                f"Hf_solid={self.hf_solid:.1f})")
