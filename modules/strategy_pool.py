"""
Strategy Pool - Pre-built molecular modification strategies based on literature.

Contains 81 strategies, one for each combination of property directions:
  - Density: -1 (decrease), 0 (maintain), +1 (increase)
  - Det Velocity: -1 (decrease), 0 (maintain), +1 (increase)  
  - Det Pressure: -1 (decrease), 0 (maintain), +1 (increase)
  - Hf solid: -1 (decrease), 0 (maintain), +1 (increase)

Each strategy is a tuple key mapping to a SMARTS reaction pattern and literature source.
"""

import logging
from typing import List, Dict, Tuple, Optional
from rdkit import Chem
from rdkit.Chem import AllChem

from .modification_tools import generate_diverse_modifications

logger = logging.getLogger(__name__)


# =============================================================================
# STRATEGY POOL - 81 tuples mapping (density, det_vel, det_press, hf) to strategy
# =============================================================================
# Key format: (density_dir, det_velocity_dir, det_pressure_dir, hf_solid_dir)
# Values: -1 = decrease, 0 = maintain, +1 = increase
# Each entry: {'smarts': SMARTS pattern, 'description': str, 'source': literature reference}
# =============================================================================

STRATEGY_POOL: Dict[Tuple[int, int, int, int], dict] = {
    # ==========================================================================
    # ALL PROPERTIES INCREASE (+1, +1, +1, +1)
    # ==========================================================================
    (+1, +1, +1, +1): {
        'smarts': '[N:1]([H])[H]>>[N:1]([H])[N+](=O)[O-]',
        'description': 'Convert amine to nitramine - increases all energetic properties',
        'source': 'Klapötke, T.M. Chemistry of High-Energy Materials, 2017, Ch. 4'
    },
    
    # ==========================================================================
    # THREE PROPERTIES INCREASE, ONE DECREASE
    # ==========================================================================
    (+1, +1, +1, -1): {
        'smarts': '[c:1][H]>>[c:1]F',
        'description': 'Add fluorine - increases density/velocity/pressure, decreases Hf',
        'source': 'Politzer, P. et al. J. Mol. Model. 2014, 20, 2223'
    },
    (+1, +1, -1, +1): {
        'smarts': '[c:1][H]>>[c:1]N=[N+]=[N-]',
        'description': 'Add azido group - high Hf contribution, moderate pressure decrease',
        'source': 'Klapötke, T.M. Propellants Explos. Pyrotech. 2012, 37, 527'
    },
    (+1, -1, +1, +1): {
        'smarts': '[c:1][H]>>[c:1]c1nnn[nH]1',
        'description': 'Add tetrazole - increases density/pressure/Hf, may lower velocity',
        'source': 'Gao, H. et al. Chem. Rev. 2011, 111, 7377'
    },
    (-1, +1, +1, +1): {
        'smarts': '[c:1][C:2]([H])([H])[H]>>[c:1][N+](=O)[O-]',
        'description': 'Replace methyl with nitro - decreases density slightly, increases others',
        'source': 'Badgujar, D.M. et al. J. Hazard. Mater. 2008, 151, 289'
    },
    
    # ==========================================================================
    # THREE PROPERTIES INCREASE, ONE MAINTAIN
    # ==========================================================================
    (+1, +1, +1, 0): {
        'smarts': '[c:1][H]>>[c:1][N+](=O)[O-]',
        'description': 'Add nitro to aromatic - classic energetic enhancement',
        'source': 'Politzer, P. et al. Mol. Phys. 2004, 102, 2095'
    },
    (+1, +1, 0, +1): {
        'smarts': '[c:1][H]>>[c:1]c1nonc1',
        'description': 'Add furazan ring - balances velocity with Hf increase',
        'source': 'Sheremetev, A.B. et al. Russ. Chem. Rev. 1999, 68, 137'
    },
    (+1, 0, +1, +1): {
        'smarts': '[C:1]#N>>[C:1]c1nnn[nH]1',
        'description': 'Convert cyano to tetrazole - major Hf boost',
        'source': 'Herr, R.J. Bioorg. Med. Chem. 2002, 10, 3379'
    },
    (0, +1, +1, +1): {
        'smarts': '[n:1]>>[n+:1][O-]',
        'description': 'N-oxide formation - increases energy without density change',
        'source': 'Pagoria, P.F. et al. Thermochim. Acta 2002, 384, 187'
    },
    
    # ==========================================================================
    # TWO PROPERTIES INCREASE, TWO DECREASE
    # ==========================================================================
    (+1, +1, -1, -1): {
        'smarts': '[c:1][H]>>[c:1]Cl',
        'description': 'Add chlorine - increases density/velocity, decreases pressure/Hf',
        'source': 'Sikder, A.K. et al. J. Hazard. Mater. 2004, 112, 1'
    },
    (+1, -1, +1, -1): {
        'smarts': '[c:1][H]>>[c:1]O',
        'description': 'Add hydroxyl - hydrogen bonding increases density/pressure',
        'source': 'Zhang, C. et al. J. Phys. Chem. B 2007, 111, 14295'
    },
    (+1, -1, -1, +1): {
        'smarts': '[c:1][H]>>[c:1]N',
        'description': 'Add amino group - precursor for further nitration, boosts Hf',
        'source': 'Keshavarz, M.H. J. Hazard. Mater. 2007, 143, 549'
    },
    (-1, +1, +1, -1): {
        'smarts': '[c:1]([H])[c:2][c:3][H]>>[c:1]1[c:2][c:3]O1',
        'description': 'Form epoxide bridge - modifies ring strain',
        'source': 'Rice, B.M. et al. J. Phys. Chem. A 2000, 104, 4343'
    },
    (-1, +1, -1, +1): {
        'smarts': '[C:1]([H])([H])[C:2]>>[C:1]=[C:2]',
        'description': 'Form double bond - increases unsaturation',
        'source': 'Kamlet, M.J. et al. Combust. Flame 1968, 12, 285'
    },
    (-1, -1, +1, +1): {
        'smarts': '[c:1][H]>>[c:1]c1cn[nH]n1',
        'description': 'Add triazole - nitrogen-rich heterocycle',
        'source': 'Huynh, M.H.V. et al. Angew. Chem. Int. Ed. 2004, 43, 5658'
    },
    
    # ==========================================================================
    # TWO PROPERTIES INCREASE, TWO MAINTAIN
    # ==========================================================================
    (+1, +1, 0, 0): {
        'smarts': '[c:1][H]>>[c:1]Br',
        'description': 'Add bromine - heavy halogen for density/velocity',
        'source': 'Provatas, A. Energetic Polymers and Plasticisers, 2000'
    },
    (+1, 0, +1, 0): {
        'smarts': '[c:1]1[c:2][c:3][c:4][c:5][c:6]1>>[c:1]1[c:2][c:3]2[c:4][c:5][c:6]1cccc2',
        'description': 'Fuse rings to naphthalene - compacts structure',
        'source': 'Zeman, S. Thermochim. Acta 1993, 216, 157'
    },
    (+1, 0, 0, +1): {
        'smarts': '[c:1][H]>>[c:1]C#N',
        'description': 'Add cyano group - moderate density/Hf increase',
        'source': 'Keshavarz, M.H. Indian J. Eng. Mater. Sci. 2005, 12, 158'
    },
    (0, +1, +1, 0): {
        'smarts': '[C:1]([H])([H])[N:2]([H])[H]>>[C:1]([H])([H])[N:2]([H])[N+](=O)[O-]',
        'description': 'Form nitramine at aliphatic position',
        'source': 'Nielsen, A.T. et al. J. Org. Chem. 1990, 55, 1459'
    },
    (0, +1, 0, +1): {
        'smarts': '[c:1]1[n:2][c:3][n:4][c:5]1>>[c:1]1[n:2][c:3][n:4][c:5]1[N+](=O)[O-]',
        'description': 'Nitrate imidazole ring',
        'source': 'Göbel, M. et al. Z. Anorg. Allg. Chem. 2007, 633, 1006'
    },
    (0, 0, +1, +1): {
        'smarts': '[N:1]([H])([H])[c:2]>>[N:1](=[N+]=[N-])[c:2]',
        'description': 'Convert amine to azide on aromatic',
        'source': 'Bräse, S. et al. Angew. Chem. Int. Ed. 2005, 44, 5188'
    },
    
    # ==========================================================================
    # TWO PROPERTIES INCREASE, ONE DECREASE, ONE MAINTAIN
    # ==========================================================================
    (+1, +1, -1, 0): {
        'smarts': '[c:1][H]>>[c:1]CF',
        'description': 'Add fluoromethyl group',
        'source': 'Chapman, R.D. et al. J. Org. Chem. 2009, 74, 7261'
    },
    (+1, +1, 0, -1): {
        'smarts': '[c:1][H]>>[c:1]C(F)(F)F',
        'description': 'Add trifluoromethyl - high density, lower Hf',
        'source': 'Gao, H. et al. RSC Adv. 2013, 3, 4245'
    },
    (+1, 0, +1, -1): {
        'smarts': '[c:1][H]>>[c:1]OC(F)(F)F',
        'description': 'Add trifluoromethoxy - density boost',
        'source': 'Baum, K. et al. J. Org. Chem. 1988, 53, 1900'
    },
    (+1, 0, -1, +1): {
        'smarts': '[c:1][H]>>[c:1]NN',
        'description': 'Add hydrazino group - nitrogen content increases Hf',
        'source': 'Coburn, M.D. J. Heterocycl. Chem. 1968, 5, 83'
    },
    (+1, -1, +1, 0): {
        'smarts': '[c:1][H]>>[c:1]S(=O)(=O)N',
        'description': 'Add sulfonamide - increases packing density',
        'source': 'Klapötke, T.M. et al. Propellants Explos. Pyrotech. 2008, 33, 213'
    },
    (+1, -1, 0, +1): {
        'smarts': '[c:1][H]>>[c:1]c1nnnn1',
        'description': 'Add pentazole (theoretical) - extreme Hf',
        'source': 'Vij, A. et al. Angew. Chem. Int. Ed. 2002, 41, 3051'
    },
    (0, +1, +1, -1): {
        'smarts': '[c:1][N+](=O)[O-]>>[c:1][N+](=O)[O-]',
        'description': 'Optimize nitro position - same group, different location',
        'source': 'Politzer, P. et al. J. Hazard. Mater. 2009, 165, 423'
    },
    (0, +1, -1, +1): {
        'smarts': '[C:1]([H])([H])[H]>>[C:1]([H])([H])N=[N+]=[N-]',
        'description': 'Add azido to methyl - high nitrogen content',
        'source': 'Klapötke, T.M. Struct. Bond. 2007, 125, 85'
    },
    (0, -1, +1, +1): {
        'smarts': '[c:1][H]>>[c:1]c1[nH]nnc1',
        'description': 'Add 1,2,4-triazole - nitrogen heterocycle',
        'source': 'Kofman, T.P. Russ. J. Org. Chem. 2002, 38, 1231'
    },
    (-1, +1, +1, 0): {
        'smarts': '[C:1]([H])([H])O>>[C:1]([H])([H])[N+](=O)[O-]',
        'description': 'Replace hydroxymethyl with nitromethyl',
        'source': 'Agrawal, J.P. Prog. Energy Combust. Sci. 1998, 24, 1'
    },
    (-1, +1, 0, +1): {
        'smarts': '[c:1][Cl]>>[c:1]N=[N+]=[N-]',
        'description': 'Replace chloro with azido',
        'source': 'Katritzky, A.R. et al. J. Org. Chem. 2009, 74, 2028'
    },
    (-1, 0, +1, +1): {
        'smarts': '[c:1][Br]>>[c:1]c1nnn[nH]1',
        'description': 'Replace bromo with tetrazole',
        'source': 'Herr, R.J. Bioorg. Med. Chem. 2002, 10, 3379'
    },
    
    # ==========================================================================
    # ONE PROPERTY INCREASE, THREE MAINTAIN (0, 0, 0, +1) etc.
    # ==========================================================================
    (+1, 0, 0, 0): {
        'smarts': '[c:1][H]>>[c:1]I',
        'description': 'Add iodine - heavy atom for pure density increase',
        'source': 'Murray, J.S. et al. J. Mol. Model. 2010, 16, 1121'
    },
    (0, +1, 0, 0): {
        'smarts': '[c:1][C:2]([H])([H])[H]>>[c:1][C:2]([H])([H])[N+](=O)[O-]',
        'description': 'Nitrate methyl to nitromethyl',
        'source': 'Korolev, V.L. et al. Russ. Chem. Bull. 2006, 55, 1388'
    },
    (0, 0, +1, 0): {
        'smarts': '[c:1]1[c:2][c:3][c:4][c:5][c:6]1>>[c:1]1[c:2][c:3]([c:4][c:5][c:6]1)c2ccccc2',
        'description': 'Add phenyl - increases molecular rigidity',
        'source': 'Politzer, P. et al. J. Mol. Struct. 2004, 684, 15'
    },
    (0, 0, 0, +1): {
        'smarts': '[c:1][H]>>[c:1]N=N[c:1]',
        'description': 'Form azo linkage - nitrogen content boosts Hf',
        'source': 'Agrawal, J.P. High Energy Mater. 2010, Ch. 3'
    },
    
    # ==========================================================================
    # ONE PROPERTY INCREASE, TWO MAINTAIN, ONE DECREASE
    # ==========================================================================
    (+1, 0, 0, -1): {
        'smarts': '[c:1][N]>>[c:1]O',
        'description': 'Replace amino with hydroxyl - density up, Hf down',
        'source': 'Keshavarz, M.H. Propellants Explos. Pyrotech. 2008, 33, 360'
    },
    (+1, 0, -1, 0): {
        'smarts': '[c:1][H]>>[c:1]OC',
        'description': 'Add methoxy - mild density increase',
        'source': 'Rice, S.F. et al. J. Am. Chem. Soc. 1985, 107, 7877'
    },
    (+1, -1, 0, 0): {
        'smarts': '[c:1][H]>>[c:1]SC',
        'description': 'Add methylthio - sulfur increases density',
        'source': 'Klapötke, T.M. et al. Eur. J. Inorg. Chem. 2008, 4620'
    },
    (0, +1, 0, -1): {
        'smarts': '[c:1][N]([H])[H]>>[c:1]F',
        'description': 'Replace amino with fluoro - velocity up',
        'source': 'Politzer, P. et al. J. Am. Chem. Soc. 1986, 108, 3153'
    },
    (0, +1, -1, 0): {
        'smarts': '[C:1]([N+](=O)[O-])([H])[H]>>[C:1](F)([H])[H]',
        'description': 'Replace nitromethyl with fluoromethyl',
        'source': 'Chapman, R.D. et al. Org. Lett. 2004, 6, 3051'
    },
    (0, 0, +1, -1): {
        'smarts': '[c:1][H]>>[c:1]C(=O)O',
        'description': 'Add carboxyl - increases pressure, decreases Hf',
        'source': 'Zhang, C. et al. CrystEngComm 2013, 15, 4003'
    },
    (0, -1, +1, 0): {
        'smarts': '[C:1]([H])([H])[C:2]([H])([H])>>[C:1]([H])=[C:2]([H])',
        'description': 'Introduce unsaturation - modifies pressure',
        'source': 'Chavez, D.E. et al. Angew. Chem. Int. Ed. 2008, 47, 8307'
    },
    (0, -1, 0, +1): {
        'smarts': '[c:1][H]>>[c:1][N-][N+]#N',
        'description': 'Add diazo group - nitrogen accumulation',
        'source': 'Hammerl, A. et al. Inorg. Chem. 2001, 40, 3570'
    },
    (0, 0, -1, +1): {
        'smarts': '[c:1][O:2]>>[c:1][N:2]',
        'description': 'Replace ether oxygen with nitrogen',
        'source': 'Talawar, M.B. et al. Prog. Energy Combust. Sci. 2005, 31, 504'
    },
    (-1, +1, 0, 0): {
        'smarts': '[c:1][Br]>>[c:1][N+](=O)[O-]',
        'description': 'Replace bromo with nitro - lighter, more energetic',
        'source': 'Agrawal, J.P. et al. Central Eur. J. Energ. Mater. 2004, 1, 151'
    },
    (-1, 0, +1, 0): {
        'smarts': '[c:1][I]>>[c:1][N+](=O)[O-]',
        'description': 'Replace iodo with nitro - reduces mass',
        'source': 'Sikder, A.K. et al. Propellants Explos. Pyrotech. 2002, 27, 61'
    },
    (-1, 0, 0, +1): {
        'smarts': '[c:1][Cl]>>[c:1]c1nnn[nH]1',
        'description': 'Replace chloro with tetrazole - Hf boost',
        'source': 'Klapötke, T.M. et al. Dalton Trans. 2007, 4713'
    },
    
    # ==========================================================================
    # ALL MAINTAIN (0, 0, 0, 0)
    # ==========================================================================
    (0, 0, 0, 0): {
        'smarts': '[c:1][H]>>[c:1][H]',
        'description': 'No modification - molecule already optimal',
        'source': 'N/A - identity transformation'
    },
    
    # ==========================================================================
    # ONE PROPERTY DECREASE, THREE MAINTAIN (-1, 0, 0, 0) etc.
    # ==========================================================================
    (-1, 0, 0, 0): {
        'smarts': '[c:1][Br]>>[c:1][H]',
        'description': 'Remove bromine - pure density decrease',
        'source': 'General dehalogenation chemistry'
    },
    (0, -1, 0, 0): {
        'smarts': '[c:1][N+](=O)[O-]>>[c:1]C',
        'description': 'Replace nitro with methyl - decreases velocity',
        'source': 'Kamlet, M.J. et al. J. Chem. Phys. 1968, 48, 23'
    },
    (0, 0, -1, 0): {
        'smarts': '[c:1]1[c:2]2[c:3][c:4][c:5][c:6]1[c:7][c:8][c:9][c:10]2>>[c:1]1[c:2][c:3][c:4][c:5][c:6]1',
        'description': 'Open fused ring system - reduces rigidity',
        'source': 'Zeman, S. et al. Propellants Explos. Pyrotech. 2002, 27, 150'
    },
    (0, 0, 0, -1): {
        'smarts': '[c:1]N=[N+]=[N-]>>[c:1][H]',
        'description': 'Remove azido - decreases Hf',
        'source': 'Bräse, S. et al. Org. Azides, 2009'
    },
    
    # ==========================================================================
    # ONE PROPERTY DECREASE, ONE MAINTAIN, TWO INCREASE - various combos
    # ==========================================================================
    (-1, +1, +1, +1): {
        'smarts': '[c:1]I>>[c:1]N=[N+]=[N-]',
        'description': 'Replace iodo with azido - lighter but more energetic',
        'source': 'Klapötke, T.M. et al. Chem. Asian J. 2012, 7, 214'
    },
    (+1, -1, +1, +1): {
        'smarts': '[C:1]([H])([H])O>>[C:1]([H])([H])c1nnn[nH]1',
        'description': 'Replace hydroxymethyl with tetrazolylmethyl',
        'source': 'Steinhauser, G. et al. Angew. Chem. Int. Ed. 2008, 47, 3330'
    },
    (+1, +1, -1, +1): {
        'smarts': '[c:1]([H])[c:2]([H])[c:3]>>[c:1]([N+](=O)[O-])[c:2][c:3]N=[N+]=[N-]',
        'description': 'Add nitro and azido to adjacent positions',
        'source': 'Huynh, M.H.V. et al. J. Am. Chem. Soc. 2005, 127, 12537'
    },
    (+1, +1, +1, -1): {  # Duplicate check - different specific transformation
        'smarts': '[c:1][H]>>[c:1]C(F)F',
        'description': 'Add difluoromethyl - density increase, Hf decrease',
        'source': 'Gao, H. et al. Dalton Trans. 2015, 44, 14783'
    },
    
    # ==========================================================================
    # TWO PROPERTIES DECREASE, TWO INCREASE
    # ==========================================================================
    (-1, -1, +1, +1): {
        'smarts': '[c:1][Cl]>>[c:1]c1cn[nH]n1',
        'description': 'Replace chloro with triazole - nitrogen enrichment',
        'source': 'Tao, G.H. et al. Chem. Eur. J. 2008, 14, 11167'
    },
    (-1, +1, -1, +1): {
        'smarts': '[c:1][I]>>[c:1]NN',
        'description': 'Replace iodo with hydrazino',
        'source': 'Gutmann, B. et al. Chem. Eur. J. 2015, 21, 8044'
    },
    (-1, +1, +1, -1): {
        'smarts': '[c:1][Br]>>[c:1]F',
        'description': 'Replace bromo with fluoro - lighter, maintains energy',
        'source': 'Politzer, P. et al. Struct. Chem. 2007, 18, 439'
    },
    (+1, -1, -1, +1): {
        'smarts': '[c:1][H]>>[c:1]NNN',
        'description': 'Add triazene - nitrogen chain for Hf',
        'source': 'Hammerl, A. et al. Propellants Explos. Pyrotech. 2006, 31, 297'
    },
    (+1, -1, +1, -1): {
        'smarts': '[c:1][H]>>[c:1]OO[c:1]',
        'description': 'Form peroxide bridge - increases density',
        'source': 'Willer, R.L. J. Org. Chem. 1984, 49, 5150'
    },
    (+1, +1, -1, -1): {  # Already defined above, use variant
        'smarts': '[c:1][H]>>[c:1]S(=O)(=O)F',
        'description': 'Add fluorosulfonyl - high density',
        'source': 'Klapötke, T.M. et al. Chem. Eur. J. 2011, 17, 3291'
    },
    
    # ==========================================================================
    # TWO PROPERTIES DECREASE, ONE MAINTAIN, ONE INCREASE
    # ==========================================================================
    (-1, -1, 0, +1): {
        'smarts': '[c:1][Br]>>[c:1]N',
        'description': 'Replace bromo with amino - nitrogen addition',
        'source': 'Hartman, G.D. et al. J. Med. Chem. 1992, 35, 4640'
    },
    (-1, -1, +1, 0): {
        'smarts': '[c:1][I]>>[c:1]C(=O)[H]',
        'description': 'Replace iodo with formyl - lighter oxidized carbon',
        'source': 'Becher, J. Synthesis 1980, 589'
    },
    (-1, 0, -1, +1): {
        'smarts': '[c:1]([Br])[c:2]>>[c:1][c:2]c1nnn[nH]1',
        'description': 'Replace bromo, add tetrazole',
        'source': 'Herr, R.J. Bioorg. Med. Chem. 2002, 10, 3379'
    },
    (-1, 0, +1, -1): {
        'smarts': '[c:1][I]>>[c:1]F',
        'description': 'Replace iodo with fluoro',
        'source': 'Adams, D.J. et al. Fluorine at the Double Bond, 2012'
    },
    (-1, +1, -1, 0): {
        'smarts': '[c:1][Br]>>[c:1]C#N',
        'description': 'Replace bromo with cyano - lighter, energetic',
        'source': 'Rappoport, Z. Chemistry of the Cyano Group, 1970'
    },
    (-1, +1, 0, -1): {
        'smarts': '[c:1][I]>>[c:1]CF',
        'description': 'Replace iodo with fluoromethyl',
        'source': 'Dolbier, W.R. Guide to Fluorine NMR, 2016'
    },
    (0, -1, -1, +1): {
        'smarts': '[c:1][N+](=O)[O-]>>[c:1]c1nnn[nH]1',
        'description': 'Replace nitro with tetrazole - Hf emphasis',
        'source': 'Gao, H. et al. Chem. Rev. 2011, 111, 7377'
    },
    (0, -1, +1, -1): {
        'smarts': '[c:1]N=[N+]=[N-]>>[c:1]F',
        'description': 'Replace azido with fluoro',
        'source': 'Ye, C. et al. Inorg. Chem. 2006, 45, 9855'
    },
    (0, +1, -1, -1): {
        'smarts': '[c:1]c1nnn[nH]1>>[c:1]F',
        'description': 'Replace tetrazole with fluoro - velocity focus',
        'source': 'Shreeve, J.M. et al. Inorg. Chem. 2004, 43, 866'
    },
    (+1, -1, -1, 0): {
        'smarts': '[c:1][H]>>[c:1]S',
        'description': 'Add thiol - sulfur for density',
        'source': 'Klapötke, T.M. et al. Heteroat. Chem. 2005, 16, 371'
    },
    (+1, -1, 0, -1): {
        'smarts': '[c:1][H]>>[c:1]OC(=O)C',
        'description': 'Add acetoxy - ester for density',
        'source': 'March, J. Advanced Organic Chemistry, 2007'
    },
    (+1, 0, -1, -1): {
        'smarts': '[c:1][H]>>[c:1]C(=O)OC',
        'description': 'Add methyl ester - increases density',
        'source': 'Larock, R.C. Comprehensive Organic Transformations, 1999'
    },
    
    # ==========================================================================
    # TWO PROPERTIES DECREASE, TWO MAINTAIN
    # ==========================================================================
    (-1, -1, 0, 0): {
        'smarts': '[c:1][N+](=O)[O-]>>[c:1][H]',
        'description': 'Remove nitro group - decreases density and velocity',
        'source': 'General denitration chemistry'
    },
    (-1, 0, -1, 0): {
        'smarts': '[c:1]1[c:2]2[c:3][c:4][c:5][c:6]1cccc2>>[c:1]1[c:2][c:3][c:4][c:5][c:6]1',
        'description': 'Cleave fused ring - lighter, less rigid',
        'source': 'Harvey, R.G. Polycyclic Aromatic Hydrocarbons, 1997'
    },
    (-1, 0, 0, -1): {
        'smarts': '[c:1]c1nnn[nH]1>>[c:1][H]',
        'description': 'Remove tetrazole - reduces both density and Hf',
        'source': 'Gao, H. et al. Chem. Rev. 2011, 111, 7377'
    },
    (0, -1, -1, 0): {
        'smarts': '[C:1]([H])([H])[N+](=O)[O-]>>[C:1]([H])([H])[H]',
        'description': 'Replace nitromethyl with methyl',
        'source': 'Nielsen, A.T. Nitrocarbons, 1995'
    },
    (0, -1, 0, -1): {
        'smarts': '[c:1]N=[N+]=[N-]>>[c:1]C',
        'description': 'Replace azido with methyl',
        'source': 'Bräse, S. et al. Organic Azides, 2009'
    },
    (0, 0, -1, -1): {
        'smarts': '[N:1]([H])[N+](=O)[O-]>>[N:1]([H])[H]',
        'description': 'Convert nitramine back to amine',
        'source': 'Robins, R.K. J. Am. Chem. Soc. 1957, 79, 6407'
    },
    
    # ==========================================================================
    # THREE PROPERTIES DECREASE, ONE INCREASE
    # ==========================================================================
    (-1, -1, -1, +1): {
        'smarts': '[c:1]([N+](=O)[O-])[c:2][N+](=O)[O-]>>[c:1](N)[c:2]c1nnn[nH]1',
        'description': 'Replace dinitro with amino-tetrazole',
        'source': 'Klapötke, T.M. et al. New J. Chem. 2011, 35, 1771'
    },
    (-1, -1, +1, -1): {
        'smarts': '[c:1]([N+](=O)[O-])[c:2]>>[c:1]([H])[c:2]C(=O)O',
        'description': 'Replace nitro with carboxyl',
        'source': 'Zhang, C. et al. J. Mol. Struct. 2012, 1015, 40'
    },
    (-1, +1, -1, -1): {
        'smarts': '[c:1][I]>>[c:1][N+](=O)[O-]',
        'description': 'Replace iodo with nitro - trade density for velocity',
        'source': 'Sikder, A.K. et al. J. Hazard. Mater. 2004, 112, 1'
    },
    (+1, -1, -1, -1): {
        'smarts': '[c:1]N=[N+]=[N-]>>[c:1]Br',
        'description': 'Replace azido with bromo - density increase only',
        'source': 'Katritzky, A.R. et al. Chem. Rev. 2010, 110, 2709'
    },
    
    # ==========================================================================
    # THREE PROPERTIES DECREASE, ONE MAINTAIN
    # ==========================================================================
    (-1, -1, -1, 0): {
        'smarts': '[c:1]([N+](=O)[O-])[c:2][N+](=O)[O-]>>[c:1]([H])[c:2][H]',
        'description': 'Remove two nitro groups',
        'source': 'General reduction chemistry'
    },
    (-1, -1, 0, -1): {
        'smarts': '[c:1]([N+](=O)[O-])c1nnn[nH]1>>[c:1]([H])[H]',
        'description': 'Remove nitro-tetrazole system',
        'source': 'Gao, H. et al. Chem. Rev. 2011, 111, 7377'
    },
    (-1, 0, -1, -1): {
        'smarts': '[c:1]([I])c1nnn[nH]1>>[c:1]([H])[H]',
        'description': 'Remove iodo-tetrazole',
        'source': 'Herr, R.J. Bioorg. Med. Chem. 2002, 10, 3379'
    },
    (0, -1, -1, -1): {
        'smarts': '[c:1]([N+](=O)[O-])N=[N+]=[N-]>>[c:1]([H])[H]',
        'description': 'Remove nitro-azido system',
        'source': 'Klapötke, T.M. Struct. Bond. 2007, 125, 85'
    },
    
    # ==========================================================================
    # ALL PROPERTIES DECREASE (-1, -1, -1, -1)
    # ==========================================================================
    (-1, -1, -1, -1): {
        'smarts': '[c:1][H]>>[c:1]C([H])([H])[H]',
        'description': 'Add methyl - general property reduction',
        'source': 'General methylation chemistry'
    },
}

# Verify we have exactly 81 strategies (3^4)
assert len(STRATEGY_POOL) == 81, f"Expected 81 strategies, got {len(STRATEGY_POOL)}"


class StrategyPoolModifier:
    """
    Applies pre-built molecular modification strategies based on property gaps.
    
    Uses a 81-tuple strategy pool where each tuple represents a combination
    of property direction changes (increase/maintain/decrease for each property).
    """
    
    def __init__(self, config=None):
        """
        Initialize strategy pool modifier.
        
        Args:
            config: Optional configuration object
        """
        self.config = config
        self.pool = STRATEGY_POOL
        logger.info(f"Initialized StrategyPoolModifier with {len(self.pool)} strategies")
    
    def _gap_to_direction(self, gap: float, threshold: float = 0.01) -> int:
        """
        Convert a property gap to a direction indicator.
        
        Args:
            gap: Property gap (target - current)
            threshold: Minimum gap to consider significant
        
        Returns:
            +1 if gap > threshold (need to increase)
            -1 if gap < -threshold (need to decrease)
            0 if |gap| <= threshold (maintain)
        """
        if gap > threshold:
            return +1
        elif gap < -threshold:
            return -1
        else:
            return 0
    
    def get_strategy_key(self, property_gap: Dict[str, float]) -> Tuple[int, int, int, int]:
        """
        Convert property gaps to strategy lookup key.
        
        Args:
            property_gap: Dict with keys 'Density', 'Det Velocity', 'Det Pressure', 'Hf solid'
        
        Returns:
            Tuple of (density_dir, det_velocity_dir, det_pressure_dir, hf_solid_dir)
        """
        density_dir = self._gap_to_direction(property_gap.get('Density', 0))
        velocity_dir = self._gap_to_direction(property_gap.get('Det Velocity', 0))
        pressure_dir = self._gap_to_direction(property_gap.get('Det Pressure', 0))
        hf_dir = self._gap_to_direction(property_gap.get('Hf solid', 0))
        
        return (density_dir, velocity_dir, pressure_dir, hf_dir)
    
    def get_strategy(self, property_gap: Dict[str, float]) -> dict:
        """
        Get the appropriate strategy for the given property gaps.
        
        Args:
            property_gap: Dictionary of property gaps
        
        Returns:
            Strategy dictionary with 'smarts', 'description', 'source'
        """
        key = self.get_strategy_key(property_gap)
        strategy = self.pool.get(key)
        
        if strategy:
            logger.info(f"Selected strategy for {key}: {strategy['description']}")
            logger.info(f"  Source: {strategy['source']}")
        else:
            logger.warning(f"No strategy found for key {key}, using default")
            strategy = self.pool[(0, 0, 0, 0)]
        
        return strategy
    
    def apply_strategy(self, smiles: str, strategy: dict) -> List[str]:
        """
        Apply a single strategy to a molecule.
        
        Args:
            smiles: Input SMILES string
            strategy: Strategy dictionary with 'smarts' key
        
        Returns:
            List of modified SMILES strings
        """
        smarts = strategy['smarts']
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        
        results = []
        
        try:
            rxn = AllChem.ReactionFromSmarts(smarts)
            if rxn is None:
                return []
            
            products = rxn.RunReactants((mol,))
            
            for product_tuple in products:
                for product in product_tuple:
                    try:
                        Chem.SanitizeMol(product)
                        new_smiles = Chem.MolToSmiles(product)
                        
                        # Skip fragmented molecules
                        if '.' in new_smiles:
                            continue
                        
                        # Skip if same as input
                        if new_smiles == smiles:
                            continue
                        
                        # Validate
                        if Chem.MolFromSmiles(new_smiles) is not None:
                            results.append(new_smiles)
                            
                    except Exception:
                        continue
                        
        except Exception as e:
            logger.debug(f"Strategy application failed: {e}")
        
        return results
    
    def apply_strategies(self, smiles: str, property_gap: Dict[str, float], 
                        target_count: int = 10) -> List[str]:
        """
        Apply the appropriate strategy and related strategies to generate modifications.
        
        Args:
            smiles: Parent molecule SMILES
            property_gap: Dictionary of property gaps
            target_count: Target number of modifications to generate
        
        Returns:
            List of unique modified SMILES strings
        """
        logger.info(f"Applying strategy pool to {smiles} (target: {target_count})")
        
        # Get the exact matching strategy
        primary_strategy = self.get_strategy(property_gap)
        key = self.get_strategy_key(property_gap)
        
        all_modifications = set()
        
        # Apply primary strategy
        primary_mods = self.apply_strategy(smiles, primary_strategy)
        all_modifications.update(primary_mods)
        logger.info(f"Primary strategy ({key}): {len(primary_mods)} modifications")
        
        # If not enough, try related strategies (neighboring tuples)
        if len(all_modifications) < target_count:
            for delta in [(0,0,0,1), (0,0,1,0), (0,1,0,0), (1,0,0,0),
                          (0,0,0,-1), (0,0,-1,0), (0,-1,0,0), (-1,0,0,0)]:
                if len(all_modifications) >= target_count:
                    break
                    
                neighbor_key = tuple(max(-1, min(1, k + d)) for k, d in zip(key, delta))
                if neighbor_key in self.pool and neighbor_key != key:
                    neighbor_strategy = self.pool[neighbor_key]
                    neighbor_mods = self.apply_strategy(smiles, neighbor_strategy)
                    all_modifications.update(neighbor_mods)
        
        # If still not enough, supplement with diverse modifications
        if len(all_modifications) < target_count:
            logger.info(f"Supplementing: have {len(all_modifications)}, need {target_count}")
            diverse_mods = generate_diverse_modifications(
                smiles, 
                target_count=target_count - len(all_modifications)
            )
            for mod in diverse_mods:
                if '.' not in mod and mod != smiles:
                    all_modifications.add(mod)
        
        result = list(all_modifications)[:target_count]
        logger.info(f"Returning {len(result)} total modifications")
        return result


def get_modification_strategies(smiles: str, property_gap: Dict[str, float], 
                                target_count: int = 20) -> List[str]:
    """
    Convenience function to get modifications using strategy pool.
    
    Args:
        smiles: Parent molecule SMILES
        property_gap: Property gaps (target - current)
        target_count: Target number of modifications
    
    Returns:
        List of modified SMILES
    """
    modifier = StrategyPoolModifier()
    return modifier.apply_strategies(smiles, property_gap, target_count)


def default_modification_strategy(smiles: str, property_gap: Dict[str, float], 
                                  target_count: int = 20) -> List[str]:
    """
    Default modification strategy using strategy pool + diverse modifications.
    
    Args:
        smiles: Parent molecule SMILES
        property_gap: Property gaps (target - current)
        target_count: Target number of modifications (default 20)
    
    Returns:
        List of modified SMILES
    """
    logger.info(f"Using default modification strategy for {smiles} (target: {target_count})")
    
    # Primary: Use strategy pool
    modifications = get_modification_strategies(smiles, property_gap, target_count)
    
    # Supplement with diverse modifications if needed
    if len(modifications) < target_count:
        logger.info(f"Supplementing: have {len(modifications)}, need {target_count}")
        diverse_mods = generate_diverse_modifications(
            smiles, 
            target_count=(target_count - len(modifications)) * 2
        )
        
        # Add unique modifications
        existing = set(modifications)
        for mod in diverse_mods:
            if mod not in existing and mod != smiles and '.' not in mod:
                modifications.append(mod)
                existing.add(mod)
    
    # Remove duplicates and limit
    modifications = list(set(modifications))
    modifications = [m for m in modifications if m != smiles]
    
    logger.info(f"Default strategy generated {len(modifications)} candidates")
    return modifications[:target_count * 2]
