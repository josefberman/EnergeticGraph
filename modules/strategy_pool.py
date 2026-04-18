"""
Strategy Pool — 81 direction-tuple strategies for energetic molecule design.

Design (see plan doc §7):

1. A curated library of chemically-valid SMARTS "primitives", each tagged
   with an empirical direction-vector Δ = (Δρ, ΔD, ΔP, ΔHf) ∈ {-1, 0, +1}^4.
   Δ-vectors come from Kamlet–Jacobs detonation correlations (Kamlet &
   Jacobs, J. Chem. Phys. 48, 23 (1968); Kamlet & Hurwitz, ibid. 1969),
   Politzer & Murray electrostatic-potential analyses (J. Mol. Model. 2014,
   20, 2223), and the group-contribution summaries in Klapötke,
   *Chemistry of High-Energy Materials* (3rd ed., 2017) and Agrawal,
   *High Energy Materials* (Wiley-VCH, 2010).

2. The 81 direction tuples (3^4) are mapped to primitives by minimising
   the L1 distance ||Δ_primitive − target_tuple||₁. Ties are broken in
   favour of (a) classic energetic chemistry (aromatic nitration,
   nitramine formation, azidation, tetrazole installation), then
   (b) fewest bond edits.

3. Every primitive SMARTS is parsed by ``rdkit.Chem.AllChem.ReactionFromSmarts``
   at import time; a failure raises ImportError so broken patterns are
   caught at startup rather than in the middle of a beam search.

The identity tuple (0, 0, 0, 0) is handled specially — there is no
meaningful "do-nothing" SMARTS reaction, so it returns the parent SMILES
unchanged when applied.
"""

import logging
from typing import Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem

from .modification_tools import generate_diverse_modifications

logger = logging.getLogger(__name__)


# =============================================================================
# Primitive library. Each entry is a tuple of:
#   (key, smarts_or_None, delta, description, source)
# `smarts_or_None` of None is reserved for the identity primitive.
# =============================================================================

_KJ = ("Kamlet & Jacobs, J. Chem. Phys. 48, 23 (1968); "
       "Klapötke, Chemistry of High-Energy Materials, 3rd ed., 2017")
_POLITZER = "Politzer & Murray, J. Mol. Model. 2014, 20, 2223"
_KLAPOTKE = "Klapötke, Chemistry of High-Energy Materials, 3rd ed., 2017"
_AGRAWAL = "Agrawal, High Energy Materials, Wiley-VCH, 2010"
_GAO = "Gao & Shreeve, Chem. Rev. 2011, 111, 7377"
_GROUP_ADD = "Group-additivity extrapolation (Klapötke 2017; Agrawal 2010)"

# (name, smarts, (dρ, dD, dP, dHf), description, source)
_PRIMITIVES: List[Tuple[str, Optional[str], Tuple[int, int, int, int], str, str]] = [
    # --- Identity -----------------------------------------------------------
    ("identity", None, (0, 0, 0, 0),
     "No modification (parent SMILES returned unchanged).",
     "N/A"),

    # --- Nitration family (classic energetic enhancement) -------------------
    ("nitrate_arom_CH",
     "[cH:1]>>[c:1][N+](=O)[O-]",
     (+1, +1, +1, 0),
     "Aromatic C–H → C–NO2. Canonical energetic substitution.",
     _KJ),
    ("nitrate_alkyl_CH",
     "[CX4;!H0:1]>>[C:1][N+](=O)[O-]",
     (+1, +1, +1, 0),
     "Aliphatic C–H → C–NO2.",
     _KLAPOTKE),
    ("nitramine_NH",
     "[NX3;H1,H2:1]>>[N:1][N+](=O)[O-]",
     (+1, +1, +1, +1),
     "Amine N–H → nitramine N–NO2 (RDX/HMX family).",
     _KLAPOTKE),

    # --- Nitrogen-rich heterocycles (Hf-dominant) ---------------------------
    ("azide_arom",
     "[cH:1]>>[c:1]N=[N+]=[N-]",
     (0, 0, -1, +1),
     "Aromatic C–H → C–N3; nitrogen-rich, high Hf.",
     _KLAPOTKE),
    ("azide_methyl",
     "[CH3:1][c:2]>>[CH2:1]([c:2])N=[N+]=[N-]",
     (0, +1, -1, +1),
     "Ar–CH3 → Ar–CH2–N3 (azidomethyl).",
     _GAO),
    ("tetrazole_CH",
     "[cH:1]>>[c:1]c1nnn[nH]1",
     (+1, 0, +1, +1),
     "Aromatic C–H → C–(1H-tetrazol-5-yl); N-rich, high Hf and density.",
     _GAO),
    ("triazole_CH",
     "[cH:1]>>[c:1]c1cn[nH]n1",
     (0, 0, +1, +1),
     "Aromatic C–H → C–(1,2,4-triazolyl).",
     _GAO),
    ("n_oxide",
     "[nX2:1]>>[n+:1][O-]",
     (0, +1, +1, +1),
     "Aromatic N → N-oxide (Pagoria et al.).",
     "Pagoria et al., Thermochim. Acta 384, 187 (2002)"),

    # --- Polar EWGs ---------------------------------------------------------
    ("cyano_CH",
     "[cH:1]>>[c:1]C#N",
     (+1, 0, +1, +1),
     "Aromatic C–H → C–CN.",
     _GAO),
    ("hydrazino_CH",
     "[cH:1]>>[c:1]NN",
     (+1, 0, -1, +1),
     "Aromatic C–H → C–NHNH2 (hydrazino).",
     "Coburn, J. Heterocycl. Chem. 5, 83 (1968)"),
    ("amino_CH",
     "[cH:1]>>[c:1]N",
     (0, -1, -1, +1),
     "Aromatic C–H → C–NH2 (weak donor; aminotriazine-style).",
     _AGRAWAL),

    # --- Halogens (density up, Hf down) -------------------------------------
    ("fluoro_arom_CH",
     "[cH:1]>>[c:1]F",
     (+1, +1, +1, -1),
     "Aromatic C–H → C–F.",
     _POLITZER),
    ("trifluoromethyl_CH",
     "[cH:1]>>[c:1]C(F)(F)F",
     (+1, +1, 0, -1),
     "Aromatic C–H → C–CF3.",
     "Gao et al., RSC Adv. 3, 4245 (2013)"),
    ("chloro_arom_CH",
     "[cH:1]>>[c:1]Cl",
     (+1, +1, -1, -1),
     "Aromatic C–H → C–Cl (chain inhibitor).",
     "Sikder et al., J. Hazard. Mater. 112, 1 (2004)"),
    ("bromo_arom_CH",
     "[cH:1]>>[c:1]Br",
     (+1, +1, 0, 0),
     "Aromatic C–H → C–Br (heavy halogen).",
     _GROUP_ADD),
    ("iodo_arom_CH",
     "[cH:1]>>[c:1]I",
     (+1, 0, 0, 0),
     "Aromatic C–H → C–I (pure density gain).",
     _POLITZER),

    # --- Oxygen functional groups -------------------------------------------
    ("hydroxyl_CH",
     "[cH:1]>>[c:1]O",
     (+1, -1, +1, -1),
     "Aromatic C–H → C–OH (H-bond-driven density increase).",
     "Zhang et al., J. Phys. Chem. B 111, 14295 (2007)"),
    ("methoxy_CH",
     "[cH:1]>>[c:1]OC",
     (+1, 0, -1, 0),
     "Aromatic C–H → C–OCH3 (methoxy).",
     _GROUP_ADD),
    ("carboxyl_CH",
     "[cH:1]>>[c:1]C(=O)O",
     (0, 0, +1, -1),
     "Aromatic C–H → C–COOH.",
     "Zhang et al., CrystEngComm 15, 4003 (2013)"),
    ("methylester_CH",
     "[cH:1]>>[c:1]C(=O)OC",
     (+1, 0, -1, -1),
     "Aromatic C–H → C–COOCH3.",
     _GROUP_ADD),

    # --- Carbon skeleton changes --------------------------------------------
    ("methyl_to_nitromethyl",
     "[c:1][CH3]>>[c:1][CH2][N+](=O)[O-]",
     (0, +1, +1, 0),
     "Ar–CH3 → Ar–CH2–NO2.",
     "Korolev et al., Russ. Chem. Bull. 55, 1388 (2006)"),
    ("dehydrogenate_CC",
     "[CX4;!H0:1][CX4;!H0:2]>>[C:1]=[C:2]",
     (0, +1, -1, +1),
     "C(sp3)–C(sp3) → C=C (introduce unsaturation).",
     "Chavez et al., Angew. Chem. Int. Ed. 47, 8307 (2008)"),
    ("add_phenyl_CH",
     "[cH:1]>>[c:1]c1ccccc1",
     (0, 0, +1, 0),
     "Aromatic C–H → C–Ph (ring stacking rigidity).",
     "Politzer et al., J. Mol. Struct. 684, 15 (2004)"),

    # --- Conversions / swaps ------------------------------------------------
    ("amino_to_nitro",
     "[c:1][NH2]>>[c:1][N+](=O)[O-]",
     (0, +1, +1, -1),
     "Ar–NH2 → Ar–NO2 (oxidation).",
     _KLAPOTKE),
    ("amino_to_azide",
     "[c:1][NH2]>>[c:1]N=[N+]=[N-]",
     (-1, 0, 0, +1),
     "Ar–NH2 → Ar–N3 (diazotization).",
     "Bräse et al., Angew. Chem. Int. Ed. 44, 5188 (2005)"),
    ("halogen_Br_to_nitro",
     "[c:1]Br>>[c:1][N+](=O)[O-]",
     (-1, +1, +1, +1),
     "Ar–Br → Ar–NO2 (lighter, more energetic).",
     _KLAPOTKE),
    ("halogen_I_to_azide",
     "[c:1]I>>[c:1]N=[N+]=[N-]",
     (-1, +1, +1, +1),
     "Ar–I → Ar–N3.",
     "Katritzky et al., Chem. Rev. 110, 2709 (2010)"),

    # --- Removals (negative deltas) -----------------------------------------
    ("remove_nitro_arom",
     "[c:1][N+](=O)[O-]>>[c:1][H]",
     (-1, -1, -1, 0),
     "Remove aromatic nitro group.",
     _GROUP_ADD),
    ("remove_bromo_arom",
     "[c:1]Br>>[c:1][H]",
     (-1, 0, 0, 0),
     "Remove aromatic Br.",
     _GROUP_ADD),
    ("remove_azide_arom",
     "[c:1]N=[N+]=[N-]>>[c:1][H]",
     (0, 0, 0, -1),
     "Remove aromatic azide.",
     _GROUP_ADD),
    ("nitro_to_methyl_arom",
     "[c:1][N+](=O)[O-]>>[c:1]C",
     (-1, -1, -1, -1),
     "Ar–NO2 → Ar–CH3 (downgrade to alkyl).",
     _GROUP_ADD),
    ("nitramine_to_amine",
     "[N:1]([N+](=O)[O-])>>[N:1][H]",
     (0, 0, -1, -1),
     "Nitramine N–NO2 → amine N–H.",
     _GROUP_ADD),
    ("methyl_to_H",
     "[cX3:1][CH3]>>[cX3:1][H]",
     (-1, 0, -1, +1),
     "Ar–CH3 → Ar–H.",
     _GROUP_ADD),
]


# ---------------------------------------------------------------------------
# Validate primitive SMARTS at import time
# ---------------------------------------------------------------------------

def _validate_primitives() -> None:
    for name, smarts, delta, desc, _src in _PRIMITIVES:
        if smarts is None:
            continue  # identity
        rxn = AllChem.ReactionFromSmarts(smarts)
        if rxn is None:
            raise ImportError(
                f"strategy_pool: primitive '{name}' has invalid SMARTS "
                f"({smarts!r})"
            )
        if rxn.GetNumReactantTemplates() != 1 or rxn.GetNumProductTemplates() != 1:
            raise ImportError(
                f"strategy_pool: primitive '{name}' must have exactly one "
                f"reactant and one product template; got "
                f"{rxn.GetNumReactantTemplates()} / {rxn.GetNumProductTemplates()}"
            )
        for d in delta:
            if d not in (-1, 0, +1):
                raise ImportError(
                    f"strategy_pool: primitive '{name}' delta {delta} is "
                    f"outside {{-1, 0, +1}}^4"
                )


_validate_primitives()


# ---------------------------------------------------------------------------
# Map the 81 direction tuples → primitive by L1-nearest Δ.
# ---------------------------------------------------------------------------

# Priority rank for tie-breaking. Lower number == higher preference.
# Canonical energetic chemistry wins over obscure swaps.
_PRIORITY = {
    "nitramine_NH": 0,
    "nitrate_arom_CH": 1,
    "nitrate_alkyl_CH": 2,
    "tetrazole_CH": 3,
    "triazole_CH": 4,
    "azide_arom": 5,
    "azide_methyl": 6,
    "n_oxide": 7,
    "amino_to_nitro": 8,
    "halogen_Br_to_nitro": 9,
    "halogen_I_to_azide": 10,
    "cyano_CH": 11,
    "hydrazino_CH": 12,
    "methyl_to_nitromethyl": 13,
    "amino_to_azide": 14,
    "fluoro_arom_CH": 15,
    "trifluoromethyl_CH": 16,
    "chloro_arom_CH": 17,
    "bromo_arom_CH": 18,
    "iodo_arom_CH": 19,
    "hydroxyl_CH": 20,
    "methoxy_CH": 21,
    "carboxyl_CH": 22,
    "methylester_CH": 23,
    "amino_CH": 24,
    "dehydrogenate_CC": 25,
    "add_phenyl_CH": 26,
    "methyl_to_H": 27,
    "remove_nitro_arom": 28,
    "remove_bromo_arom": 29,
    "remove_azide_arom": 30,
    "nitro_to_methyl_arom": 31,
    "nitramine_to_amine": 32,
    "identity": 99,
}


def _l1(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> int:
    return sum(abs(x - y) for x, y in zip(a, b))


def _build_strategy_pool() -> Dict[Tuple[int, int, int, int], dict]:
    pool: Dict[Tuple[int, int, int, int], dict] = {}
    for d1 in (-1, 0, +1):
        for d2 in (-1, 0, +1):
            for d3 in (-1, 0, +1):
                for d4 in (-1, 0, +1):
                    target = (d1, d2, d3, d4)
                    if target == (0, 0, 0, 0):
                        # Force identity for the true no-op tuple.
                        name = "identity"
                        smarts, delta, desc, src = None, (0, 0, 0, 0), \
                            "No modification required (all properties on target).", "N/A"
                    else:
                        best = None
                        for prim_name, prim_smarts, prim_delta, prim_desc, prim_src in _PRIMITIVES:
                            if prim_name == "identity":
                                continue  # identity only valid for (0,0,0,0)
                            dist = _l1(prim_delta, target)
                            prio = _PRIORITY.get(prim_name, 1000)
                            key = (dist, prio)
                            if best is None or key < best[0]:
                                best = (key, prim_name, prim_smarts, prim_delta, prim_desc, prim_src)
                        _, name, smarts, delta, desc, src = best
                    pool[target] = {
                        'primitive': name,
                        'smarts': smarts,
                        'delta': delta,
                        'description': desc,
                        'source': src,
                    }
    return pool


STRATEGY_POOL: Dict[Tuple[int, int, int, int], dict] = _build_strategy_pool()

# Sanity check — we must have exactly 3^4 = 81 keys.
assert len(STRATEGY_POOL) == 81, f"Expected 81 strategies, got {len(STRATEGY_POOL)}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class StrategyPoolModifier:
    """Applies pre-built molecular modification strategies based on property gaps."""

    def __init__(self, config=None):
        self.config = config
        self.pool = STRATEGY_POOL
        logger.info(f"Initialized StrategyPoolModifier with {len(self.pool)} strategies")

    def _gap_to_direction(self, gap: float, threshold: float = 0.01) -> int:
        if gap > threshold:
            return +1
        if gap < -threshold:
            return -1
        return 0

    def get_strategy_key(self, property_gap: Dict[str, float]) -> Tuple[int, int, int, int]:
        return (
            self._gap_to_direction(property_gap.get('Density', 0)),
            self._gap_to_direction(property_gap.get('Det Velocity', 0)),
            self._gap_to_direction(property_gap.get('Det Pressure', 0)),
            self._gap_to_direction(property_gap.get('Hf solid', 0)),
        )

    def get_strategy(self, property_gap: Dict[str, float]) -> dict:
        key = self.get_strategy_key(property_gap)
        strategy = self.pool.get(key, self.pool[(0, 0, 0, 0)])
        logger.info(f"Strategy for {key}: {strategy['primitive']} — {strategy['description']}")
        return strategy

    def apply_strategy(self, smiles: str, strategy: dict) -> List[str]:
        smarts = strategy.get('smarts')
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        # Identity primitive: nothing to do.
        if smarts is None:
            return []

        results: List[str] = []
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
                        if not new_smiles or '.' in new_smiles or new_smiles == smiles:
                            continue
                        if Chem.MolFromSmiles(new_smiles) is not None:
                            results.append(new_smiles)
                    except Exception:
                        continue
        except Exception as e:
            logger.debug(f"Strategy '{strategy.get('primitive')}' application failed: {e}")

        return results

    def apply_strategies(self, smiles: str, property_gap: Dict[str, float],
                         target_count: int = 10) -> List[str]:
        primary = self.get_strategy(property_gap)
        key = self.get_strategy_key(property_gap)
        logger.info(f"Applying strategy pool to {smiles} (key={key}, target={target_count})")

        all_mods: "set[str]" = set(self.apply_strategy(smiles, primary))

        # Explore Δ-neighbours for diversity.
        if len(all_mods) < target_count:
            for delta in [(0, 0, 0, 1), (0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0),
                          (0, 0, 0, -1), (0, 0, -1, 0), (0, -1, 0, 0), (-1, 0, 0, 0)]:
                if len(all_mods) >= target_count:
                    break
                neighbor_key = tuple(max(-1, min(1, k + d)) for k, d in zip(key, delta))
                if neighbor_key == key:
                    continue
                neighbor = self.pool.get(neighbor_key)
                if neighbor is not None:
                    all_mods.update(self.apply_strategy(smiles, neighbor))

        if len(all_mods) < target_count:
            diverse = generate_diverse_modifications(
                smiles, target_count=target_count - len(all_mods)
            )
            for m in diverse:
                if '.' not in m and m != smiles:
                    all_mods.add(m)

        return list(all_mods)[:target_count]


def get_modification_strategies(smiles: str, property_gap: Dict[str, float],
                                target_count: int = 20) -> List[str]:
    return StrategyPoolModifier().apply_strategies(smiles, property_gap, target_count)


def default_modification_strategy(smiles: str, property_gap: Dict[str, float],
                                  target_count: int = 20) -> List[str]:
    mods = get_modification_strategies(smiles, property_gap, target_count)

    if len(mods) < target_count:
        diverse = generate_diverse_modifications(
            smiles, target_count=(target_count - len(mods)) * 2
        )
        existing = set(mods)
        for m in diverse:
            if m not in existing and m != smiles and '.' not in m:
                mods.append(m)
                existing.add(m)

    mods = [m for m in set(mods) if m != smiles]
    return mods[:target_count * 2]
