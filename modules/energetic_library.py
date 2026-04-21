"""
Curated library of well-characterized energetic compounds.

Used by the RAG retriever as a fallback when a novel compound's IUPAC name
returns zero hits: we fall back to literature search for the most similar
known energetic material(s), penalising the confidence by the Tanimoto
similarity.

Each entry lists a common name (the primary literature search term),
a canonical SMILES, optional synonyms, and an informal chemical family.
All SMILES are validated at import time — invalid ones are skipped with a
warning, never silently dropped in a way that breaks similarity search.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KnownEnergetic:
    name: str
    smiles: str
    aliases: Tuple[str, ...] = ()
    family: str = ""


_RAW_LIBRARY: List[KnownEnergetic] = [
    # ---------------------------------------------------------------- Nitramines
    KnownEnergetic('RDX', 'O=N(=O)N1CN(N(=O)=O)CN(N(=O)=O)C1',
                   ('hexahydro-1,3,5-trinitro-1,3,5-triazine', 'cyclonite'), 'nitramine'),
    KnownEnergetic('HMX', 'O=N(=O)N1CN(N(=O)=O)CN(N(=O)=O)CN1N(=O)=O',
                   ('octahydro-1,3,5,7-tetranitro-1,3,5,7-tetrazocine', 'octogen'), 'nitramine'),
    KnownEnergetic('CL-20', 'O=N(=O)N1C2N(N(=O)=O)C3N(N(=O)=O)C1N(N(=O)=O)C2N3N(=O)=O',
                   ('hexanitrohexaazaisowurtzitane', 'HNIW'), 'caged nitramine'),
    KnownEnergetic('TNAZ', 'O=N(=O)C1(N(=O)=O)CN(N(=O)=O)C1',
                   ('1,3,3-trinitroazetidine',), 'nitramine'),
    KnownEnergetic('DNNC', 'O=N(=O)N1CC(N(=O)=O)CC1N(=O)=O',
                   ('1,3,5-trinitropyrrolidine',), 'nitramine'),
    KnownEnergetic('EDNA', 'O=N(=O)NCCN(=O)=O',
                   ('ethylenedinitramine', 'haleite'), 'nitramine'),
    KnownEnergetic('Tetryl', 'Cc1ccc([N+](=O)[O-])c([N+](=O)[O-])c1N(C)[N+](=O)[O-]',
                   ('2,4,6-trinitrophenylmethylnitramine',), 'nitramine'),
    KnownEnergetic('TEX', 'O=N(=O)N1C2OC3OC1N(N(=O)=O)C(N3N(=O)=O)N2N(=O)=O',
                   ('4,10-dinitro-2,6,8,12-tetraoxa-4,10-diazaisowurtzitane',), 'caged nitramine'),
    KnownEnergetic('K-6', 'O=C1N(N(=O)=O)CN(N(=O)=O)CN1N(=O)=O',
                   ('keto-RDX', '2-keto-1,3,5-trinitro-1,3,5-triazinane'), 'nitramine'),
    KnownEnergetic('BCHMX', 'O=N(=O)N1CC2CC1CN(N(=O)=O)C2',
                   ('bicyclo-HMX',), 'nitramine'),
    KnownEnergetic('MEDINA', 'O=N(=O)NCN(N(=O)=O)C',
                   ('methylenedinitramine',), 'nitramine'),
    KnownEnergetic('DINGU', 'O=C1N(N(=O)=O)C2N(N(=O)=O)C(=O)N(N(=O)=O)C2N1N(=O)=O',
                   ('1,4-dinitroglycoluril',), 'nitramine'),
    KnownEnergetic('TNGU', 'O=C1N(N(=O)=O)C2(N(=O)=O)N(N(=O)=O)C(=O)N(N(=O)=O)C2N1N(=O)=O',
                   ('tetranitroglycoluril', 'SORGUYL'), 'nitramine'),
    KnownEnergetic('TNAD', 'O=N(=O)N1CCN(N(=O)=O)C2CN(N(=O)=O)CCN12',
                   ('1,4,5,8-tetranitro-1,4,5,8-tetraazadecalin',), 'nitramine'),
    KnownEnergetic('HNFX', 'O=N(=O)N1CC(F)(F)C(F)(F)C1N(=O)=O',
                   ('tetrafluoro-tetranitro-diazacyclobutane',), 'fluorinated nitramine'),
    KnownEnergetic('DNGU', 'O=C1N(N(=O)=O)C2N(N(=O)=O)C(=O)NC2N1',
                   ('2,6-dinitroglycoluril',), 'nitramine'),

    # --------------------------------------------------------------- Nitroalkanes
    KnownEnergetic('Nitromethane', 'C[N+](=O)[O-]', ('NM',), 'nitroalkane'),
    KnownEnergetic('Nitroethane', 'CC[N+](=O)[O-]', (), 'nitroalkane'),
    KnownEnergetic('Tetranitromethane', 'O=N(=O)C(N(=O)=O)(N(=O)=O)N(=O)=O',
                   ('TNM',), 'nitroalkane'),
    KnownEnergetic('Hexanitroethane', 'O=N(=O)C(N(=O)=O)(N(=O)=O)C(N(=O)=O)(N(=O)=O)N(=O)=O',
                   ('HNE',), 'nitroalkane'),
    KnownEnergetic('BDNPF', 'O=N(=O)C(CON(=O)=O)(CON(=O)=O)N(=O)=O',
                   ('bis(dinitropropyl)formal',), 'nitroalkane'),
    KnownEnergetic('BDNPA', 'CC(C)(COC(C)(C)N(=O)=O)N(=O)=O',
                   ('bis(dinitropropyl) acetal',), 'nitroalkane'),

    # -------------------------------------------------------------- Nitroaromatics
    KnownEnergetic('TNT', 'Cc1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-]',
                   ('2,4,6-trinitrotoluene',), 'nitroaromatic'),
    KnownEnergetic('TATB', 'Nc1c([N+](=O)[O-])c(N)c([N+](=O)[O-])c(N)c1[N+](=O)[O-]',
                   ('1,3,5-triamino-2,4,6-trinitrobenzene',), 'nitroaromatic'),
    KnownEnergetic('Picric acid', 'Oc1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-]',
                   ('2,4,6-trinitrophenol',), 'nitroaromatic'),
    KnownEnergetic('Picramide', 'Nc1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-]',
                   ('2,4,6-trinitroaniline',), 'nitroaromatic'),
    KnownEnergetic('Styphnic acid', 'Oc1c([N+](=O)[O-])c(O)c([N+](=O)[O-])cc1[N+](=O)[O-]',
                   ('2,4,6-trinitroresorcinol',), 'nitroaromatic'),
    KnownEnergetic('TNB', 'O=[N+]([O-])c1cc([N+](=O)[O-])cc([N+](=O)[O-])c1',
                   ('1,3,5-trinitrobenzene',), 'nitroaromatic'),
    KnownEnergetic('Hexanitrobenzene',
                   'O=[N+]([O-])c1c([N+](=O)[O-])c([N+](=O)[O-])c([N+](=O)[O-])c([N+](=O)[O-])c1[N+](=O)[O-]',
                   ('HNB',), 'nitroaromatic'),
    KnownEnergetic('DATB', 'Nc1c([N+](=O)[O-])c(N)c([N+](=O)[O-])cc1[N+](=O)[O-]',
                   ('1,3-diamino-2,4,6-trinitrobenzene',), 'nitroaromatic'),
    KnownEnergetic('DNAN', 'COc1cc([N+](=O)[O-])ccc1[N+](=O)[O-]',
                   ('2,4-dinitroanisole',), 'nitroaromatic'),
    KnownEnergetic('2,4-DNT', 'Cc1ccc([N+](=O)[O-])cc1[N+](=O)[O-]',
                   ('2,4-dinitrotoluene',), 'nitroaromatic'),
    KnownEnergetic('2,6-DNT', 'Cc1c([N+](=O)[O-])cccc1[N+](=O)[O-]',
                   ('2,6-dinitrotoluene',), 'nitroaromatic'),
    KnownEnergetic('TNA', 'COc1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-]',
                   ('2,4,6-trinitroanisole',), 'nitroaromatic'),
    KnownEnergetic('Picryl chloride', 'Clc1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-]',
                   ('2,4,6-trinitrochlorobenzene',), 'nitroaromatic'),
    KnownEnergetic('Nitrobenzene', 'O=[N+]([O-])c1ccccc1', (), 'nitroaromatic'),
    KnownEnergetic('1,3-Dinitrobenzene', 'O=[N+]([O-])c1cccc([N+](=O)[O-])c1',
                   ('m-DNB',), 'nitroaromatic'),
    KnownEnergetic('HNBP',
                   'O=[N+]([O-])c1cc([N+](=O)[O-])cc(-c2cc([N+](=O)[O-])cc([N+](=O)[O-])c2[N+](=O)[O-])c1[N+](=O)[O-]',
                   ('hexanitrobiphenyl',), 'nitroaromatic'),
    KnownEnergetic('Hexanitrostilbene',
                   'O=[N+]([O-])c1cc([N+](=O)[O-])cc(/C=C/c2cc([N+](=O)[O-])cc([N+](=O)[O-])c2[N+](=O)[O-])c1[N+](=O)[O-]',
                   ('HNS',), 'nitroaromatic'),
    KnownEnergetic('4-Nitroaniline', 'Nc1ccc([N+](=O)[O-])cc1',
                   ('p-nitroaniline',), 'nitroaromatic'),
    KnownEnergetic('4-Nitrotoluene', 'Cc1ccc([N+](=O)[O-])cc1', (), 'nitroaromatic'),
    KnownEnergetic('PYX',
                   'O=N(=O)c1cc([N+](=O)[O-])c(Nc2ccc(Nc3cc([N+](=O)[O-])cc([N+](=O)[O-])c3-c3cccc(N(=O)=O)n3)nc2)c([N+](=O)[O-])c1',
                   ('picrylaminodinitropyridine',), 'nitroaromatic'),

    # -------------------------------------------------------------- Nitrate esters
    KnownEnergetic('PETN', 'O=N(=O)OCC(CON(=O)=O)(CON(=O)=O)CON(=O)=O',
                   ('pentaerythritol tetranitrate',), 'nitrate ester'),
    KnownEnergetic('Nitroglycerin', 'O=N(=O)OCC(ON(=O)=O)CON(=O)=O',
                   ('NG', 'glyceryl trinitrate'), 'nitrate ester'),
    KnownEnergetic('EGDN', 'O=N(=O)OCCON(=O)=O',
                   ('ethylene glycol dinitrate',), 'nitrate ester'),
    KnownEnergetic('DEGDN', 'O=N(=O)OCCOCCON(=O)=O',
                   ('diethylene glycol dinitrate',), 'nitrate ester'),
    KnownEnergetic('TMETN', 'CC(CON(=O)=O)(CON(=O)=O)CON(=O)=O',
                   ('metriol trinitrate',), 'nitrate ester'),
    KnownEnergetic('BTTN', 'O=N(=O)OCC(ON(=O)=O)C(ON(=O)=O)CON(=O)=O',
                   ('butanetriol trinitrate',), 'nitrate ester'),
    KnownEnergetic('Mannitol hexanitrate',
                   'O=N(=O)OCC(ON(=O)=O)C(ON(=O)=O)C(ON(=O)=O)C(ON(=O)=O)CON(=O)=O',
                   ('MHN', 'nitromannite'), 'nitrate ester'),
    KnownEnergetic('Erythritol tetranitrate',
                   'O=N(=O)OCC(ON(=O)=O)C(ON(=O)=O)CON(=O)=O',
                   ('ETN',), 'nitrate ester'),
    KnownEnergetic('Xylitol pentanitrate',
                   'O=N(=O)OCC(ON(=O)=O)C(ON(=O)=O)C(ON(=O)=O)CON(=O)=O',
                   ('XPN',), 'nitrate ester'),
    KnownEnergetic('Isosorbide dinitrate',
                   'O=N(=O)O[C@H]1CO[C@H]2[C@@H](ON(=O)=O)CO[C@@H]12',
                   ('ISDN',), 'nitrate ester'),
    KnownEnergetic('Methyl nitrate', 'CON(=O)=O', (), 'nitrate ester'),
    KnownEnergetic('Ethyl nitrate', 'CCON(=O)=O', (), 'nitrate ester'),

    # ---------------------------------------------------------- Azoles / heterocycles
    KnownEnergetic('FOX-7', 'NC(=C([N+](=O)[O-])[N+](=O)[O-])N',
                   ('DADNE', '1,1-diamino-2,2-dinitroethylene'), 'nitroenamine'),
    KnownEnergetic('FOX-12', 'NC(N)=NNC(=N)N.O=N(=O)NC(=N)N',
                   ('GUDN', 'guanylurea dinitramide'), 'ionic'),
    KnownEnergetic('LLM-105',
                   'Nc1nc(N)[n+]([O-])c([N+](=O)[O-])c1[N+](=O)[O-]',
                   ('2,6-diamino-3,5-dinitropyrazine-1-oxide', 'ANPZ-O'), 'nitroheterocycle'),
    KnownEnergetic('LLM-116', 'Nc1[nH]nc([N+](=O)[O-])c1[N+](=O)[O-]',
                   ('4-amino-3,5-dinitro-1H-pyrazole',), 'nitroheterocycle'),
    KnownEnergetic('NTO', 'O=C1NN=C(N(=O)=O)N1',
                   ('3-nitro-1,2,4-triazol-5-one',), 'nitroheterocycle'),
    KnownEnergetic('ANTA', 'Nc1nnc([N+](=O)[O-])[nH]1',
                   ('3-amino-5-nitro-1,2,4-triazole',), 'nitroheterocycle'),
    KnownEnergetic('DNAT', 'O=N(=O)c1nnc(N(=O)=O)[nH]1',
                   ('3,5-dinitro-1,2,4-triazole',), 'nitroheterocycle'),
    KnownEnergetic('DNI', 'O=N(=O)c1[nH]cc([N+](=O)[O-])n1',
                   ('2,4-dinitroimidazole',), 'nitroheterocycle'),
    KnownEnergetic('TNI', 'O=N(=O)c1[nH]c([N+](=O)[O-])c([N+](=O)[O-])n1',
                   ('trinitroimidazole',), 'nitroheterocycle'),
    KnownEnergetic('MTNI', 'Cn1cc([N+](=O)[O-])c([N+](=O)[O-])n1',
                   ('1-methyl-2,4-dinitroimidazole',), 'nitroheterocycle'),
    KnownEnergetic('MTNP', 'Cn1nc([N+](=O)[O-])c([N+](=O)[O-])n1',
                   ('1-methyl-3,5-dinitropyrazole',), 'nitroheterocycle'),
    KnownEnergetic('DNPP', 'O=N(=O)c1[nH]nc(-c2nnc([N+](=O)[O-])[nH]2)n1',
                   ('3,3\'-dinitro-5,5\'-bi-1H-1,2,4-triazole',), 'triazole'),
    KnownEnergetic('ICM-102', 'Nc1nn(C)c([N+](=O)[O-])c1[N+](=O)[O-]',
                   ('ICM-102',), 'nitroheterocycle'),
    KnownEnergetic('TKX-55', 'Nc1nn([N+](=O)[O-])c([N+](=O)[O-])n1',
                   ('TKX-55',), 'nitroheterocycle'),
    KnownEnergetic('MAD-X1', 'O=[N+]([O-])c1nc([N+](=O)[O-])c2nonc2n1',
                   ('MAD-X1',), 'nitroheterocycle'),
    KnownEnergetic('CL-14', 'Nc1nc([N+](=O)[O-])nc(N)n1',
                   ('CL-14 analogue',), 'nitroheterocycle'),

    # ---------------------------------------------------------- Tetrazines / triazines
    KnownEnergetic('HATO', 'Nc1nnc(N)nn1',
                   ('3,6-diamino-1,2,4,5-tetrazine',), 'tetrazine'),
    KnownEnergetic('LAX-112', 'Nc1n[n+]([O-])c(N)[n+]([O-])n1',
                   ('3,6-diamino-1,2,4,5-tetrazine-1,4-dioxide', 'DATDO'), 'tetrazine'),
    KnownEnergetic('BTATz', 'Nc1nnn(N)n1',
                   ('bis(tetrazolylamino)tetrazine',), 'tetrazine'),
    KnownEnergetic('DAAT', 'Nc1nnc(N)n1N=Nc1nnc(N)n1N',
                   ('3,3\'-diamino-4,4\'-azo-1,2,4-triazole',), 'triazole'),
    KnownEnergetic('HNAzO', 'O=N(=O)C1=NN(N(=O)=O)N=N1',
                   ('dinitroazotetrazine',), 'tetrazine'),
    KnownEnergetic('Cyanuric triazide',
                   '[N-]=[N+]=Nc1nc(N=[N+]=[N-])nc(N=[N+]=[N-])n1',
                   ('CTA', '2,4,6-triazido-1,3,5-triazine'), 'azide'),

    # ----------------------------------------------------------------- Furazans / furoxans
    KnownEnergetic('DAAF', 'Nc1nonc1N=Nc1c(N)non1',
                   ('3,3\'-diamino-4,4\'-azoxyfurazan',), 'furazan'),
    KnownEnergetic('DAAzF', 'Nc1nonc1N=Nc1c(N)non1',
                   ('3,3\'-diamino-4,4\'-azofurazan',), 'furazan'),
    KnownEnergetic('BTF',
                   'O=N(=O)c1c(N(=O)=O)c(N(=O)=O)c2nonc2c1N(=O)=O',
                   ('benzotrifuroxan',), 'furoxan'),
    KnownEnergetic('DNBF',
                   '[O-][n+]1onc2c([N+](=O)[O-])cc([N+](=O)[O-])cc21',
                   ('dinitrobenzofuroxan',), 'furoxan'),
    KnownEnergetic('BNFF', 'O=N(=O)c1nonc1-c1nonc1N(=O)=O',
                   ('3,3\'-dinitro-4,4\'-bifurazan',), 'furazan'),
    KnownEnergetic('DNTF', 'O=N(=O)c1nonc1-c1nonc1N(=O)=O',
                   ('3,4-bis(3-nitrofurazan-4-yl)furoxan',), 'furazan'),

    # ------------------------------------------------------------------- Tetrazoles
    KnownEnergetic('Tetrazole', 'c1nnn[nH]1', ('1H-tetrazole',), 'tetrazole'),
    KnownEnergetic('5-ATZ', 'Nc1nnn[nH]1', ('5-aminotetrazole', '5-AT'), 'tetrazole'),
    KnownEnergetic('5-NT', 'O=N(=O)c1nnn[nH]1', ('5-nitrotetrazole',), 'tetrazole'),
    KnownEnergetic('TKX-50', 'NO.NO.[O-]n1nnnc1-c1nnnn1[O-]',
                   ('dihydroxylammonium bistetrazolediolate',), 'ionic'),

    # ------------------------------------------------------- Guanidines and ureas
    KnownEnergetic('Nitroguanidine', 'NC(=N)N[N+](=O)[O-]',
                   ('NQ', 'picrite'), 'nitroamine'),
    KnownEnergetic('Nitrourea', 'O=C(N)N[N+](=O)[O-]', (), 'nitroamine'),
    KnownEnergetic('TAG', 'NC(N)=NN', ('triaminoguanidine',), 'amine'),

    # -------------------------------------------------------------- Ionic / salts
    KnownEnergetic('ADN', '[NH4+].[N-]([N+](=O)[O-])[N+](=O)[O-]',
                   ('ammonium dinitramide',), 'ionic'),
    KnownEnergetic('HNF', 'N.N.O=N(=O)C([N+](=O)[O-])[N+](=O)[O-]',
                   ('hydrazinium nitroformate',), 'ionic'),
    KnownEnergetic('Hydrazine nitrate', '[NH4+].O=N(=O)[O-]',
                   ('HN',), 'ionic'),
    KnownEnergetic('Hydrazine perchlorate', '[NH4+].O=[Cl](=O)(=O)[O-]',
                   ('HP',), 'ionic'),
    KnownEnergetic('Ammonium nitrate', '[NH4+].[O-][N+](=O)[O-]', ('AN',), 'oxidizer'),
    KnownEnergetic('Ammonium perchlorate', '[NH4+].O=[Cl](=O)(=O)[O-]',
                   ('AP',), 'oxidizer'),
    KnownEnergetic('Potassium perchlorate', '[K+].O=[Cl](=O)(=O)[O-]',
                   ('KP',), 'oxidizer'),
    KnownEnergetic('Potassium chlorate', '[K+].O=[Cl]([O-])=O', (), 'oxidizer'),
    KnownEnergetic('Potassium nitrate', '[K+].[O-][N+](=O)[O-]',
                   ('saltpeter',), 'oxidizer'),
    KnownEnergetic('Sodium nitrate', '[Na+].[O-][N+](=O)[O-]', (), 'oxidizer'),
    KnownEnergetic('Guanidine nitrate', 'NC(N)=N.O=[N+]([O-])O',
                   ('GN',), 'ionic'),
    KnownEnergetic('Urea nitrate', 'O=C(N)N.O=[N+]([O-])O',
                   ('UN',), 'ionic'),
    KnownEnergetic('Methylamine nitrate', 'CN.O=[N+]([O-])O',
                   ('MAN',), 'ionic'),
    KnownEnergetic('Aminoguanidine nitrate', 'NC(N)=NN.O=[N+]([O-])O',
                   ('AGN',), 'ionic'),
    KnownEnergetic('Diaminoguanidine nitrate', 'NC(=NN)NN.O=[N+]([O-])O',
                   ('DAGN',), 'ionic'),
    KnownEnergetic('TAGN', 'NC(N)=NN.O=N(=O)[O-]',
                   ('triaminoguanidinium nitrate',), 'ionic'),
    KnownEnergetic('Guanidinium perchlorate', 'N=C(N)N.O=[Cl](=O)(=O)[O-]',
                   ('CP',), 'ionic'),

    # ----------------------------------------------------------- Primary explosives
    KnownEnergetic('Lead azide', '[N-]=[N+]=[N-].[N-]=[N+]=[N-].[Pb+2]',
                   ('lead(II) azide',), 'primary explosive'),
    KnownEnergetic('Silver azide', '[N-]=[N+]=[N-].[Ag+]', (), 'primary explosive'),
    KnownEnergetic('Mercury fulminate', '[O-][N+]#C[Hg]C#[N+][O-]',
                   (), 'primary explosive'),
    KnownEnergetic('DDNP',
                   'O=[N+]([O-])c1cc([N+](=O)[O-])c(O)c(N=[N+]=[N-])c1',
                   ('diazodinitrophenol',), 'primary explosive'),
    KnownEnergetic('Lead styphnate',
                   '[Pb+2].[O-]c1c([N+](=O)[O-])c([O-])c([N+](=O)[O-])cc1[N+](=O)[O-]',
                   (), 'primary explosive'),
    KnownEnergetic('DBX-1', 'O=[N+]([O-])c1nnn([Cu])n1',
                   ('copper(I) 5-nitrotetrazolate',), 'primary explosive'),
    KnownEnergetic('Silver fulminate', '[O-][N+]#C[Ag]', (), 'primary explosive'),

    # ----------------------------------------------------------------- Peroxides
    KnownEnergetic('TATP', 'CC1(C)OOC(C)(C)OOC(C)(C)OO1',
                   ('triacetone triperoxide',), 'peroxide'),
    KnownEnergetic('DADP', 'CC1(C)OOC(C)(C)OO1',
                   ('diacetone diperoxide',), 'peroxide'),
    KnownEnergetic('HMTD', 'C1OCON2CON1CO2',
                   ('hexamethylene triperoxide diamine',), 'peroxide'),
    KnownEnergetic('MEKP', 'CC(C)(OO)OO',
                   ('methyl ethyl ketone peroxide',), 'peroxide'),

    # ------------------------------------------------------ Fluorodinitro / halogenated
    KnownEnergetic('DFP', 'O=N(=O)C(F)(F)N(=O)=O',
                   ('difluorodinitromethane',), 'fluoro nitro'),
    KnownEnergetic('FEFO', 'O=N(=O)C(F)(F)COCC(F)(F)N(=O)=O',
                   ('bis(2-fluoro-2,2-dinitroethyl)formal',), 'fluoro nitro'),
    KnownEnergetic('NMF', 'O=N(=O)C(F)(F)CON(=O)=O', (), 'fluoro nitro'),

    # ----------------------------------------------------------------- Azides / misc
    KnownEnergetic('Azidoethyl nitrate', 'O=N(=O)OCCN=[N+]=[N-]',
                   (), 'azide'),
    KnownEnergetic('GAP', 'C(C=C)(COCC=C)N=[N+]=[N-]',
                   ('glycidyl azide polymer monomer',), 'binder'),
    KnownEnergetic('HTPB', 'CC(=C)C=C',
                   ('hydroxyl-terminated polybutadiene monomer',), 'binder'),
]


def _fingerprint(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


@dataclass(frozen=True)
class _Entry:
    compound: KnownEnergetic
    mol: Chem.Mol
    canonical_smiles: str
    fp: object


def _build_library() -> List[_Entry]:
    entries: List[_Entry] = []
    skipped = 0
    for c in _RAW_LIBRARY:
        mol = Chem.MolFromSmiles(c.smiles)
        if mol is None:
            skipped += 1
            logger.warning(f"energetic_library: skipping {c.name!r} — invalid SMILES")
            continue
        entries.append(_Entry(
            compound=c,
            mol=mol,
            canonical_smiles=Chem.MolToSmiles(mol),
            fp=_fingerprint(mol),
        ))
    logger.info(f"energetic_library: loaded {len(entries)} known compounds "
                f"({skipped} skipped)")
    return entries


LIBRARY: List[_Entry] = _build_library()


def find_similar(smiles: str, top_k: int = 3,
                 min_tanimoto: float = 0.30) -> List[Tuple[KnownEnergetic, float]]:
    """Return up to ``top_k`` library entries most similar to ``smiles``.

    Exact matches (Tanimoto = 1.0 on identical canonical SMILES) are excluded
    since they can't serve as analogues.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    query_fp = _fingerprint(mol)
    query_canon = Chem.MolToSmiles(mol)

    scored: List[Tuple[KnownEnergetic, float]] = []
    for e in LIBRARY:
        if e.canonical_smiles == query_canon:
            continue
        t = DataStructs.TanimotoSimilarity(query_fp, e.fp)
        if t >= min_tanimoto:
            scored.append((e.compound, t))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
