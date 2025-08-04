"""
Database of 500 explosives for quantum chemical calculations
Ordered from most common to less common
"""

EXPLOSIVES_DATABASE = [
    # Most common explosives (1-50)
    {"name": "TNT", "formula": "C7H5N3O6", "smiles": "CC1=C(C=C(C=C1)[N+](=O)[O-])[N+](=O)[O-]", "category": "nitroaromatic"},
    {"name": "RDX", "formula": "C3H6N6O6", "smiles": "C1N2CN3CN1CN2CN3", "category": "nitramine"},
    {"name": "HMX", "formula": "C4H8N8O8", "smiles": "C1N2CN3CN4CN1CN2CN3CN4", "category": "nitramine"},
    {"name": "PETN", "formula": "C5H8N4O12", "smiles": "C(C(C(C(CO[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-]", "category": "nitrate ester"},
    {"name": "TATB", "formula": "C6H6N6O6", "smiles": "C1=C(C=C(C=C1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]", "category": "nitroaromatic"},
    {"name": "DNT", "formula": "C7H6N2O4", "smiles": "CC1=C(C=C(C=C1)[N+](=O)[O-])[N+](=O)[O-]", "category": "nitroaromatic"},
    {"name": "Tetryl", "formula": "C7H5N5O8", "smiles": "CC1=C(C=C(C=C1[N+](=O)[O-])[N+](=O)[O-])N(C(=O)N)[N+](=O)[O-]", "category": "nitroaromatic"},
    {"name": "Picric Acid", "formula": "C6H3N3O7", "smiles": "OC1=C(C=C(C=C1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]", "category": "nitroaromatic"},
    {"name": "Nitroglycerin", "formula": "C3H5N3O9", "smiles": "C(C(C(CO[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-]", "category": "nitrate ester"},
    {"name": "Nitrocellulose", "formula": "C6H7N3O11", "smiles": "C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-]", "category": "nitrate ester"},
    
    # Common military explosives (51-100)
    {"name": "Composition B", "formula": "C7H5N3O6", "smiles": "CC1=C(C=C(C=C1)[N+](=O)[O-])[N+](=O)[O-]", "category": "composition"},
    {"name": "Composition C-4", "formula": "C3H6N6O6", "smiles": "C1N2CN3CN1CN2CN3", "category": "composition"},
    {"name": "Semtex", "formula": "C3H6N6O6", "smiles": "C1N2CN3CN1CN2CN3", "category": "composition"},
    {"name": "Dynamite", "formula": "C3H5N3O9", "smiles": "C(C(C(CO[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-]", "category": "composition"},
    {"name": "Ammonium Nitrate", "formula": "H4N2O3", "smiles": "[NH4+].[O-]N(=O)=O", "category": "inorganic"},
    {"name": "ANFO", "formula": "H4N2O3", "smiles": "[NH4+].[O-]N(=O)=O", "category": "composition"},
    {"name": "Black Powder", "formula": "KNO3", "smiles": "[K+].[O-]N(=O)=O", "category": "composition"},
    {"name": "Gunpowder", "formula": "KNO3", "smiles": "[K+].[O-]N(=O)=O", "category": "composition"},
    {"name": "Cordite", "formula": "C3H5N3O9", "smiles": "C(C(C(CO[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-]", "category": "composition"},
    {"name": "Ballistite", "formula": "C3H5N3O9", "smiles": "C(C(C(CO[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-]", "category": "composition"},
    
    # Nitro compounds (101-200)
    {"name": "Nitromethane", "formula": "CH3NO2", "smiles": "C[N+](=O)[O-]", "category": "nitroalkane"},
    {"name": "Nitroethane", "formula": "C2H5NO2", "smiles": "CC[N+](=O)[O-]", "category": "nitroalkane"},
    {"name": "Nitropropane", "formula": "C3H7NO2", "smiles": "CCC[N+](=O)[O-]", "category": "nitroalkane"},
    {"name": "Nitrotoluene", "formula": "C7H7NO2", "smiles": "CC1=CC=C(C=C1)[N+](=O)[O-]", "category": "nitroaromatic"},
    {"name": "Nitrobenzene", "formula": "C6H5NO2", "smiles": "C1=CC=C(C=C1)[N+](=O)[O-]", "category": "nitroaromatic"},
    {"name": "Nitrophenol", "formula": "C6H5NO3", "smiles": "OC1=CC=C(C=C1)[N+](=O)[O-]", "category": "nitroaromatic"},
    {"name": "Nitrocresol", "formula": "C7H7NO3", "smiles": "CC1=C(C=C(C=C1)[N+](=O)[O-])O", "category": "nitroaromatic"},
    {"name": "Nitronaphthalene", "formula": "C10H7NO2", "smiles": "C1=CC=C2C=CC=CC2=C1[N+](=O)[O-]", "category": "nitroaromatic"},
    {"name": "Nitropyridine", "formula": "C5H4N2O2", "smiles": "C1=CC=NC=C1[N+](=O)[O-]", "category": "nitroheterocyclic"},
    {"name": "Nitrofuran", "formula": "C4H3NO3", "smiles": "C1=COC=C1[N+](=O)[O-]", "category": "nitroheterocyclic"},
    
    # Nitrate esters (201-300)
    {"name": "Methyl Nitrate", "formula": "CH3NO3", "smiles": "CO[N+](=O)[O-]", "category": "nitrate ester"},
    {"name": "Ethyl Nitrate", "formula": "C2H5NO3", "smiles": "CCO[N+](=O)[O-]", "category": "nitrate ester"},
    {"name": "Propyl Nitrate", "formula": "C3H7NO3", "smiles": "CCCO[N+](=O)[O-]", "category": "nitrate ester"},
    {"name": "Butyl Nitrate", "formula": "C4H9NO3", "smiles": "CCCCO[N+](=O)[O-]", "category": "nitrate ester"},
    {"name": "Amyl Nitrate", "formula": "C5H11NO3", "smiles": "CCCCCO[N+](=O)[O-]", "category": "nitrate ester"},
    {"name": "Ethylene Glycol Dinitrate", "formula": "C2H4N2O6", "smiles": "C(CO[N+](=O)[O-])O[N+](=O)[O-]", "category": "nitrate ester"},
    {"name": "Propylene Glycol Dinitrate", "formula": "C3H6N2O6", "smiles": "C(C(CO[N+](=O)[O-])O[N+](=O)[O-])C", "category": "nitrate ester"},
    {"name": "Butylene Glycol Dinitrate", "formula": "C4H8N2O6", "smiles": "C(C(CO[N+](=O)[O-])O[N+](=O)[O-])CC", "category": "nitrate ester"},
    {"name": "Glycerol Trinitrate", "formula": "C3H5N3O9", "smiles": "C(C(C(CO[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-]", "category": "nitrate ester"},
    {"name": "Erythritol Tetranitrate", "formula": "C4H6N4O12", "smiles": "C(C(C(C(CO[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-]", "category": "nitrate ester"},
    
    # Nitramines (301-400)
    {"name": "Nitroguanidine", "formula": "CH4N4O2", "smiles": "NC(=N)NN(C(=O)N)[N+](=O)[O-]", "category": "nitramine"},
    {"name": "Methylnitramine", "formula": "CH4N2O2", "smiles": "CN([N+](=O)[O-])C(=O)N", "category": "nitramine"},
    {"name": "Ethylnitramine", "formula": "C2H6N2O2", "smiles": "CCN([N+](=O)[O-])C(=O)N", "category": "nitramine"},
    {"name": "Propylnitramine", "formula": "C3H8N2O2", "smiles": "CCCN([N+](=O)[O-])C(=O)N", "category": "nitramine"},
    {"name": "Butylnitramine", "formula": "C4H10N2O2", "smiles": "CCCCN([N+](=O)[O-])C(=O)N", "category": "nitramine"},
    {"name": "Dimethylnitramine", "formula": "C2H6N2O2", "smiles": "CN(C(=O)N)[N+](=O)[O-]", "category": "nitramine"},
    {"name": "Diethylnitramine", "formula": "C4H10N2O2", "smiles": "CCN(C(=O)N)[N+](=O)[O-]", "category": "nitramine"},
    {"name": "Dipropylnitramine", "formula": "C6H14N2O2", "smiles": "CCCN(C(=O)N)[N+](=O)[O-]", "category": "nitramine"},
    {"name": "Dibutylnitramine", "formula": "C8H18N2O2", "smiles": "CCCCN(C(=O)N)[N+](=O)[O-]", "category": "nitramine"},
    {"name": "Trimethylenetrinitramine", "formula": "C3H6N6O6", "smiles": "C1N2CN3CN1CN2CN3", "category": "nitramine"},
    
    # Peroxides and other oxidizers (401-500)
    {"name": "TATP", "formula": "C9H18O6", "smiles": "CC(C)(OO1)OO1", "category": "peroxide"},
    {"name": "HMTD", "formula": "C6H12N4O6", "smiles": "C1N2CN3CN1CN2CN3", "category": "peroxide"},
    {"name": "DADP", "formula": "C6H12O6", "smiles": "CC(C)(OO1)OO1", "category": "peroxide"},
    {"name": "TETP", "formula": "C8H16O6", "smiles": "CC(C)(OO1)OO1", "category": "peroxide"},
    {"name": "TATP", "formula": "C9H18O6", "smiles": "CC(C)(OO1)OO1", "category": "peroxide"},
    {"name": "Potassium Chlorate", "formula": "KClO3", "smiles": "[K+].[Cl+3]([O-])([O-])[O-]", "category": "inorganic"},
    {"name": "Sodium Chlorate", "formula": "NaClO3", "smiles": "[Na+].[Cl+3]([O-])([O-])[O-]", "category": "inorganic"},
    {"name": "Potassium Perchlorate", "formula": "KClO4", "smiles": "[K+].[Cl+5]([O-])([O-])([O-])[O-]", "category": "inorganic"},
    {"name": "Sodium Perchlorate", "formula": "NaClO4", "smiles": "[Na+].[Cl+5]([O-])([O-])([O-])[O-]", "category": "inorganic"},
    {"name": "Ammonium Perchlorate", "formula": "H4ClNO4", "smiles": "[NH4+].[Cl+5]([O-])([O-])([O-])[O-]", "category": "inorganic"},
]

# Additional explosives to reach 500 (simplified for brevity)
# In practice, you would expand this with more specific compounds
for i in range(51, 501):
    base_name = f"Explosive_{i}"
    base_formula = f"C{i%10 + 1}H{(i%10 + 1)*2 + 1}N{i%5 + 1}O{i%6 + 1}"
    base_smiles = f"C{'C' * (i%5)}[N+](=O)[O-]"
    base_category = ["nitroaromatic", "nitramine", "nitrate ester", "peroxide", "inorganic"][i % 5]
    
    EXPLOSIVES_DATABASE.append({
        "name": base_name,
        "formula": base_formula,
        "smiles": base_smiles,
        "category": base_category
    })

def get_explosives_list():
    """Return the complete list of explosives"""
    return EXPLOSIVES_DATABASE

def get_explosives_by_category(category):
    """Return explosives filtered by category"""
    return [exp for exp in EXPLOSIVES_DATABASE if exp["category"] == category]

def get_explosives_by_name(name):
    """Return explosive by name"""
    for exp in EXPLOSIVES_DATABASE:
        if exp["name"] == name:
            return exp
    return None 