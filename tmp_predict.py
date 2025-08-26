from prediction import predict_properties

SMILES_TNEB = "Cc1c(cc(cc1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]"

props = predict_properties(SMILES_TNEB)
order = [
    'Density',
    'Detonation velocity',
    'Explosion capacity',
    'Explosion pressure',
    'Explosion heat'
]

# Print as a single CSV line
print(
    ",".join(str(props[k]) for k in order)
)



