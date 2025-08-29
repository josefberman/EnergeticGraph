from prediction import predict_properties

SMILES_TNEB = "O=[N+]([O-])c1cc(cc([N+]([O-])=O)c1O)[N+]([O-])=O"

props = predict_properties.invoke(SMILES_TNEB)
order = [
    'Density',
    'Detonation velocity',
    'Explosion capacity',
    'Explosion pressure',
    'Explosion heat'
]

# Human-readable output
print("Predicted properties for SMILES:", SMILES_TNEB)
max_key = max(len(k) for k in order)
for k in order:
    v = props.get(k, "NA")
    if isinstance(v, float):
        v = f"{v:.4f}"
    print(f"{k.ljust(max_key)} : {v}")

# Also print a compact CSV line (header + values) for easy copy/paste
header = ",".join(order)
values = ",".join(str(props.get(k, "")) for k in order)
print("\nCSV header:")
print(header)
print("CSV values:")
print(values)



