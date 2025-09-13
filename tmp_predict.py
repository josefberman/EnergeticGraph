import argparse
from prediction import predict_properties

parser = argparse.ArgumentParser(description="Predict energetic properties for a SMILES string")
parser.add_argument("--smiles", help="SMILES string to predict")
args = parser.parse_args()

props = predict_properties.invoke(args.smiles)
order = [
    'Density',
    'Detonation velocity',
    'Explosion capacity',
    'Explosion pressure',
    'Explosion heat'
]

# Human-readable output
print("Predicted properties for SMILES:", args.smiles)
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



