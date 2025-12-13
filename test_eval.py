from prediction_2 import import_data, evaluate_models
import pandas as pd
import os

if __name__ == "__main__":
    print("Importing data...")
    df = import_data()
    print(f"Data imported. Shape: {df.shape}")
    
    # Use a small subset to test quickly
    df_small = df.head(5).copy()
    
    print("Running evaluate_models on small subset...")
    evaluate_models(df_small)
    
    if os.path.exists('evaluated_molecules.xlsx'):
        print("Success: evaluated_molecules.xlsx created.")
    else:
        print("Failure: evaluated_molecules.xlsx not found.")


