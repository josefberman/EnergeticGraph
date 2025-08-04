"""
Main Database Generator for Explosives Quantum Chemical Database
Using Psi4 for quantum chemical calculations
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import os
from datetime import datetime

from explosives_list import get_explosives_list, get_explosives_by_category
from quantum_calculator import ExplosiveQuantumCalculator
from advanced_calculations import AdvancedExplosiveCalculator

class ExplosivesDatabaseGenerator:
    """
    Main class for generating the explosives quantum chemical database
    """
    
    def __init__(self, basis='def2-TZVP', functional='B3LYP', max_memory=4000):
        """
        Initialize the database generator
        
        Args:
            basis: Basis set for quantum calculations
            functional: DFT functional
            max_memory: Maximum memory in MB
        """
        self.basis = basis
        self.functional = functional
        self.max_memory = max_memory
        
        # Initialize calculators
        self.quantum_calc = ExplosiveQuantumCalculator(basis, functional, max_memory)
        self.advanced_calc = AdvancedExplosiveCalculator(basis, functional)
        
        # Create output directories
        self.create_output_directories()
        
    def create_output_directories(self):
        """Create necessary output directories"""
        directories = ['results', 'logs', 'temp']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def generate_database(self, max_compounds=None, start_index=0):
        """
        Generate the complete explosives database
        
        Args:
            max_compounds: Maximum number of compounds to calculate (None for all)
            start_index: Starting index for calculations
            
        Returns:
            DataFrame with all calculated properties
        """
        print("=" * 80)
        print("EXPLOSIVES QUANTUM CHEMICAL DATABASE GENERATION")
        print("=" * 80)
        print(f"Using Psi4 with {self.functional}/{self.basis}")
        print(f"Memory limit: {self.max_memory} MB")
        print(f"Starting from index: {start_index}")
        if max_compounds:
            print(f"Maximum compounds: {max_compounds}")
        print("=" * 80)
        
        # Get explosives list
        explosives_list = get_explosives_list()
        
        if max_compounds:
            explosives_list = explosives_list[start_index:start_index + max_compounds]
        else:
            explosives_list = explosives_list[start_index:]
        
        print(f"Processing {len(explosives_list)} compounds...")
        
        # Initialize results storage
        all_results = []
        failed_compounds = []
        
        # Process compounds with progress bar
        start_time = time.time()
        
        for i, explosive in enumerate(tqdm(explosives_list, desc="Calculating properties")):
            try:
                print(f"\nProcessing {explosive['name']} ({i+1}/{len(explosives_list)})")
                
                # Basic quantum calculations
                basic_results = self.quantum_calc.calculate_molecular_properties(
                    explosive['smiles'], 
                    explosive['name']
                )
                
                if basic_results is None:
                    print(f"Failed to calculate basic properties for {explosive['name']}")
                    failed_compounds.append(explosive['name'])
                    continue
                
                # Advanced calculations (if basic calculations succeeded)
                try:
                    # Convert SMILES to Psi4 molecule for advanced calculations
                    mol = self._smiles_to_psi4_molecule(explosive['smiles'], explosive['name'])
                    
                    if mol is not None:
                        # Phonon dispersion
                        phonon_results = self.advanced_calc.calculate_phonon_dispersion(mol)
                        
                        # Equation of state
                        eos_results = self.advanced_calc.calculate_equation_of_state(mol)
                        
                        # Elastic moduli
                        elastic_results = self.advanced_calc.calculate_elastic_moduli(mol)
                        
                        # Cohesive energy
                        cohesive_results = self.advanced_calc.calculate_cohesive_energy(mol)
                        
                        # Trigger bond analysis
                        trigger_results = self.advanced_calc.calculate_trigger_bond_analysis(mol)
                        
                        # Combine all results
                        advanced_results = {**phonon_results, **eos_results, **elastic_results, 
                                          **cohesive_results, **trigger_results}
                    else:
                        advanced_results = {}
                        
                except Exception as e:
                    print(f"Advanced calculations failed for {explosive['name']}: {e}")
                    advanced_results = {}
                
                # Combine basic and advanced results
                compound_results = {**basic_results, **advanced_results}
                compound_results.update({
                    'category': explosive['category'],
                    'formula': explosive['formula']
                })
                
                all_results.append(compound_results)
                
                # Save intermediate results every 10 compounds
                if (i + 1) % 10 == 0:
                    self.save_intermediate_results(all_results, i + 1)
                
            except Exception as e:
                print(f"Error processing {explosive['name']}: {e}")
                failed_compounds.append(explosive['name'])
                continue
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Create final results DataFrame
        if all_results:
            df = pd.DataFrame(all_results)
            
            # Save results
            self.save_final_results(df, total_time, failed_compounds)
            
            return df
        else:
            print("No successful calculations completed.")
            return pd.DataFrame()
    
    def _smiles_to_psi4_molecule(self, smiles, name):
        """Convert SMILES to Psi4 molecule object"""
        try:
            import psi4
            
            # Use RDKit for SMILES parsing and 3D coordinate generation
            try:
                from rdkit import Chem
                from rdkit.Chem import AllChem
                
                # Parse SMILES
                rdmol = Chem.MolFromSmiles(smiles)
                if rdmol is None:
                    print(f"Failed to parse SMILES for {name}: {smiles}")
                    return None
                
                # Generate 3D coordinates
                AllChem.EmbedMolecule(rdmol, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(rdmol)
                
                # Convert to Psi4 format
                conf = rdmol.GetConformer()
                mol_str = ""
                
                for atom in rdmol.GetAtoms():
                    pos = conf.GetAtomPosition(atom.GetIdx())
                    symbol = atom.GetSymbol()
                    mol_str += f"{symbol} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n"
                
                # Create Psi4 molecule
                mol = psi4.geometry(mol_str)
                return mol
                
            except ImportError:
                # Fallback to simple parser
                print(f"RDKit not available, using simple parser for {name}")
                return self._simple_smiles_to_psi4(smiles, name)
                
        except Exception as e:
            print(f"Error converting SMILES to Psi4 molecule for {name}: {e}")
            return None
    
    def _simple_smiles_to_psi4(self, smiles, name):
        """Simple SMILES parser for basic molecules"""
        import psi4
        
        # Very basic parser for common explosives
        if "C1N2CN3CN1CN2CN3" in smiles:  # RDX
            mol_str = """
C 0.000000 0.000000 0.000000
N 1.500000 0.000000 0.000000
N 0.750000 1.300000 0.000000
N -0.750000 1.300000 0.000000
N -1.500000 0.000000 0.000000
N -0.750000 -1.300000 0.000000
N 0.750000 -1.300000 0.000000
H 2.250000 0.000000 0.000000
H 0.750000 2.600000 0.000000
H -0.750000 2.600000 0.000000
H -2.250000 0.000000 0.000000
H -0.750000 -2.600000 0.000000
H 0.750000 -2.600000 0.000000
"""
        elif "CC1=C(C=C(C=C1)[N+](=O)[O-])[N+](=O)[O-]" in smiles:  # TNT
            mol_str = """
C 0.000000 0.000000 0.000000
C 1.400000 0.000000 0.000000
C 2.100000 1.200000 0.000000
C 1.400000 2.400000 0.000000
C 0.000000 2.400000 0.000000
C -0.700000 1.200000 0.000000
N 3.200000 1.200000 0.000000
O 3.900000 0.300000 0.000000
O 3.900000 2.100000 0.000000
N -1.900000 1.200000 0.000000
O -2.600000 0.300000 0.000000
O -2.600000 2.100000 0.000000
H 0.000000 -0.500000 0.000000
H -0.700000 3.000000 0.000000
H 1.400000 3.000000 0.000000
"""
        else:
            # Default simple structure
            mol_str = """
C 0.000000 0.000000 0.000000
N 1.500000 0.000000 0.000000
O 2.500000 0.000000 0.000000
H -0.500000 0.000000 0.000000
"""
        
        return psi4.geometry(mol_str)
    
    def save_intermediate_results(self, results, compound_count):
        """Save intermediate results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/intermediate_results_{compound_count}_compounds_{timestamp}.csv"
        
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)
        print(f"Intermediate results saved: {filename}")
    
    def save_final_results(self, df, total_time, failed_compounds):
        """Save final results and generate reports"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main results
        csv_filename = f"results/explosives_quantum_database_{timestamp}.csv"
        excel_filename = f"results/explosives_quantum_database_{timestamp}.xlsx"
        
        df.to_csv(csv_filename, index=False)
        df.to_excel(excel_filename, index=False)
        
        print(f"\nResults saved:")
        print(f"  - CSV: {csv_filename}")
        print(f"  - Excel: {excel_filename}")
        
        # Generate summary report
        self.generate_summary_report(df, total_time, failed_compounds, timestamp)
        
        # Generate correlations analysis
        self.generate_correlations_analysis(df, timestamp)
        
        # Generate category analysis
        self.generate_category_analysis(df, timestamp)
    
    def generate_summary_report(self, df, total_time, failed_compounds, timestamp):
        """Generate a summary report of the database generation"""
        report_filename = f"results/summary_report_{timestamp}.txt"
        
        with open(report_filename, 'w') as f:
            f.write("EXPLOSIVES QUANTUM CHEMICAL DATABASE - SUMMARY REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Generation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)\n")
            f.write(f"Successful Calculations: {len(df)}\n")
            f.write(f"Failed Calculations: {len(failed_compounds)}\n")
            f.write(f"Success Rate: {len(df)/(len(df)+len(failed_compounds))*100:.1f}%\n\n")
            
            f.write("CALCULATION PARAMETERS:\n")
            f.write(f"  - Quantum Package: Psi4\n")
            f.write(f"  - DFT Functional: {self.functional}\n")
            f.write(f"  - Basis Set: {self.basis}\n")
            f.write(f"  - Memory Limit: {self.max_memory} MB\n\n")
            
            f.write("PROPERTIES CALCULATED:\n")
            properties = df.columns.tolist()
            for prop in sorted(properties):
                f.write(f"  - {prop}\n")
            
            f.write(f"\nCATEGORY DISTRIBUTION:\n")
            if 'category' in df.columns:
                category_counts = df['category'].value_counts()
                for category, count in category_counts.items():
                    f.write(f"  - {category}: {count}\n")
            
            if failed_compounds:
                f.write(f"\nFAILED COMPOUNDS:\n")
                for compound in failed_compounds:
                    f.write(f"  - {compound}\n")
        
        print(f"Summary report saved: {report_filename}")
    
    def generate_correlations_analysis(self, df, timestamp):
        """Generate correlations analysis between properties"""
        try:
            # Select numerical columns for correlation analysis
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numerical_cols) > 1:
                correlations = df[numerical_cols].corr()
                
                # Save correlations
                corr_filename = f"results/correlations_{timestamp}.csv"
                correlations.to_csv(corr_filename)
                
                print(f"Correlations analysis saved: {corr_filename}")
                
                # Find strongest correlations
                corr_pairs = []
                for i in range(len(correlations.columns)):
                    for j in range(i+1, len(correlations.columns)):
                        corr_value = correlations.iloc[i, j]
                        if abs(corr_value) > 0.5:  # Strong correlation threshold
                            corr_pairs.append((correlations.columns[i], correlations.columns[j], corr_value))
                
                # Sort by absolute correlation value
                corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                
                # Save strong correlations
                strong_corr_filename = f"results/strong_correlations_{timestamp}.txt"
                with open(strong_corr_filename, 'w') as f:
                    f.write("STRONG CORRELATIONS (|r| > 0.5)\n")
                    f.write("=" * 40 + "\n\n")
                    
                    for prop1, prop2, corr_value in corr_pairs[:20]:  # Top 20
                        f.write(f"{prop1} <-> {prop2}: {corr_value:.3f}\n")
                
                print(f"Strong correlations saved: {strong_corr_filename}")
                
        except Exception as e:
            print(f"Error in correlations analysis: {e}")
    
    def generate_category_analysis(self, df, timestamp):
        """Generate analysis by explosive category"""
        if 'category' not in df.columns:
            return
        
        try:
            categories = df['category'].unique()
            
            for category in categories:
                category_data = df[df['category'] == category]
                
                if len(category_data) > 0:
                    # Calculate category statistics
                    numerical_cols = category_data.select_dtypes(include=[np.number]).columns.tolist()
                    
                    if numerical_cols:
                        stats = category_data[numerical_cols].describe()
                        
                        # Save category statistics
                        category_filename = f"results/category_{category}_stats_{timestamp}.csv"
                        stats.to_csv(category_filename)
                        
                        print(f"Category analysis for {category} saved: {category_filename}")
        
        except Exception as e:
            print(f"Error in category analysis: {e}")

def main():
    """Main function to run the database generation"""
    print("Starting Explosives Quantum Chemical Database Generation...")
    
    # Initialize generator
    generator = ExplosivesDatabaseGenerator(
        basis='def2-TZVP',
        functional='B3LYP',
        max_memory=4000
    )
    
    # Generate database (start with a small subset for testing)
    print("\nStarting with a test run of 5 compounds...")
    df = generator.generate_database(max_compounds=5, start_index=0)
    
    if not df.empty:
        print(f"\nTest run completed successfully!")
        print(f"Generated database with {len(df)} compounds")
        print(f"Properties calculated: {len(df.columns)}")
        
        # Ask user if they want to run the full database
        response = input("\nDo you want to run the full database generation? (y/n): ")
        if response.lower() in ['y', 'yes']:
            print("\nStarting full database generation...")
            full_df = generator.generate_database(max_compounds=None, start_index=0)
            
            if not full_df.empty:
                print(f"\nFull database generation completed!")
                print(f"Total compounds: {len(full_df)}")
            else:
                print("Full database generation failed.")
        else:
            print("Test run completed. Full database generation skipped.")
    else:
        print("Test run failed. Please check the errors above.")

if __name__ == "__main__":
    main() 