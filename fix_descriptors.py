#!/usr/bin/env python3
"""
Script to fix all descriptor functions with proper error handling
"""

def fix_descriptor_functions():
    """Fix all descriptor functions in prediction.py"""
    
    # Read the current file
    with open('prediction.py', 'r') as f:
        content = f.read()
    
    # Define the functions to fix
    functions_to_fix = [
        'get_ono2_count',
        'get_ono_count', 
        'get_cno_count',
        'get_cnn_count',
        'get_nnn_count',
        'get_cnh2_count',
        'get_cnoc_count',
        'get_cnf_count',
        'get_c_count',
        'get_n_count',
        'get_h_count',
        'get_f_count',
        'get_no_count',
        'get_co_count',
        'get_coh_count',
        'get_noc_count',
        'calc_ob_100',
        'get_n_over_c'
    ]
    
    # Fix each function
    for func_name in functions_to_fix:
        # Find the function definition
        import re
        
        # Pattern to match function definition and body
        pattern = rf'def {func_name}\(smiles: str\):(.*?)(?=def|\Z)'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            func_body = match.group(1)
            
            # Check if it already has try-except
            if 'try:' not in func_body:
                # Extract the lines
                lines = func_body.strip().split('\n')
                
                # Find the mol = Chem.MolFromSmiles(smiles) line
                for i, line in enumerate(lines):
                    if 'mol = Chem.MolFromSmiles(smiles)' in line:
                        # Insert try-except around the function
                        new_lines = [
                            '    try:',
                            '        mol = Chem.MolFromSmiles(smiles)',
                            '        if mol is None:',
                            '            return 0',
                        ]
                        
                        # Add the rest of the function body
                        for j in range(i + 1, len(lines)):
                            if lines[j].strip() and not lines[j].strip().startswith('"""'):
                                new_lines.append('        ' + lines[j])
                        
                        new_lines.extend([
                            '    except:',
                            '        return 0'
                        ])
                        
                        # Replace the function body
                        new_func_body = '\n'.join(new_lines)
                        content = content.replace(match.group(0), f'def {func_name}(smiles: str):{new_func_body}')
                        break
    
    # Write the fixed content back
    with open('prediction.py', 'w') as f:
        f.write(content)
    
    print("✅ Fixed all descriptor functions with error handling")

if __name__ == "__main__":
    fix_descriptor_functions() 