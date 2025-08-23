import glob
import os
from bs4 import BeautifulSoup
import pandas as pd

# RDKit imports
from rdkit import Chem


def extract_from_html(html_path: str):
    print(html_path)
    # Read the HTML content (the file seems to be in GBK encoding)
    with open(f"{html_path}", "r", encoding="GBK", errors="replace") as f:
        html_text = f.read()

    # Parse with BeautifulSoup
    soup = BeautifulSoup(html_text, "html.parser")

    # Locate the main results table - you may need to refine this selector.
    # E.g. if the table has class="newform", we can do:
    results_table = soup.find("table", class_="newform")

    # Each data row is typically under <tr>. The first row(s) might be headers, so skip them.
    all_rows = results_table.find_all("tr")[7:]  # might need to offset more if the first few <tr> are headers

    extracted_data = []

    for row in all_rows:
        # Find the columns/cells
        cells = row.find_all("td")[:8]
        if not cells:
            continue  # skip empty/spacer rows

        # ---- 1) Extract numeric / physical data ----
        try:
            density_text = cells[2].get_text(strip=True)
            detonation_speed_text = cells[3].get_text(strip=True)
            explosion_pressure_text = cells[4].get_text(strip=True)
            explosive_heat_text = cells[5].get_text(strip=True)
            explosive_capacity_text = cells[6].get_text(strip=True)
        except IndexError:
            continue

        # ---- 2) Find the embedded MOL / Gaussian block ----
        param_tag = row.find("param", {"name": "mol"})
        if not param_tag:
            # If there's no <param name="mol"> in this row, skip or continue
            continue

        raw_mol_block = param_tag.get("value", "").strip()

        # Because your snippet shows lines continuing after the param, you might have to do
        # something like:
        #
        #   siblings_text = []
        #   for sib in param_tag.next_siblings:
        #       if not sib.name:  # it's a NavigableString
        #           siblings_text.append(str(sib))
        #       else:
        #           # If we reach another significant tag, maybe break out
        #           break
        #   extra_block = "\n".join(siblings_text).strip()
        #
        #   raw_mol_block = raw_mol_block + "\n" + extra_block
        #
        # Then parse out the real coordinates/bonds. This is very dependent on the real HTML.

        # ---- 3) Convert raw MOL (or SDF) text to SMILES via RDKit ----
        # If you have a valid MOL block in raw_mol_block, you can do:
        smiles = None
        if raw_mol_block:
            try:
                rdkit_mol = Chem.MolFromMolBlock(raw_mol_block, sanitize=True)
                if rdkit_mol is not None:
                    smiles = Chem.MolToSmiles(rdkit_mol)
                else:
                    # Might happen if the block is incomplete
                    smiles = ""
            except Exception as e:
                print("Failed to parse mol block for row:", e)
                smiles = ""

        # Store in a dictionary
        row_data = {
            "Density": density_text,
            "Detonation speed": detonation_speed_text,
            "Explosion pressure": explosion_pressure_text,
            "Explosive heat": explosive_heat_text,
            "Explosion capacity": explosive_capacity_text,
            "SMILES": smiles
        }
        extracted_data.append(row_data)
    # print(extracted_data)

    # Put everything in a DataFrame
    df = pd.DataFrame(extracted_data)

    # For inspection:
    # print(df.head(10))

    # Finally, you can save to CSV
    df.to_csv("extracted_chemical_data.csv", index=False, encoding="utf-8", mode='a')

def extract_all_html_files(dir_path: str):
    html_files = glob.glob(os.path.join(dir_path, "*.html"))
    for html_file in html_files:
        extract_from_html(html_file)

extract_all_html_files('./organchem database/')