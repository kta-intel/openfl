import numpy as np
import os
from os import path, makedirs
import pandas as pd
from torch_geometric.data import download_url, extract_zip
import random
from rdkit import Chem
import sys

DEFAULT_PATH = path.join(path.expanduser('~'), '.openfl', 'data')
URLS =['https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip',
       'https://ndownloader.figshare.com/files/3195404']


def files_exist(files) -> bool:
    return len(files) != 0 and all([path.exists(f) for f in files])


def download(root_dir: str = DEFAULT_PATH):
    makedirs(root_dir, exist_ok=True)

    check_files = [path.join(root_dir, 'gdb9.sdf'), 
                   path.join(root_dir, 'gdb9.sdf.csv'), 
                   path.join(root_dir, 'uncharacterized.txt')]

    if files_exist(check_files):
        return


    file_path = download_url(URLS[0], root_dir)
    extract_zip(file_path, root_dir)
    os.unlink(file_path)

    file_path = download_url(URLS[1], root_dir)
    os.rename(path.join(root_dir, '3195404'),
              path.join(root_dir, 'uncharacterized.txt'))


def split_raw_data(root_dir, output_dir=None, num_splits=1):
    if output_dir is None:
        output_dir = root_dir

    check_files = [path.join(output_dir, f'split_{num_splits}', 'raw', 'gdb9.sdf'), 
                   path.join(output_dir, f'split_{num_splits}', 'raw', 'gdb9.sdf.csv'), 
                   path.join(output_dir, f'split_{num_splits}', 'raw', 'uncharacterized.txt')]

    if files_exist(check_files):
        return

    # Load the molecular structures from the SDF file
    sdf_path = path.join(root_dir, 'gdb9.sdf')
    sdf_supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
    molecules = [mol for mol in sdf_supplier if mol is not None]

    # Load the molecular properties from the CSV file
    csv_path = path.join(root_dir, 'gdb9.sdf.csv')
    properties_df = pd.read_csv(csv_path)

    # Load the uncharacterized molecules from the text file
    unchar_path = path.join(root_dir, 'uncharacterized.txt')
    with open(unchar_path, 'r') as f:
        uncharacterized_lines = f.readlines()
    header_lines = uncharacterized_lines[:9]
    data_lines = uncharacterized_lines[9:-2]
    trailing_lines = uncharacterized_lines[-2:]
    uncharacterized_indices = [int(x.split()[0]) - 1 for x in data_lines]

    # Create a mapping of molecule ID to molecule and properties
    mol_dict = {mol.GetProp('_Name'): mol for mol in molecules}
    prop_dict = properties_df.set_index('mol_id').to_dict(orient='index')

    # Combine the data into a list of tuples (molecule, properties)
    combined_data = [(mol_dict[mol_id], prop_dict[mol_id]) for mol_id in mol_dict if mol_id in prop_dict]

    # Randomly shuffle the combined data
    random.shuffle(combined_data)

    # Calculate the size of each split, accounting for any remainder
    split_size = len(combined_data) // num_splits
    remainder = len(combined_data) % num_splits

    # Create the splits
    splits = []
    start_idx = 0
    for i in range(num_splits):
        end_idx = start_idx + split_size + (1 if i < remainder else 0)
        splits.append(combined_data[start_idx:end_idx])
        start_idx = end_idx

    # Save the splits to the output directory
    for i, split in enumerate(splits):
        print(f"Processing split {i+1}/{num_splits}...")
        split_dir = path.join(output_dir, f'split_{i+1}', 'raw')
        makedirs(split_dir, exist_ok=True)

        # Initialize progress counters
        total_mols = len(split)
        checkpoint_interval = max(total_mols // 10, 1)  # Update every 10% or at least once

        # Save the SDF and CSV files for each split
        with Chem.SDWriter(path.join(split_dir, 'gdb9.sdf')) as writer:
            split_dfs = []
            for count, (mol, props) in enumerate(split, 1):
                writer.write(mol)
                props_df = pd.DataFrame([props], index=[mol.GetProp('_Name')])
                props_df.reset_index(inplace=True)
                props_df.rename(columns={'index': 'mol_id'}, inplace=True)
                split_dfs.append(props_df)
                if count % checkpoint_interval == 0 or count == total_mols:
                    print(f"  Processed {count}/{total_mols} molecules...")

            split_df = pd.concat(split_dfs, ignore_index=True)
            split_df.to_csv(path.join(split_dir, 'gdb9.sdf.csv'), index=False)

        # Save the uncharacterized molecules for each split
        with open(path.join(split_dir, 'uncharacterized.txt'), 'w') as f:
            f.writelines(header_lines)
            for line in data_lines:
                index = int(line.split()[0]) - 1
                mol_id = f'gdb_{index+1}'  # Convert index back to mol_id format
                if mol_id in [mol.GetProp('_Name') for mol, _ in split]:
                    f.write(line)
            f.writelines(trailing_lines)

        print(f"Completed split {i+1}/{num_splits}.")


def prepare_data(split_dir):
    check_files = [path.join(split_dir, 'train.csv'), 
                   path.join(split_dir, 'val.csv'), 
                   path.join(split_dir, 'test.csv')]

    if files_exist(check_files):
        return
        
    csv_path = path.join(split_dir, 'gdb9.sdf.csv')
    dataset = pd.read_csv(csv_path)

    n_samples = len(dataset)
    n_train = int(0.8 * n_samples)
    n_test = int(0.1 * n_samples)
    n_val = n_samples - (n_train + n_test)

    # Shuffle dataset with df.sample, then split
    train, val, test = np.split(dataset.sample(frac=1, random_state=42), [n_train, n_val + n_train])

    train.to_csv(path.join(split_dir, 'train.csv'))
    val.to_csv(path.join(split_dir, 'val.csv'))
    test.to_csv(path.join(split_dir, 'test.csv'))


if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print('Number of collaborators not provided. Exiting')
        sys.exit()

    base_dir = 'qm9'
    collaborators = int(sys.argv[1])
    download(base_dir)
    print("Dataset downloaded")

    output_dir = 'data' 

    # split data across collaborators
    split_raw_data(base_dir, output_dir, num_splits=collaborators)

    # prepare data for training for each collaborator
    for i in range(collaborators):
        prepare_data(path.join(output_dir, f'split_{i+1}', 'raw'))


