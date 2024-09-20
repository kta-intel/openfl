# Copyright 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import hashlib
import numpy as np
import os
from os import path, makedirs
import pandas as pd
from torch_geometric.data import download_url, extract_zip
import random
from rdkit import Chem

import sys
sys.path.insert(1, os.getcwd())
from src.digress.datasets import qm9_dataset


DEFAULT_PATH = path.join(path.expanduser('~'), '.openfl', 'data')
URLS =['https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip',
       'https://ndownloader.figshare.com/files/3195404']
EXPECTED_HASHES = {
    'qm9.zip': '1b2f3a9bee4e8f10d8dda1696cc6f1512b3970066efa995ac9a7049ab0dcdf0ea46787e6cadafe131da8ea46b42857af',
    'uncharacterized.txt': '5bb2f845068ce15c5b4a3cbb9ac1be1ba8eb8022c825c2c65e3f5eb0347dc38cb3e06fd7dae0115c3161e063a215d61b'
}


def calculate_file_hash(file_path, hash_type='sha384'):
    """Calculate the hash of a file."""
    hash_obj = hashlib.new(hash_type)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def verify_file_hash(file_path, expected_hash, hash_type='sha384'):
    """Verify the hash of a file."""
    calculated_hash = calculate_file_hash(file_path, hash_type)
    if calculated_hash != expected_hash:
        raise ValueError(f'Hash mismatch: {calculated_hash} != {expected_hash}')


def files_exist(files) -> bool:
    return len(files) != 0 and all([path.exists(f) for f in files])


def download(root_dir: str = DEFAULT_PATH):
    makedirs(root_dir, exist_ok=True)

    # Download and verify qm9.zip
    qm9_zip_path = download_url(URLS[0], root_dir)
    verify_file_hash(qm9_zip_path, EXPECTED_HASHES['qm9.zip'])
    extract_zip(qm9_zip_path, root_dir)
    os.unlink(qm9_zip_path)

    # Download and verify uncharacterized.txt
    unchar_txt_path = download_url(URLS[1], root_dir)
    os.rename(unchar_txt_path, path.join(root_dir, 'uncharacterized.txt'))
    verify_file_hash(path.join(root_dir, 'uncharacterized.txt'), EXPECTED_HASHES['uncharacterized.txt'])


def split_raw_data(root_dir, output_dir=None, num_splits=1):
    if output_dir is None:
        output_dir = root_dir

    check_files = [path.join(output_dir, f'{num_splits}', 'raw', 'gdb9.sdf'), 
                   path.join(output_dir, f'{num_splits}', 'raw', 'gdb9.sdf.csv'), 
                   path.join(output_dir, f'{num_splits}', 'raw', 'uncharacterized.txt')]

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
        split_dir = path.join(output_dir, f'{i+1}', 'raw')
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


def prepare_data(split_dir, pre_process=False):
    check_files = [path.join(split_dir, 'raw', 'train.csv'), 
                   path.join(split_dir, 'raw', 'val.csv'), 
                   path.join(split_dir, 'raw', 'test.csv')]

    if files_exist(check_files) == False:
        csv_path = path.join(split_dir, 'raw', 'gdb9.sdf.csv')
        dataset = pd.read_csv(csv_path)

        n_samples = len(dataset)
        n_train = int(0.8 * n_samples)
        n_test = int(0.1 * n_samples)
        n_val = n_samples - (n_train + n_test)

        # Shuffle dataset with df.sample, then split
        train, val, test = np.split(dataset.sample(frac=1, random_state=42), [n_train, n_val + n_train])

        train.to_csv(path.join(split_dir, 'raw', 'train.csv'))
        val.to_csv(path.join(split_dir, 'raw', 'val.csv'))
        test.to_csv(path.join(split_dir, 'raw', 'test.csv'))
        
    if pre_process:
        # Pre-processing data would require code from the model owner, so might not be realistic to include at this stage
        from omegaconf import OmegaConf
        cfg = OmegaConf.load("./src/digress/configs/qm9_config.yaml")
        cfg.dataset.datadir = split_dir
        datamodule = qm9_dataset.QM9DataModule(cfg)
        dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
        train_smiles = qm9_dataset.get_train_smiles(cfg=cfg, train_dataloader=datamodule.train_dataloader(),
                                                    dataset_infos=dataset_infos, evaluate_dataset=False)


def str_to_bool(s):
    return s.lower() in ['true', '1', 'y', 'yes']


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

    if len(sys.argv) > 2:
        pre_process = str_to_bool(sys.argv[2])
    else:
        pre_process = False

    # prepare data for training for each collaborator
    for i in range(collaborators):
        prepare_data(path.join(output_dir, f'{i+1}'), pre_process=pre_process)


