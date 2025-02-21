import os
import sys
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

def main(num_partitions):
    # Directory to save the partitions
    save_dir = "data"

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Initialize FederatedDataset
    partitioner = IidPartitioner(num_partitions=num_partitions)
    fds = FederatedDataset(
        dataset="uoft-cs/cifar10",
        partitioners={"train": partitioner},
    )

    # Function to save partition data
    def save_partition_data(partition_id, partition_train_test):
        partition_dir = os.path.join(save_dir, f"{partition_id}")
        os.makedirs(partition_dir, exist_ok=True)
        
        train_data_path = os.path.join(partition_dir, "train")
        test_data_path = os.path.join(partition_dir, "test")
        
        partition_train_test["train"].save_to_disk(train_data_path)
        partition_train_test["test"].save_to_disk(test_data_path)

    # Download, split, and save the dataset
    for partition_id in range(num_partitions):
        partition = fds.load_partition(partition_id)
        partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
        save_partition_data(partition_id, partition_train_test)

    print("Dataset downloaded, split, and saved successfully.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python setup_data.py <num_partitions>")
        sys.exit(1)
    
    num_partitions = int(sys.argv[1])
    main(num_partitions)