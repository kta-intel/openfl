# Copyright (C) 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

from logging import getLogger
from openfl.federated import PyTorchDataLoader
from omegaconf import OmegaConf

from src.digress.analysis.visualization import MolecularVisualization
from src.digress.datasets import guacamol_dataset
from src.digress.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from src.digress.diffusion.extra_features_molecular import ExtraMolecularFeatures
from src.digress.metrics.molecular_metrics import SamplingMolecularMetrics
from src.digress.metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete


logger = getLogger(__name__)


class GuacamolDataLoader(PyTorchDataLoader):
    """PyTorch data loader for MNIST dataset."""

    def __init__(self, data_path, batch_size, **kwargs):
        """Instantiate the data object.

        Args:
            data_path: The file path to the data
            batch_size: The batch size of the data loader
            **kwargs: Additional arguments, passed to super
        """
        super().__init__(batch_size, **kwargs)

        #TODO: Let user specify path in plan.yaml (let it default to workspace directory)
        self.cfg = OmegaConf.load('./src/digress_config.yaml')

        if self.cfg.dataset.name != 'guacamol':
            raise ValueError('GuacamolDataLoader is only compatible with the guacamol dataset')

        self.cfg.train.batch_size = self.batch_size
        self.cfg.dataset.datadir = data_path

        self.datamodule, self.model_kwargs = load_datamodule_and_model_args(self.cfg)

        if self.datamodule:
            self.datamodule = reduce_size(self.datamodule, train_size=0.10, val_size=0.10)

    def get_feature_shape(self):
        """Return input dims."""
        return self.model_kwargs['dataset_infos'].input_dims

    def get_train_loader(self):
        """Return train dataloader."""
        return self.datamodule.train_dataloader()
        # return self.datamodule.train_dataloader().dataset[:100]

    def get_train_data_size(self):
        """Return size of train dataset."""
        return len(self.datamodule.train_dataset)

    def get_valid_loader(self):
        """Return validation dataloader."""
        return self.datamodule.val_dataloader()
        # return self.datamodule.val_dataloader().dataset[:100]

    def get_valid_data_size(self):
        """Return size of validation dataset."""
        return len(self.datamodule.val_dataset)

def load_datamodule_and_model_args(cfg):
    """
    Args:
        cfg (dict): configurations for the model and dataset.

    Return:
        datamodule and model args.
    """
    # ######## DEBUG ###########
    # cfg.dataset.datadir = '../../federated_experiment/collaborator1/split_1'
    # datamodule = guacamol_dataset.GuacamolDataModule(cfg)
    # ######## DEBUG ###########

    if cfg.dataset.datadir == 'initialize':
        datamodule = None
    else:
        # cfg.dataset.datadir = '../../federated_experiment/collaborator1/split_1'
        datamodule = guacamol_dataset.GuacamolDataModule(cfg)

    dataset_infos = guacamol_dataset.Guacamolinfos(datamodule, cfg)

    if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
        extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    else:
        extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

    dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                    domain_features=domain_features, cfg=cfg)

    if cfg.model.type == 'discrete':
        train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)

    sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles=None)
    visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)

    model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                    'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                    'extra_features': extra_features, 'domain_features': domain_features}

    return datamodule, model_kwargs

def reduce_size(datamodule, train_size=1, val_size=1, seed=42):
    """
    Reduces the size of the datasets within the datamodule to the given percentages.

    Args:
        datamodule: The datamodule containing the datasets to be reduced.
        train_size: The percentage of the training dataset to keep.
        val_size: The percentage of the validation dataset to keep.
        seed: The random seed for reproducibility.

    Returns:
        A datamodule with reduced dataset sizes.
    """
    # Set the random seed for reproducibility
    import random
    from torch.utils.data import Subset

    random.seed(seed)

    # Reduce the size of the training dataset
    train_dataset = datamodule.train_dataset
    train_indices = list(range(len(train_dataset)))
    reduced_train_size = int(len(train_dataset) * train_size)
    train_subset_indices = random.sample(train_indices, reduced_train_size)
    datamodule.train_dataset = Subset(train_dataset, train_subset_indices)

    # Reduce the size of the validation dataset
    val_dataset = datamodule.val_dataset
    val_indices = list(range(len(val_dataset)))
    reduced_val_size = int(len(val_dataset) * val_size)
    val_subset_indices = random.sample(val_indices, reduced_val_size)
    datamodule.val_dataset = Subset(val_dataset, val_subset_indices)

    # Return the datamodule with the reduced datasets
    return datamodule



