# Copyright 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

from logging import getLogger
from openfl.federated import PyTorchDataLoader
from omegaconf import OmegaConf

# from src.digress.analysis.visualization import MolecularVisualization
from src.digress.datasets import qm9_dataset
from src.digress.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from src.digress.diffusion.extra_features_molecular import ExtraMolecularFeatures
from src.digress.metrics.molecular_metrics import SamplingMolecularMetrics
from src.digress.metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete


logger = getLogger(__name__)


def check_guidance_enabled(func):
    """Decorator to ensure guidance is enabled when regressor is True."""
    def check_guidance_and_call(self, *args, **kwargs):
        regressor = kwargs.get('regressor', False)
        if regressor and not getattr(getattr(self.cfg, 'guidance', None), 'use_guidance', False):
            raise RuntimeError("Guidance is not enabled. Regressor was not initialized")
        return func(self, *args, **kwargs)
    return check_guidance_and_call


class DiGressDataLoader(PyTorchDataLoader):
    """PyTorch data loader for MNIST dataset."""

    def __init__(self, data_path, batch_size, data_config_path, **kwargs):
        """Instantiate the data object.

        Args:
            data_path: The file path to the data
            batch_size: The batch size of the data loader
            **kwargs: Additional arguments, passed to super
        """
        super().__init__(batch_size, **kwargs)

        self.cfg = OmegaConf.load(data_config_path)

        self.cfg.train.batch_size = self.batch_size
        self.cfg.dataset.datadir = data_path

        # Diffusion:
        self.datamodule, self.model_kwargs = load_datamodule_and_model_args(self.cfg)
        # Regressor:
        if getattr(getattr(self.cfg, 'guidance', None), 'use_guidance', False):
            self.datamodule_r, self.model_kwargs_r = load_datamodule_and_model_args(self.cfg, 
                                                                                    regressor=True)

    @check_guidance_enabled                                                                         
    def get_feature_shape(self, regressor=False):
        """Return input dims."""
        if regressor:
            return self.model_kwargs_r['dataset_infos'].input_dims
        else:
            return self.model_kwargs['dataset_infos'].input_dims

    @check_guidance_enabled                                                                         
    def get_train_loader(self, regressor=False):
        """Return train dataloader."""
        if regressor:
            return self.datamodule_r.train_dataloader()
        else:
            return self.datamodule.train_dataloader()

    @check_guidance_enabled                                                                         
    def get_train_data_size(self, regressor=False):
        """Return size of train dataset."""
        if regressor:
            return len(self.datamodule_r.train_dataset)
        else:
            return len(self.datamodule.train_dataset)

    @check_guidance_enabled                                                                         
    def get_valid_loader(self, regressor=False):
        """Return validation dataloader."""
        if regressor:
            return self.datamodule_r.val_dataloader()
        else:
            return self.datamodule.val_dataloader()

    @check_guidance_enabled                                                                         
    def get_valid_data_size(self, regressor=False):
        """Return size of validation dataset."""
        if regressor:
            return len(self.datamodule_r.val_dataset)
        else:
            return len(self.datamodule.val_dataset)


def load_datamodule_and_model_args(cfg, regressor: bool = False):
    """
    Args:
        cfg (dict): configurations for the model and dataset.

    Return:
        datamodule and model args.
    """

    if cfg.dataset.name == 'qm9':
        if cfg.dataset.datadir == 'initialize':
            datamodule = None
            dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
            train_smiles = None
        else:
            datamodule = qm9_dataset.QM9DataModule(cfg=cfg, regressor=regressor)
            dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
            train_smiles = qm9_dataset.get_train_smiles(cfg=cfg, train_dataloader=datamodule.train_dataloader(),
                                                        dataset_infos=dataset_infos, evaluate_dataset=False)
    else:
        raise ValueError(f"Dataset {cfg.dataset.name} not supported.")


    if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
        extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    else:
        extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

    dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                            domain_features=domain_features, cfg=cfg, regressor=regressor)


    if regressor == True:
        # Output dims for regression only
        dataset_infos.output_dims = {'X': 0, 'E': 0, 'y': 2 if cfg.general.guidance_target == 'both' else 1}

    if cfg.model.type == 'discrete':
        train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
    else:
        raise ValueError(f"Model type {cfg.model.type} not supported.")

    sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles=train_smiles)
    # visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)

    model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                    'sampling_metrics': sampling_metrics, 'visualization_tools': None,
                    'extra_features': extra_features, 'domain_features': domain_features}

    return datamodule, model_kwargs