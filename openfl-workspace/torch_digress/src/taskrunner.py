# Copyright (C) 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""
import numpy as np
from pytorch_lightning import Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Iterator
from typing import Tuple

from openfl.federated import PyTorchTaskRunner
from openfl.utilities import Metric, change_tags, TensorKey
from openfl.utilities.split import split_tensor_dict_for_holdouts

from src.digress.diffusion_model_discrete import DiscreteDenoisingDiffusion

class DiGress(PyTorchTaskRunner):
    def __init__(self, device="cuda", **kwargs):
        """Initialize.

        Args:
            device: The hardware device to use for training (Default = "cpu")
            **kwargs: Additional arguments to pass to the function

        """
        super().__init__(device=device, **kwargs)

        # Define the model
        if self.data_loader.cfg.model.type == 'discrete':
            self.model = DiscreteDenoisingDiffusion(cfg=self.data_loader.cfg, **self.data_loader.model_kwargs)
        else:
            raise ValueError(f"Model type: <{cfg.model.type}> not currently supported")

        self.optimizer = self.model.configure_optimizers()

    def train_task(
        self, col_name, round_num, input_tensor_dict, epochs=1, **kwargs
    ):
        """Train batches task.

        Train the model on the requested number of batches.

        Args:
            col_name:            Name of the collaborator
            round_num:           What round is it
            input_tensor_dict:   Required input tensors (for model)
            use_tqdm (bool):     Use tqdm to print a progress bar (Default=True)
            epochs:              The number of epochs to train

        Returns:
            global_output_dict:  Tensors to send back to the aggregator
            local_output_dict:   Tensors to maintain in the local TensorDB
        """
        if self.device=='cuda':
            torch.cuda.empty_cache()
            torch.set_float32_matmul_precision('medium')

        trainer = Trainer(accelerator=self.device, devices=[0], max_epochs=epochs)
        self.rebuild_model(round_num, input_tensor_dict)
        # import pdb; pdb.set_trace()

        trainer.fit(self.model, self.data_loader.get_train_loader())
        loss = trainer.logged_metrics['train loss'].item()

        metric = Metric(name='Train Loss', value=np.array(loss))

        origin = col_name
        tags = ("trained",)
        output_metric_dict = {
            TensorKey(metric.name, origin, round_num, True, ("metric",)): metric.value
        }

        # output model tensors (Doesn't include TensorKey)
        output_model_dict = self.get_tensor_dict(with_opt_vars=True)
        global_model_dict, local_model_dict = split_tensor_dict_for_holdouts(
            self.logger, output_model_dict, **self.tensor_dict_split_fn_kwargs
        )

        # Create global tensorkeys
        global_tensorkey_model_dict = {
            TensorKey(tensor_name, origin, round_num, False, tags): nparray
            for tensor_name, nparray in global_model_dict.items()
        }
        # Create tensorkeys that should stay local
        local_tensorkey_model_dict = {
            TensorKey(tensor_name, origin, round_num, False, tags): nparray
            for tensor_name, nparray in local_model_dict.items()
        }
        # The train/validate aggregated function of the next round will look
        # for the updated model parameters.
        # This ensures they will be resolved locally
        next_local_tensorkey_model_dict = {
            TensorKey(tensor_name, origin, round_num + 1, False, ("model",)): nparray
            for tensor_name, nparray in local_model_dict.items()
        }

        global_tensor_dict = {**output_metric_dict, **global_tensorkey_model_dict}
        local_tensor_dict = {
            **local_tensorkey_model_dict,
            **next_local_tensorkey_model_dict,
        }

        # Update the required tensors if they need to be pulled from the
        # aggregator
        # TODO this logic can break if different collaborators have different
        # roles between rounds.
        # For example, if a collaborator only performs validation in the first
        # round but training in the second, it has no way of knowing the
        # optimizer state tensor names to request from the aggregator because
        # these are only created after training occurs. A work around could
        # involve doing a single epoch of training on random data to get the
        # optimizer names, and then throwing away the model.
        if self.opt_treatment == "CONTINUE_GLOBAL":
            self.initialize_tensorkeys_for_functions(with_opt_vars=True)

        # This will signal that the optimizer values are now present,
        # and can be loaded when the model is rebuilt
        self.training_round_completed = True

        # Return global_tensor_dict, local_tensor_dict
        return global_tensor_dict, local_tensor_dict

    def validate_task(
        self, col_name, round_num, input_tensor_dict, use_tqdm=False, **kwargs
    ):
        """Validate Task.

        Run validation of the model on the local data.

        Args:
            col_name:            Name of the collaborator
            round_num:           What round is it
            input_tensor_dict:   Required input tensors (for model)
            use_tqdm (bool):     Use tqdm to print a progress bar (Default=True)

        Returns:
            global_output_dict:  Tensors to send back to the aggregator
            local_output_dict:   Tensors to maintain in the local TensorDB

        """
        if self.device=='cuda':
            torch.cuda.empty_cache()
            torch.set_float32_matmul_precision('medium')

        trainer = Trainer(accelerator=self.device, devices=[0], max_epochs=1)
        self.rebuild_model(round_num, input_tensor_dict, validation=True)

        # import pdb; pdb.set_trace()
        trainer.validate(self.model, self.data_loader.get_valid_loader())

        loss = trainer.logged_metrics['val loss'].item()
        metric = Metric(name='Val Loss', value=np.array(loss))

        origin = col_name
        suffix = "validate"
        if kwargs["apply"] == "local":
            suffix += "_local"
        else:
            suffix += "_agg"
        tags = ("metric",)
        tags = change_tags(tags, add_field=suffix)
        # TODO figure out a better way to pass in metric for this pytorch
        #  validate function
        output_tensor_dict = {
            TensorKey(metric.name, origin, round_num, True, tags): metric.value
        }

        # Empty list represents metrics that should only be stored locally
        return output_tensor_dict, {}

    def save_native(
        self,
        filepath,
        model_state_dict_key="model_state_dict",
        optimizer_state_dict_key="optimizer_state_dict",
        **kwargs,
    ):
        """
        Save model and optimizer states in a picked file specified by the \
        filepath. model_/optimizer_state_dicts are stored in the keys provided. \
        Uses pt.save().

        Args:
            filepath (string)                 : Path to pickle file to be
                                                created by torch.save().
            model_state_dict_key (string)     : key for model state dict
                                                in pickled file.
            optimizer_state_dict_key (string) : key for optimizer state
                                                dict in picked file.
            kwargs                            : unused

        Returns:
            None
        """
        pickle_dict = {
            model_state_dict_key: self.model.state_dict(),
            optimizer_state_dict_key: self.optimizer.state_dict(),
        }
        torch.save(pickle_dict, filepath)