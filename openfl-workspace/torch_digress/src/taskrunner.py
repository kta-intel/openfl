# Copyright 2024 Intel Corporation
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
from src.digress.guidance.qm9_regressor_discrete import Qm9RegressorDiscrete
from src.digress.guidance.guidance_diffusion_model_discrete import DiscreteDenoisingDiffusion as DiscreteDenoisingDiffusionGuidance

class DiGress(PyTorchTaskRunner):
    def __init__(self, device="cpu", precision=32, **kwargs):
        """Initialize.

        Args:
            device:    The hardware device to use for training (Default = "cpu")
            precision: The precision to use for computation (Default = 32)
            **kwargs:  Additional arguments to pass to the function

        """
        super().__init__(device=device, **kwargs)

        # Define the discrete model
        if self.data_loader.cfg.model.type == 'discrete':
            self.diffusion_model = DiscreteDenoisingDiffusion(cfg=self.data_loader.cfg, **self.data_loader.model_kwargs)
            self.optimizer = self.diffusion_model.configure_optimizers()

            if self.data_loader.model_kwargs_r:
                self.regressor = Qm9RegressorDiscrete(cfg=self.data_loader.cfg, **self.data_loader.model_kwargs_r)
                self.optimizer_r = self.regressor.configure_optimizers()
                self.conditional_model = DiscreteDenoisingDiffusionGuidance(cfg=self.data_loader.cfg, **self.data_loader.model_kwargs)
        else:
            raise ValueError(f"Model type: <{cfg.model.type}> not currently supported")

        self.precision = precision

    def train_task(
        self, col_name, round_num, input_tensor_dict, epochs=1, **kwargs
    ):
        """Train batches task.

        Train the model on the requested number of batches.

        Args:
            col_name:            Name of the collaborator
            round_num:           What round is it
            input_tensor_dict:   Required input tensors (for model)
            epochs:              The number of epochs to train
            **kwargs:            Additional arguments

        Returns:
            global_output_dict:  Tensors to send back to the aggregator
            local_output_dict:   Tensors to maintain in the local TensorDB
        """

        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.set_float32_matmul_precision('medium')

            # TODO: Let user specify which device
            trainer = Trainer(accelerator=self.device, devices=[0], max_epochs=1, precision=self.precision, 
                              enable_checkpointing=False, logger=False)
            if self.regressor:
                trainer_regressor = Trainer(accelerator=self.device, devices=[0], max_epochs=1, precision=self.precision,
                                            enable_checkpointing=False, logger=False)
        elif self.device == 'cpu':
            trainer = Trainer(accelerator=self.device, max_epochs=1, precision=self.precision, 
                              enable_checkpointing=False, logger=False)
            if self.regressor:
                trainer_regressor = Trainer(accelerator=self.device, max_epochs=1, precision=self.precision, 
                                            enable_checkpointing=False, logger=False)

        # Remove conditional model tensors (not part of the diffusion model, but can't be separately set during validation)
        # input_tensor_dict = {k: v for k, v in input_tensor_dict.items() if not k.startswith("conditional_model")}

        self.rebuild_model(round_num, input_tensor_dict)

        trainer.fit(self.diffusion_model, self.data_loader.get_train_loader())
        loss = trainer.logged_metrics['train loss'].item()
        metric1 = Metric(name='Train Loss (diffusion)', value=np.array(loss))

        origin = col_name
        tags = ("trained",)

        if self.regressor:
            trainer_regressor.fit(self.regressor, self.data_loader.get_train_loader(regressor=True))
            loss_regressor = trainer_regressor.logged_metrics['train loss'].item()

            metric2 = Metric(name='Train Loss (regressor)', value=np.array(loss_regressor))

            output_metric_dict = {
                TensorKey(metric1.name, origin, round_num, True, ("metric",)): metric1.value,
                TensorKey(metric2.name, origin, round_num, True, ("metric",)): metric2.value
            }
        else:
             output_metric_dict = {
                TensorKey(metric1.name, origin, round_num, True, ("metric",)): metric1.value,
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

        if self.opt_treatment == "CONTINUE_GLOBAL":
            self.initialize_tensorkeys_for_functions(with_opt_vars=True)

        # This will signal that the optimizer values are now present,
        # and can be loaded when the model is rebuilt
        self.training_round_completed = True

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
            use_tqdm (bool):     Use tqdm to print a progress bar (Default=False)
            **kwargs:            Additional arguments

        Returns:
            global_output_dict:  Tensors to send back to the aggregator
            local_output_dict:   Tensors to maintain in the local TensorDB

        """

        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.set_float32_matmul_precision('medium')

            # TODO: Let user specify which device
            trainer = Trainer(accelerator=self.device, devices=[0], max_epochs=1, precision=self.precision, 
                              enable_checkpointing=False, logger=False)
            if self.regressor:
                trainer_regressor = Trainer(accelerator=self.device, devices=[0], max_epochs=1, precision=self.precision,
                                            enable_checkpointing=False, logger=False)
                trainer_guidance = Trainer(accelerator=self.device, devices=[0], max_epochs=1, precision=self.precision,
                                        enable_checkpointing=False, logger=False)
        elif self.device == 'cpu':
            trainer = Trainer(accelerator=self.device, max_epochs=1, precision=self.precision, 
                              enable_checkpointing=False, logger=False)
            if self.regressor:
                trainer_regressor = Trainer(accelerator=self.device, max_epochs=1, precision=self.precision, 
                                            enable_checkpointing=False, logger=False)
                trainer_guidance = Trainer(accelerator=self.device, devices=[0], max_epochs=1, precision=self.precision,
                                        enable_checkpointing=False, logger=False)

        self.rebuild_model(round_num, input_tensor_dict)

        trainer.validate(self.diffusion_model, self.data_loader.get_valid_loader())
        metric1 = Metric(name='Val NLL (diffusion)', 
                        value=np.array(trainer.logged_metrics['val_nll'].item()))

        origin = col_name
        suffix = "validate"
        if kwargs["apply"] == "local":
            suffix += "_local"
        else:
            suffix += "_agg"
        tags = ("metric",)
        tags = change_tags(tags, add_field=suffix)

        if self.regressor:
            trainer_regressor.validate(self.regressor, self.data_loader.get_valid_loader(regressor=True))
            metric2 = Metric(name='Val MAE (regressor)', 
                            value=np.array(trainer_regressor.logged_metrics['val_mae'].item()))

            self.conditional_model.load_state_dict(self.diffusion_model.state_dict(), strict=False)
            self.conditional_model.guidance_model = self.regressor

            trainer_guidance.validate(self.conditional_model, self.data_loader.get_valid_loader(regressor=True))

            metric3 = Metric(name='Validity',
                             value=np.array(trainer_guidance.logged_metrics['Validity'].item()))
            metric4 = Metric(name='Uniqueness', 
                             value=np.array(trainer_guidance.logged_metrics['Uniqueness'].item()))

            output_tensor_dict = {
                TensorKey(metric1.name, origin, round_num, True, tags): metric1.value,
                TensorKey(metric2.name, origin, round_num, True, tags): metric2.value,
                TensorKey(metric3.name, origin, round_num, True, tags): metric3.value,
                TensorKey(metric4.name, origin, round_num, True, tags): metric4.value
            }
        else:
            metric2 = Metric(name='Validity',
                            value=np.array(trainer.logged_metrics['Validity'].item()))
            metric3 = Metric(name='Uniqueness', 
                            value=np.array(trainer.logged_metrics['Uniqueness'].item()))

            output_tensor_dict = {
                TensorKey(metric1.name, origin, round_num, True, tags): metric1.value,
                TensorKey(metric2.name, origin, round_num, True, tags): metric2.value,
                TensorKey(metric3.name, origin, round_num, True, tags): metric3.value
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
        pickle_dict_diffusion = {
            model_state_dict_key: self.diffusion_model.state_dict(),
            optimizer_state_dict_key: self.optimizer.state_dict(),
        }
        pickle_dict_regressor = {
            model_state_dict_key: self.regressor.state_dict(),
            optimizer_state_dict_key: self.optimizer_r.state_dict(),
        }
        torch.save(pickle_dict_diffusion, "diffusion_model_" + filepath)
        torch.save(pickle_dict_regressor, "regressor_" + filepath)

    def set_tensor_dict(self, tensor_dict, with_opt_vars=False):
        """Set the tensor dictionary.

        Args:
            tensor_dict: The tensor dictionary
            with_opt_vars (bool): Return the tensor dictionary including the
                                  optimizer tensors (Default=False)

        """
        # Sets tensors for model layers and optimizer state.
        # FIXME: self.parameters() instead? Unclear if load_state_dict() or
        #  simple assignment is better
        # for now, state dict gives us names, which is good
        # FIXME: do both and sanity check each time?

        # get device for correct placement of tensors
        device = self.device

        new_state = {}
        # Grabbing keys from model's state_dict helps to confirm we have
        # everything
        for k in self.state_dict():
            if k in tensor_dict:
                # Update the key in A with the value from B
                new_state[k] = torch.tensor(tensor_dict[k]).to(device)
            else:
                # If the key does not exist in B, keep the original value from A
                new_state[k] = self.state_dict()[k]

        # set model state
        self.load_state_dict(new_state)

        if with_opt_vars:
            # see if there is state to restore first
            if tensor_dict.pop("__opt_state_needed") == "true":
                _set_optimizer_state(self.get_optimizer(), device, tensor_dict)

            # sanity check that we did not record any state that was not used
            assert len(tensor_dict) == 0