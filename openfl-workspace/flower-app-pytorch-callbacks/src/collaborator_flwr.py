# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Collaborator module."""

import logging
from enum import Enum
from time import sleep
from typing import List, Optional, Tuple

import openfl.callbacks as callbacks_module
from openfl.databases import TensorDB
from openfl.pipelines import NoCompressionPipeline, TensorCodec
from openfl.protocols import utils
from openfl.utilities import TensorKey
from openfl.transport.grpc import connector

logger = logging.getLogger(__name__)

from openfl.component import Collaborator
from openfl.component.collaborator.collaborator import DevicePolicy


class Collaborator_subclass(Collaborator):
    r"""The Collaborator object class.

    Attributes:
        collaborator_name (str): The common name for the collaborator.
        aggregator_uuid (str): The unique id for the client.
        federation_uuid (str): The unique id for the federation.
        client (object): The client object.
        task_runner (object): The task runner object.
        task_config (dict): The task configuration.
        opt_treatment (str)*: The optimizer state treatment.
        device_assignment_policy (str): The device assignment policy.
        delta_updates (bool)*: If True, only model delta gets sent. If False,
            whole model gets sent to collaborator.
        compression_pipeline (object): The compression pipeline.
        db_store_rounds (int): The number of rounds to store in the database.
        single_col_cert_common_name (str): The common name for the single
            column certificate.

    .. note::
        \* - Plan setting.
    """

    def __init__(
        self,
        collaborator_name,
        aggregator_uuid,
        federation_uuid,
        client,
        task_runner,
        task_config,
        opt_treatment="RESET",
        device_assignment_policy="CPU_ONLY",
        delta_updates=False,
        compression_pipeline=None,
        db_store_rounds=1,
        log_memory_usage=False,
        write_logs=False,
        callbacks: Optional[List] = None,
    ):
        super().__init__(
            collaborator_name,
            aggregator_uuid,
            federation_uuid,
            client,
            task_runner,
            task_config,
            opt_treatment,
            device_assignment_policy,
            delta_updates,
            compression_pipeline,
            db_store_rounds,
            log_memory_usage,
            write_logs,
            callbacks,
        )

    def tmp_callback(self, message_handler):
        model_dict = message_handler.extract_model_dict()
        round_number = message_handler.extract_round_number()
        print(f"Received model from {message_handler.origin} for round {round_number}")

    def do_task(self, task, round_number) -> dict:
        """Perform the specified task.

        Args:
            task (list_of_str): List of tasks.
            round_number (int): Actual round number.

        Returns:
            A dictionary of reportable metrics of the current collaborator for the task.
        """
        # map this task to an actual function name and kwargs
        if hasattr(self.task_runner, "TASK_REGISTRY"):
            func_name = task.function_name
            task_name = task.name
            kwargs = {}
            if task.task_type == "validate":
                if task.apply_local:
                    kwargs["apply"] = "local"
                else:
                    kwargs["apply"] = "global"
        else:
            if isinstance(task, str):
                task_name = task
            else:
                task_name = task.name
            func_name = self.task_config[task_name]["function"]
            kwargs = self.task_config[task_name]["kwargs"]
        if func_name=="start_client_adapter":
            # TODO: Need to determine a more general way to handle this in order to enable
            # additional tasks to be added to be added to Connector
            if hasattr(self.task_runner, func_name):
                method = getattr(self.task_runner, func_name)
                if callable(method):
                    framework = self.task_config['settings']["connect_to"]
                    LocalGRPCServer = connector.get_local_grpc_server(framework)
                    local_grpc_server = LocalGRPCServer(self.client, self.collaborator_name)
                    local_grpc_server.set_model_verification_call_back(self.tmp_callback)
                    method(local_grpc_server, **kwargs) 
                    # TODO: better to use self.send_task_results(global_output_tensor_dict, round_number, task_name)
                    # maybe set global_output_tensor to empty
                    self.client.send_local_task_results(self.collaborator_name, round_number, task_name)
                    metrics = {f'{self.collaborator_name}/start_client_adapter': 'Completed'}
                    return metrics
                else:
                    raise AttributeError(f"{func_name} is not callable on {self.task_runner}")
            else:
                raise AttributeError(f"{func_name} does not exist on {self.task_runner}")

        # this would return a list of what tensors we require as TensorKeys
        required_tensorkeys_relative = self.task_runner.get_required_tensorkeys_for_function(
            func_name, **kwargs
        )

        # models actually return "relative" tensorkeys of (name, LOCAL|GLOBAL,
        # round_offset)
        # so we need to update these keys to their "absolute values"
        required_tensorkeys = []
        for (
            tname,
            origin,
            rnd_num,
            report,
            tags,
        ) in required_tensorkeys_relative:
            if origin == "GLOBAL":
                origin = self.aggregator_uuid
            else:
                origin = self.collaborator_name

            # rnd_num is the relative round. So if rnd_num is -1, get the
            # tensor from the previous round
            required_tensorkeys.append(
                TensorKey(tname, origin, rnd_num + round_number, report, tags)
            )

        # print('Required tensorkeys = {}'.format(
        # [tk[0] for tk in required_tensorkeys]))
        input_tensor_dict = self.get_numpy_dict_for_tensorkeys(required_tensorkeys)

        # now we have whatever the model needs to do the task
        if hasattr(self.task_runner, "TASK_REGISTRY"):
            # New interactive python API
            # New `Core` TaskRunner contains registry of tasks
            func = self.task_runner.TASK_REGISTRY[func_name]
            logger.debug("Using Interactive Python API")

            # So far 'kwargs' contained parameters read from the plan
            # those are parameters that the eperiment owner registered for
            # the task.
            # There is another set of parameters that created on the
            # collaborator side, for instance, local processing unit identifier:s
            if (
                self.device_assignment_policy is DevicePolicy.CUDA_PREFERRED
                and len(self.cuda_devices) > 0
            ):
                kwargs["device"] = f"cuda:{self.cuda_devices[0]}"
            else:
                kwargs["device"] = "cpu"
        else:
            # TaskRunner subclassing API
            # Tasks are defined as methods of TaskRunner
            func = getattr(self.task_runner, func_name)
            logger.debug("Using TaskRunner subclassing API")

        global_output_tensor_dict, local_output_tensor_dict = func(
            col_name=self.collaborator_name,
            round_num=round_number,
            input_tensor_dict=input_tensor_dict,
            **kwargs,
        )

        # Save global and local output_tensor_dicts to TensorDB
        self.tensor_db.cache_tensor(global_output_tensor_dict)
        self.tensor_db.cache_tensor(local_output_tensor_dict)

        # send the results for this tasks; delta and compression will occur in
        # this function
        metrics = self.send_task_results(global_output_tensor_dict, round_number, task_name)
        return metrics

