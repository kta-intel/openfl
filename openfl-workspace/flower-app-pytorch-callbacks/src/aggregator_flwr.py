# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Aggregator module."""

import logging
import queue
import time
from threading import Lock
from typing import List, Optional

import openfl.callbacks as callbacks_module
from openfl.component.aggregator.straggler_handling import CutoffTimePolicy, StragglerPolicy
from openfl.databases import PersistentTensorDB, TensorDB
from openfl.interface.aggregation_functions import WeightedAverage
from openfl.pipelines import NoCompressionPipeline, TensorCodec
from openfl.protocols import base_pb2, utils
from openfl.protocols.base_pb2 import NamedTensor
from openfl.utilities import TaskResultKey, TensorKey, change_tags

logger = logging.getLogger(__name__)

from openfl.component import Aggregator

class Aggregator_subclass(Aggregator):
    """An Aggregator is the central node in federated learning.

    Attributes:
        round_number (int): Current round number.
        single_col_cert_common_name (str): Common name for single
            collaborator certificate.
        straggler_handling_policy: Policy for handling stragglers.
        _end_of_round_check_done (list of bool): Indicates if end of round
            check is done for each round.
        stragglers (list): List of stragglers.
        rounds_to_train (int): Number of rounds to train.
        authorized_cols (list of str): IDs of enrolled collaborators.
        uuid (int): Aggregator UUID.
        federation_uuid (str): Federation UUID.
        assigner: Object assigning tasks to collaborators.
        quit_job_sent_to (list): Collaborators sent a quit job.
        tensor_db (TensorDB): Object for tensor database.
        db_store_rounds* (int): Rounds to store in TensorDB.
        logger: Object for logging.
        write_logs (bool): Flag to enable metric writer callback.
        best_model_score (optional): Score of the best model. Defaults to
            None.
        metric_queue (queue.Queue): Queue for metrics.
        compression_pipeline: Pipeline for compressing data.
        tensor_codec (TensorCodec): Codec for tensor compression.
        init_state_path* (str): Initial weight file location.
        best_state_path* (str): Where to store the best model weight.
        last_state_path* (str): Where to store the latest model weight.
        best_tensor_dict (dict): Dict of the best tensors.
        last_tensor_dict (dict): Dict of the last tensors.
        collaborator_tensor_results (dict): Dict of collaborator tensor
            results.
        collaborator_tasks_results (dict): Dict of collaborator tasks
            results.
        collaborator_task_weight (dict): Dict of col task weight.
        lock: A threading Lock object used to ensure thread-safe operations.

    .. note::
        - plan setting
    """

    def __init__(
        self,
        aggregator_uuid,
        federation_uuid,
        authorized_cols,
        init_state_path,
        best_state_path,
        last_state_path,
        assigner,
        connector,
        use_delta_updates=True,
        straggler_handling_policy: StragglerPolicy = CutoffTimePolicy,
        rounds_to_train=256,
        single_col_cert_common_name=None,
        compression_pipeline=None,
        db_store_rounds=1,
        initial_tensor_dict=None,
        log_memory_usage=False,
        write_logs=False,
        callbacks: Optional[List] = None,
        persist_checkpoint=True,
        persistent_db_path=None,
        task_group: str = "learning",
    ):
        super().__init__(
            aggregator_uuid,
            federation_uuid,
            authorized_cols,
            init_state_path,
            best_state_path,
            last_state_path,
            assigner,
            connector,
            use_delta_updates,
            straggler_handling_policy,
            rounds_to_train,
            single_col_cert_common_name,
            compression_pipeline,
            db_store_rounds,
            initial_tensor_dict,
            log_memory_usage,
            write_logs,
            callbacks,
            persist_checkpoint,
            persistent_db_path,
            task_group,
        )

        if self.connector:
            self.connector.set_callback_for_local_grpc_client(self.tmp_callback)

    def tmp_callback(self, message_handler):
        if message_handler.origin == 'aggregator':
            model_dict = message_handler.extract_model_dict()
            round_number = message_handler.extract_round_number()
            print(f"Received model from {message_handler.origin} for round {round_number}")
        else:
            metrics = message_handler.extract_metrics()
            round_number = message_handler.extract_round_number()
            print(f"Received {metrics} from {message_handler.origin} for round {round_number}")
        
        # round_number, model_dict = self.connector.extract_model_dict(deserialized_response)
        # self.model_hash = get_model_hash(model_dict)
        # import pdb; pdb.set_trace()

    # def _log_iteration_info(self,phase='init'):
    #     """
    #     Calls the governor's log iteration info API
    #     """

    #     # Load the saved model because it's layers will never change
    #     tensor_dict, round_number = utils.deconstruct_model_proto(
    #         self.model, compression_pipeline=self.compression_pipeline)

    #     tensor_key_dict = {
    #         TensorKey(k, self.uuid, round_number, False, ('model',)):
    #             v for k, v in tensor_dict.items()
    #     }

    #     model_dict = {}
    #     for tensor_key in tensor_key_dict:
    #         name,_,_,_,_ = tensor_key
    #         nparray = self.tensor_db.get_tensor_from_cache(tensor_key)
    #         model_dict[name] = nparray

    #     self.model_hash = get_model_hash(model_dict)

    #     model_score = str(list(self.metric_queue.queue))
    #     model_update_download_location = "Undefined"
    #     misc_info = "Undefined"
    #     payload = self.plan_id + \
    #                 self.previous_model_hash + \
    #                 self.model_hash + \
    #                 model_score + \
    #                 model_update_download_location + \
    #                 misc_info
    #     signature = self.ecdsa_p_384_signer.sign(payload)