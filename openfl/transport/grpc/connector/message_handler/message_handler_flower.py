from flwr.common.serde import message_from_proto
from flwr.common.recordset_compat import recordset_to_fitins, recordset_to_fitres, recordset_to_evaluateins
from flwr.common import parameters_to_ndarrays
from flwr.common import MessageType
import numpy as np

from openfl.transport.grpc.connector.message_handler.message_handler import MessageHandler

class MessageHandlerFlower(MessageHandler):
    def extract_round_number(self):
        """
        Extract the round number from the deserialized Flower message.

        Returns:
            The round number, or None if extraction fails.
        """
        try:
            message = message_from_proto(self.deserialized_message.messages_list[0])
            if message.metadata.message_type in [MessageType.TRAIN, MessageType.EVALUATE]:
                if message.metadata.message_type == MessageType.EVALUATE:
                    # if it is Eval, model will be the current round's global model
                    return int(message.metadata.group_id)
                elif message.metadata.message_type == MessageType.TRAIN:
                    if message.metadata.reply_to_message:
                        # if it is FitRes, model will be the current round's local model(s) before aggregation
                        return int(message.metadata.group_id)
                    else:
                        # if it is FitIns, model will be the previous round's global model
                        return int(message.metadata.group_id) - 1
        except Exception as e:
            print(f"Failed to extract round number. Error: {e}")
            return None

    def extract_model_dict(self):
        """
        Extract the model dictionary from the deserialized Flower message.

        Returns:
            A dictionary of named arrays, or None if extraction fails.
        """
        try:
            message = message_from_proto(self.deserialized_message.messages_list[0])
            if  message.metadata.message_type == MessageType.TRAIN:
                if message.metadata.reply_to_message:
                    # if it is FitRes, model will be the current round's local model(s) before aggregation
                    ins = recordset_to_fitres(message.content, keep_input=True)
                else:
                    # if it is FitIns, model will be the previous round's global model
                    ins = recordset_to_fitins(message.content, keep_input=True)
            elif message.metadata.message_type == MessageType.EVALUATE:
                # if it is EvalIns, model will be the current round's global model
                ins = recordset_to_evaluateins(message.content, keep_input=True)
            else:
                return None

            # Convert parameters to a list of numpy arrays
            nd_arrays: list[np.ndarray] = parameters_to_ndarrays(ins.parameters)

            # Create a dictionary with named arrays for saving
            return nd_arrays
            # return {f'array_{i}': arr for i, arr in enumerate(nd_arrays)}
        except Exception as e:
            print(f"Failed to extract model dictionary. Error: {e}")
            return None

    def extract_metrics(self):
        """
        Extract metrics from the deserialized Flower message.

        Returns:
            A dictionary of metrics, or None if extraction fails.
        """
        try:
            message = message_from_proto(self.deserialized_message.messages_list[0])
            metrics = {}

            if message.metadata.message_type == MessageType.TRAIN and message.metadata.reply_to_message:
                # This is the training loss when fine-tuning the global model at each collaborator
                metrics['local_model_validation'] = message.content.configs_records['fitres.metrics']['train_loss']
                metrics['num_examples'] = message.content.metrics_records['fitres.num_examples']['num_examples']

            elif message.metadata.message_type == MessageType.EVALUATE:
                # This is the validation loss when evaluating the global model at each collaborator
                metrics['aggregated_model_validation'] = message.content.metrics_records['evaluateres.loss']['loss']
                metrics['num_examples'] = message.content.metrics_records['evaluateres.num_examples']['num_examples']
            else:
                return None
            return metrics
        except Exception as e:
            print(f"Failed to extract metrics. Error: {e}")
            return None