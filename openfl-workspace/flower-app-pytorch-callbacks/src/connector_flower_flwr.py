import toml
from openfl.transport.grpc.connector.flower.local_grpc_client import LocalGRPCClient

import os
os.environ["FLWR_HOME"] = os.path.join(os.getcwd(), "src/.flwr")
os.makedirs(os.environ["FLWR_HOME"], exist_ok=True)

from openfl.component import ConnectorFlower

class ConnectorFlower_subclass(ConnectorFlower):
    """
    Connector subclass for the Flower framework.
    Responsible for generating the Flower server command.
    """

    def __init__(self, 
                 superlink_params: dict, 
                 flwr_run_params: dict = None, 
                 automatic_shutdown: bool = False,
                 flwr_app_name: dict = None, 
                 **kwargs):
        """
        Initialize ConnectorFlower by building the server command from the superlink_params.
        Args:
            superlink_params (dict): A dictionary of Flower server settings.
            flwr_run_params (dict, optional): A dictionary containing the Flower run parameters. Defaults to None.
        """
        super().__init__(superlink_params, flwr_run_params, automatic_shutdown, **kwargs)
        self.automatic_shutdown = automatic_shutdown
        self.superlink_params = superlink_params

        self.flwr_app_name = flwr_app_name
        self.flwr_run_params = flwr_run_params
        self.rounds_to_train = self._get_rounds_to_train()

    def extract_model_dict(self, deserialized_message):
        """
        Acquire the round information from the Flower server.
        """
        from flwr.common.serde import message_from_proto
        from flwr.common.recordset_compat import recordset_to_fitins, recordset_to_evaluateins
        from flwr.common import parameters_to_ndarrays
        from flwr.common import MessageType
        import numpy as np

        message = message_from_proto(deserialized_message.messages_list[0])

        if message.metadata.message_type == MessageType.TRAIN:
            round_number = int(message.metadata.group_id) - 1
            ins=recordset_to_fitins(message.content, keep_input=True)

        elif message.metadata.message_type == MessageType.EVALUATE:
            round_number = int(message.metadata.group_id)
            ins=recordset_to_evaluateins(message.content, keep_input=True)

            # Convert parameters to a list of numpy arrays
        nd_arrays: list[np.ndarray] = parameters_to_ndarrays(ins.parameters)

        # Create a dictionary with named arrays for saving
        array_dict = {f'array_{i}': arr for i, arr in enumerate(nd_arrays)}

        return round_number, array_dict

    def set_callback_for_local_grpc_client(self, callback):
        self.local_grpc_client.set_callback(callback)

    def _get_rounds_to_train(self):
        # Load in the number of server rounds from the pyproject.toml file
        if self.flwr_app_name:
            flwr_app_name = self.flwr_app_name
        else:
            flwr_app_name = self.flwr_run_params.get("flwr_app_name")

        toml_file_path = os.path.join('src', flwr_app_name, 'pyproject.toml')
        toml_data = toml.load(toml_file_path)

        rounds_to_train = toml_data['tool']['flwr']['app']['config']['num-server-rounds']
        return rounds_to_train

