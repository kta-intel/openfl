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
                 **kwargs):
        """
        Initialize ConnectorFlower by building the server command from the superlink_params.
        Args:
            superlink_params (dict): A dictionary of Flower server settings.
            flwr_run_params (dict, optional): A dictionary containing the Flower run parameters. Defaults to None.
        """
        super().__init__(superlink_params, flwr_run_params, automatic_shutdown, **kwargs)

    def set_callback_for_local_grpc_client(self, callback):
        self.local_grpc_client.set_callback(callback)


