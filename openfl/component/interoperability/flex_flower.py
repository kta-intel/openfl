from openfl.component.interoperability.flex import FederatedLearningExchange
from openfl.transport.grpc.flex.flower.local_grpc_client import LocalGRPCClient

class FLEXFlower(FederatedLearningExchange):
    """
    FLEX subclass for the Flower framework.
    Responsible for generating the Flower server command.
    """

    def __init__(self, superlink_params: dict, **kwargs):
        """
        Initialize FLEXFlower by building the server command from the superlink_params.
        Args:
            superlink_params (dict): A dictionary of Flower server settings.
        """
        self.superlink_params = superlink_params
        command = self._build_command()
        super().__init__(command)
        
        flex_address = self.superlink_params.get("fleet-api-address", "0.0.0.0:9092")
        self.local_grpc_client = LocalGRPCClient(flex_address)

    def _build_command(self) -> list[str]:
        """
        Build the Flower server command based on settings.
        Args:
            superlink_params (dict): Settings to configure the Flower server.
        Returns:
            list[str]: A list representing the Flower server start command.
        """
        command = ["flower-superlink", "--fleet-api-type", "grpc-adapter"]

        if "insecure" in self.superlink_params:
            if self.superlink_params["insecure"]:
                command += ["--insecure"]

        if "serverappio-api-address" in self.superlink_params:
            command += ["--serverappio-api-address", str(self.superlink_params["serverappio-api-address"])]
            # flwr default: 0.0.0.0:9091

        if "fleet-api-address" in self.superlink_params:
            command += ["--fleet-api-address", str(self.superlink_params["fleet-api-address"])]
            # flwr default: 0.0.0.0:9092

        if "exec-api-address" in self.superlink_params:
            command += ["--exec-api-address", str(self.superlink_params["exec-api-address"])]
            # flwr default: 0.0.0.0:9093

        return command