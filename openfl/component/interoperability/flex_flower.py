from openfl.component.interoperability.flex import FederatedLearningExchange

class FLEXFlower(FederatedLearningExchange):
    """
    FLEX subclass for the Flower framework.
    Responsible for generating the Flower server command.
    """

    def __init__(self, superlink_params: dict, **kwargs):
        """
        Initialize FLEXFlower by building the server command from settings.
        Args:
            settings (dict): A dictionary of Flower server settings.
        """
        self._settings = superlink_params
        command = self._build_command(superlink_params)
        super().__init__(command)

    def _build_command(self, superlink_params: dict) -> list[str]:
        """
        Build the Flower server command based on settings.
        Args:
            settings (dict): Settings to configure the Flower server.
        Returns:
            list[str]: A list representing the Flower server start command.
        """
        command = ["flower-superlink", "--fleet-api-type", "grpc-adapter"]

        if "insecure" in superlink_params:
            if superlink_params["insecure"]:
                command += ["--insecure"]

        if "serverappio-api-address" in superlink_params:
            command += ["--serverappio-api-address", str(superlink_params["serverappio-api-address"])]
            # flwr default: 0.0.0.0:9091

        if "fleet-api-address" in superlink_params:
            command += ["--fleet-api-address", str(superlink_params["fleet-api-address"])]
            # flwr default: 0.0.0.0:9092

        if "exec-api-address" in superlink_params:
            command += ["--exec-api-address", str(superlink_params["exec-api-address"])]
            # flwr default: 0.0.0.0:9093

        return command

    @property
    def address(self) -> str:
        """
        Get the fleet API address from the settings.
        Returns:
            str: The fleet API address.
        """
        return self._settings.get("fleet-api-address", "0.0.0.0:9092")