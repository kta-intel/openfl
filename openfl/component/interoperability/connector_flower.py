import subprocess
from openfl.component.interoperability.connector import Connector
from openfl.transport.grpc.connector.flower.local_grpc_client import LocalGRPCClient

import os
os.environ["FLWR_HOME"] = os.path.join(os.getcwd(), "src/.flwr")

class ConnectorFlower(Connector):
    """
    Connector subclass for the Flower framework.
    Responsible for generating the Flower server command.
    """

    def __init__(self, superlink_params: dict, flwr_run_params: dict = None, **kwargs):
        """
        Initialize ConnectorFlower by building the server command from the superlink_params.
        Args:
            superlink_params (dict): A dictionary of Flower server settings.
            flwr_run_params (dict, optional): A dictionary containing the Flower run parameters. Defaults to None.
        """
        self.superlink_params = superlink_params
        self.flwr_run_params = flwr_run_params
        command = self._build_command()
        super().__init__(command, component_name="Flower")
        
        connector_address = self.superlink_params.get("fleet-api-address", "0.0.0.0:9092")
        self.local_grpc_client = LocalGRPCClient(connector_address)
        
        self.flwr_run_command = self._build_flwr_run_command() if flwr_run_params else None
        self.flwr_run_process = None

    def _build_command(self) -> list[str]:
        """
        Start the Flower SuperLink based on settings.
        Args:
            superlink_params (dict): Settings to configure the Flower server.
        Returns:
            list[str]: A list representing the Flower server start command.
        """
        if self.superlink_params.get("patch"):
            command = ["python", "src/patch/flower_superlink_patch.py", "--fleet-api-type", "grpc-adapter"]
        else:
            command = ["flower-superlink", "--fleet-api-type", "grpc-adapter"]

        if "insecure" in self.superlink_params:
            if self.superlink_params["insecure"]:
                command += ["--insecure"]
        else:
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

    def _build_flwr_run_command(self) -> list[str]:
        """
        Build the `flwr run` command to run the Flower application.
        Returns:
            list[str]: A list representing the flwr_run command.
        """
        flwr_app_name = self.flwr_run_params.get("flwr_app_name")
        federation_name = self.flwr_run_params.get("federation_name")

        if self.flwr_run_params.get("patch"):
            command = ["python", "src/patch/flwr_run_patch.py", "run", f"./src/{flwr_app_name}"]
        else:
            command = ["flwr", "run", f"./src/{flwr_app_name}"]

        if federation_name:
            command.append(federation_name)
        return command

    def start(self):
        """
        Start the `flower-superlink` and `flwr run` subprocesses with the provided commands.
        """
        super().start()
        
        if self.flwr_run_command:
            self.logger.info(f"[OpenFL Connector] Starting `flwr run` subprocess: {' '.join(self.flwr_run_command)}")
            self.flwr_run_process = subprocess.Popen(self.flwr_run_command)

    def stop(self):
        """
        Stop the `flower-superlink` subprocess.
        """
        super().stop()