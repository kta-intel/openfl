import subprocess
import toml
import json
from openfl.component.interoperability.connector import Connector
from openfl.transport.grpc.connector.flower.local_grpc_client import LocalGRPCClient

import os
os.environ["FLWR_HOME"] = os.path.join(os.getcwd(), "src/.flwr")
os.makedirs(os.environ["FLWR_HOME"], exist_ok=True)

class ConnectorFlower(Connector):
    """
    Connector subclass for the Flower framework.
    Responsible for generating the Flower server command.
    """

    def __init__(self, flwr_app_name: dict, superlink_params: dict, flwr_run_params: dict = None, **kwargs):
        """
        Initialize ConnectorFlower by building the server command from the superlink_params.
        Args:
            superlink_params (dict): A dictionary of Flower server settings.
            flwr_run_params (dict, optional): A dictionary containing the Flower run parameters. Defaults to None.
        """
        self.flwr_app_name = flwr_app_name
        self.superlink_params = superlink_params
        self.flwr_run_params = flwr_run_params
        command = self._build_command()

        super().__init__(command, component_name="Flower")
        
        self.local_grpc_client = self._get_local_grpc_client()

        self.flwr_run_command = self._build_flwr_run_command() if flwr_run_params else None

    def _get_local_grpc_client(self):
        """
        Create and return a LocalGRPCClient instance based on superlink_params
        and the number of server rounds from the pyproject.toml file.

        Returns:
            LocalGRPCClient: An instance of LocalGRPCClient initialized with the
                             connector address and number of server rounds.
        """
        connector_address = self.superlink_params.get("fleet-api-address", "0.0.0.0:9092")

        # Load in the number of server rounds from the pyproject.toml file
        toml_file_path = os.path.join('src', self.flwr_app_name, 'pyproject.toml')
        toml_data = toml.load(toml_file_path)

        num_server_rounds = toml_data['tool']['flwr']['app']['config']['num-server-rounds']

        return LocalGRPCClient(connector_address, num_server_rounds)

    def _build_command(self) -> list[str]:
        """
        Start the Flower SuperLink based on superlink_params.

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
        federation_name = self.flwr_run_params.get("federation_name")

        if self.flwr_run_params.get("patch"):
            command = ["python", "src/patch/flwr_run_patch.py", "run", f"./src/{self.flwr_app_name}", "--format", "json"]
        else:
            command = ["flwr", "run", f"./src/{self.flwr_app_name}", "--format", "json"]

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
            flwr_run_process = subprocess.run(self.flwr_run_command, stdout=subprocess.PIPE, text=True)
            print(flwr_run_process.stdout)
            stdout_output = json.loads(flwr_run_process.stdout)
            self.local_grpc_client.set_run_id(stdout_output['run-id'])

    def stop(self):
        """
        Stop the `flower-superlink` subprocess.
        """
        super().stop()