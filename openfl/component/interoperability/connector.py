import subprocess
import psutil
from logging import getLogger
import os

class Connector:
    """
    A skeletal base class for managing a server process of an external federated learning framework and 
    the connection with OpenFL's server
    """

    def __init__(self, command: list[str], component_name: str = "Base", **kwargs):
        """
        Initialize the OpenFL Connector.
        Args:
            command (list[str]): The command to run the server process.
            component_name (str): The name of the specific Connector component being used.
        """
        self.local_grpc_client = None
        self._command = command
        self._process = None
        self.logger = getLogger(__name__)
        self.component_name = component_name

    def start(self):
        """
        Start the server process with the provided command.
        """
        if self._process is None:
            self.logger.info(f"[OpenFL Connector] Starting server process: {' '.join(self._command)}")
            self._process = subprocess.Popen(self._command)
            self.logger.info(f"[OpenFL Connector] server process started with PID: {self._process.pid}")
        else:
            self.logger.info("[OpenFL Connector] server process is already running.")

    def stop(self):
        """
        Stop the server process if it is running.
        """

        # find and terminate sub_process processes
        main_process = psutil.Process()
        sub_processes = main_process.children(recursive=True)
        self.logger.info(f"[OpenFL Connector] Subprocesses: {sub_processes}")
        for sub_process in sub_processes:
            try:
                self.logger.info(f"[OpenFL Connector] Stopping server process with PID: {sub_process.pid}...")
                sub_process.terminate()
            except psutil.NoSuchProcess:
                continue

        _, still_alive = psutil.wait_procs(sub_processes, timeout=1)
        for p in still_alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                continue

        self.logger.info("[OpenFL Connector] Server processes stopped.")

    def get_local_grpc_client(self):
        """
        Get the local gRPC client.
        """
        return self.local_grpc_client
    
    def print_Connector_info(self):
        """
        Print information indicating which Connector component is being used.
        """
        self.logger.info(f"OpenFL Connector Enabled: {self.component_name}")
