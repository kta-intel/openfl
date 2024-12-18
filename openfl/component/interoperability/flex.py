import subprocess
from logging import getLogger

class FederatedLearningExchange:
    """
    A skeletal base class for managing a subprocess.
    """

    def __init__(self, command: list[str], component_name: str = "Base", **kwargs):
        """
        Initialize FLEX with a command to run as a subprocess.
        Args:
            command (list[str]): The command to run the server as a subprocess.
            component_name (str): The name of the specific FLEX component being used.
        """
        self.local_grpc_client = None
        self._command = command
        self._process = None
        self.logger = getLogger(__name__)
        self.component_name = component_name

    def start(self):
        """
        Start the subprocess with the provided command.
        """
        if self._process is None:
            self.logger.info(f"[FLEX] Starting subprocess: {' '.join(self._command)}")
            self._process = subprocess.Popen(self._command)
            self.logger.info(f"[FLEX] Subprocess started with PID: {self._process.pid}")
        else:
            self.logger.info("[FLEX] Subprocess is already running.")

    def stop(self):
        """
        Stop the subprocess if it is running.
        """
        if self._process:
            self.logger.info(f"[FLEX] Stopping subprocess with PID: {self._process.pid}...")
            self._process.terminate()
            self._process.wait()
            self._process = None
            self.logger.info("[FLEX] Subprocess stopped.")
        else:
            self.logger.info("[FLEX] No subprocess is currently running.")

    def get_local_grpc_client(self):
        """
        Get the local gRPC client.
        """
        return self.local_grpc_client
    
    def print_flex_info(self):
        """
        Print information indicating which FLEX component is being used.
        """
        self.logger.info(f"FLEX Enabled: {self.component_name}")