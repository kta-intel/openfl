import subprocess
import psutil
from logging import getLogger

class FederatedLearningExchange:
    """
    A skeletal base class for managing a server process.
    """

    def __init__(self, command: list[str], component_name: str = "Base", **kwargs):
        """
        Initialize FLEX to run a server process.
        Args:
            command (list[str]): The command to run the server process.
            component_name (str): The name of the specific FLEX component being used.
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
            self.logger.info(f"[FLEX] Starting server process: {' '.join(self._command)}")
            self._process = subprocess.Popen(self._command)
            self.logger.info(f"[FLEX] server process started with PID: {self._process.pid}")
        else:
            self.logger.info("[FLEX] server process is already running.")

    def stop(self):
        """
        Stop the server process if it is running.
        """
        if self._process:
            self.logger.info(f"[FLEX] Stopping server process with PID: {self._process.pid}...")
            # find and terminate sub_process processes
            main_process = psutil.Process(self._process.pid)
            sub_processes = main_process.children(recursive=True)
            for sub_process in sub_processes:
                self.logger.info(f"[FLEX] Stopping server subprocess  with PID: {sub_process.pid}...")
                sub_process.terminate()
            _, still_alive = psutil.wait_procs(sub_processes, timeout=1)
            for p in still_alive:
                p.kill()
            # Terminate the main process
            self._process.terminate()
            try:
                self._process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
            self.logger.info("[FLEX] Server process stopped.")
        else:
            self.logger.info("[FLEX] No server process is currently running.")

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