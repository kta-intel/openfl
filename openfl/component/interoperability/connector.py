import subprocess
import psutil
import signal
import sys
from logging import getLogger

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

        # Register signal handler for clean termination
        signal.signal(signal.SIGINT, self._handle_sigint)

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
        if self._process:
            try:
                self.logger.info(f"[OpenFL Connector] Stopping server process with PID: {self._process.pid}...")
                # find and terminate sub_process processes
                main_process = psutil.Process(self._process.pid)
                sub_processes = main_process.children(recursive=True)
                for sub_process in sub_processes:
                    self.logger.info(f"[OpenFL Connector] Stopping server subprocess with PID: {sub_process.pid}...")
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
                self.logger.info("[OpenFL Connector] Server process stopped.")
            except Exception as e:
                self.logger.error(f"Error during graceful shutdown: {e}")
                self.logger.info("Attempting forceful termination of superlink process...")
                self._process.kill()
                self.logger.info("Superlink process forcefully terminated.")
        else:
            self.logger.info("[OpenFL Connector] No server process is currently running.")

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

    def _handle_sigint(self, signum, frame):
        """
        Handle the SIGINT signal (Ctrl+C) to cleanly stop the server process and its children.
        """
        self.logger.info("[OpenFL Connector] SIGINT received. Terminating server process...")
        self.stop()
        sys.exit(0)