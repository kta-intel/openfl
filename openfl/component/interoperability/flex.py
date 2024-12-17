import subprocess
import logging

logger = logging.getLogger(__name__)

class FederatedLearningExchange:
    """
    A skeletal base class for managing a subprocess.
    """

    def __init__(self, command: list[str], **kwargs):
        """
        Initialize FLEX with a command to run as a subprocess.
        Args:
            command (list[str]): The command to run the server as a subprocess.
        """
        self._command = command
        self._process = None

    def start(self):
        """
        Start the subprocess with the provided command.
        """
        if self._process is None:
            logger.info(f"[FLEX] Starting subprocess: {' '.join(self._command)}")
            self._process = subprocess.Popen(self._command)
            logger.info(f"[FLEX] Subprocess started with PID: {self._process.pid}")
        else:
            logger.info("[FLEX] Subprocess is already running.")

    def stop(self):
        """
        Stop the subprocess if it is running.
        """
        if self._process:
            logger.info(f"[FLEX] Stopping subprocess with PID: {self._process.pid}...")
            self._process.terminate()
            self._process.wait()
            self._process = None
            logger.info("[FLEX] Subprocess stopped.")
        else:
            logger.info("[FLEX] No subprocess is currently running.")
