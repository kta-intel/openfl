import threading
import grpc
from concurrent.futures import ThreadPoolExecutor
from flwr.proto import grpcadapter_pb2_grpc
from multiprocessing import cpu_count
from openfl.federated.task.runner import TaskRunner
from openfl.transport.grpc.connector.flower.local_grpc_server import LocalGRPCServer
import subprocess
from logging import getLogger
import signal
import psutil
import time
import os

os.environ["FLWR_HOME"] = os.path.join(os.getcwd(), "src/.flwr")
os.makedirs(os.environ["FLWR_HOME"], exist_ok=True)

class FlowerTaskRunner(TaskRunner):
    """
    FlowerTaskRunner is a task runner that executes Flower SuperNode
    to initialize the experiment from the client side.

    This class is responsible for starting a local gRPC server and a Flower SuperNode
    in a subprocess. It also provides options for automatic shutdown based on subprocess
    activity.

    Shutdown Options:
    - Manual Shutdown: The server and supernode process can be manually stopped by pressing CTRL+C.
    - Automatic Shutdown: If enabled, the system will monitor the activity of subprocesses and 
      automatically shut down if no new subprocess starts within a certain time frame.
    """
    def __init__(self, auto_shutdown=True, **kwargs):
        """
        Initializes the FlowerTaskRunner.

        Args:
            auto_shutdown (bool): Whether to enable automatic shutdown based on subprocess activity.
                Default is True. Set to False for long-lived components.
            **kwargs: Additional parameters to pass to the functions.
        """
        super().__init__(**kwargs)
        self.logger = getLogger(__name__)
        self.num_partitions = self.data_loader.get_node_configs()[0]
        self.partition_id = self.data_loader.get_node_configs()[1]

        base_port = 5000
        self.client_port = base_port + self.partition_id
        self.auto_shutdown = auto_shutdown
        self.patch = kwargs.get('patch')
        self.shutdown_requested = False # Flag signal shutdown

    def start_client_adapter(self, openfl_client, collaborator_name, **kwargs):
        """
        Starts the local gRPC server and the Flower SuperNode.

        Args:
            openfl_client: The OpenFL client instance used to communicate with the OpenFL server.
            collaborator_name: The name of the collaborator.
            **kwargs: Additional parameters, including 'local_server_port'.

        The method performs the following steps:
        1. Starts a local gRPC server to handle communication between the OpenFL client and the Flower SuperNode.
        2. Launches the Flower SuperNode in a subprocess.
        3. Sets up signal handlers for manual shutdown (via CTRL+C).
        4. If auto_shutdown is enabled, monitors run activity and initiates shutdown if no new subprocesses start within the expected time frame.

        Shutdown Process:
        - When a shutdown signal (SIGINT or SIGTERM) is received, the method will:
            1. Terminate all child processes of the SuperNode subprocess.
            2. Terminate the main SuperNode subprocess.
            3. Stop the gRPC server.
            4. Log the shutdown process and set the termination event to stop the server.
        """
        local_server_port = kwargs.get('local_server_port')

        def message_callback():
            """
            Callback function to handle messaging events.
            If auto_shutdown is enabled, logs a message indicating that the final reply 
            has been sent and triggers the SIGTERM signal handler to initiate shutdown.
            """
            self.shutdown_requested = True

        server = grpc.server(ThreadPoolExecutor(max_workers=cpu_count()))
        grpcadapter_pb2_grpc.add_GrpcAdapterServicer_to_server(
            LocalGRPCServer(openfl_client, collaborator_name, message_callback), server
        )
        server.add_insecure_port(f'[::]:{local_server_port}')
        server.start()
        self.logger.info(f"OpenFL local gRPC server started, listening on port {local_server_port}.")

        if self.patch:
            command = [
                "python",
                "src/patch/flower_supernode_patch.py",
                "--insecure",
                "--grpc-adapter",
                "--superlink", f"127.0.0.1:{local_server_port}",
                "--clientappio-api-address", f"127.0.0.1:{self.client_port}",
                "--node-config", f"num-partitions={self.num_partitions} partition-id={self.partition_id}"
            ]
        else:
            command = [
                "flower-supernode",
                "--insecure",
                "--grpc-adapter",
                "--superlink", f"127.0.0.1:{local_server_port}",
                "--clientappio-api-address", f"127.0.0.1:{self.client_port}",
                "--node-config", f"num-partitions={self.num_partitions} partition-id={self.partition_id}"
            ]

        supernode_process = subprocess.Popen(command, shell=False)

        termination_event = threading.Event()

        def signal_handler(_sig, _frame):
            """
            Handles shutdown signals (SIGINT or SIGTERM) to terminate the SuperNode process and stop the local gRPC server.
            Args:
                _sig: The signal number.
                _frame: The current stack frame (not used).
            """

            def terminate_process(process, timeout=5):
                """
                Helper function to terminate a process gracefully.
                Args:
                    process: The process to terminate.
                    timeout: The timeout for waiting for the process to terminate.
                """
                try:
                    process.terminate()
                    process.wait(timeout=timeout)
                except psutil.TimeoutExpired:
                    self.logger.debug(f"Timeout expired while waiting for process {process.pid} to terminate. Killing the process.")
                    process.kill()
                except psutil.NoSuchProcess:
                    self.logger.debug(f"Process {process.pid} does not exist. Skipping.")
                    pass

            if supernode_process.poll() is None:
                try:
                    main_subprocess = psutil.Process(supernode_process.pid)
                    client_app_processes = main_subprocess.children(recursive=True)
                    
                    for client_app_process in client_app_processes:
                        terminate_process(client_app_process)

                    terminate_process(main_subprocess)
                    self.logger.info("SuperNode process terminated.")

                except Exception as e:
                    self.logger.debug(f"Error during graceful shutdown: {e}")
                    # Gramine does not detect psutil.Process
                    # Give time for clientapp to stop then directly shutdown the supernode_process
                    time.sleep(10)
                    supernode_process.kill()
                        
                    self.logger.info("SuperNode process terminated.")
            else:
                self.logger.info("SuperNode process already terminated.")

            self.logger.info("Shutting down local gRPC server...")
            server.stop(0)
            self.logger.info("local gRPC server stopped.")
            termination_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        self.logger.info("Press CTRL+C to stop the server and SuperNode process.")
        
        try:
            while not termination_event.is_set():
                if self.shutdown_requested:
                    signal_handler(signal.SIGTERM, None)
                time.sleep(0.1)
        except KeyboardInterrupt:
            signal_handler(signal.SIGINT, None)