import grpc
from concurrent.futures import ThreadPoolExecutor
from flwr.proto import grpcadapter_pb2_grpc
from multiprocessing import cpu_count
from openfl.federated.task.runner import TaskRunner
from openfl.transport.grpc.connector.flower.local_grpc_server import LocalGRPCServer
import subprocess
from logging import getLogger
import signal
import threading
import psutil
import time

import os
os.environ["FLWR_HOME"] = os.path.join(os.getcwd(), "src/.flwr")

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
        self.shutdown_initiated = False  # Flag to ensure signal handler runs only once

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
        4. If auto_shutdown is enabled, monitors subprocess activity and initiates shutdown if no new subprocess starts within the expected time frame.

        Shutdown Process:
        - When a shutdown signal (SIGINT or SIGTERM) is received, the method will:
            1. Terminate all child processes of the supernode subprocess.
            2. Terminate the main supernode subprocess.
            3. Stop the gRPC server.
            4. Log the shutdown process and set the termination event to stop the server.
        """
        local_server_port = kwargs.get('local_server_port')

        server = grpc.server(ThreadPoolExecutor(max_workers=cpu_count()))
        grpcadapter_pb2_grpc.add_GrpcAdapterServicer_to_server(LocalGRPCServer(openfl_client, collaborator_name), server)
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
            Handles shutdown signals (SIGINT or SIGTERM) to terminate the supernode process and stop the local gRPC server.

            Args:
                _sig: The signal number.
                _frame: The current stack frame (not used).
            """
            if self.shutdown_initiated:
                return
            self.shutdown_initiated = True

            self.logger.info("Received shutdown signal. Terminating supernode process...")

            if supernode_process.poll() is None:
                try:
                    main_subprocess = psutil.Process(supernode_process.pid)
                    client_app_processes = main_subprocess.children(recursive=True)
                    for client_app_process in client_app_processes:
                        client_app_process.terminate()
                    _, still_alive = psutil.wait_procs(client_app_processes, timeout=1)
                    for p in still_alive:
                        p.kill()
                    supernode_process.terminate()
                    try:
                        supernode_process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        supernode_process.kill()
                    self.logger.info("Supernode process terminated.")
                except psutil.NoSuchProcess:
                    self.logger.info("Supernode process already terminated.")
            else:
                self.logger.info("Supernode process already terminated.")

            self.logger.info("Shutting down local gRPC server...")
            server.stop(0)
            self.logger.info("local gRPC server stopped.")
            termination_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        monitor_thread = None

        if self.auto_shutdown:
            self.logger.info("Automatic shutdown enabled. Monitoring subprocess activity...")

            def monitor_subprocesses():
                """
                Monitors the activity of subprocesses and initiates shutdown if no new subprocess starts within the expected time frame.
                """
                try:
                    main_subprocess = psutil.Process(supernode_process.pid)
                except psutil.NoSuchProcess:
                    return

                previous_end_time = None
                intervals = []

                while not termination_event.is_set():
                    client_app_processes = main_subprocess.children(recursive=True)
                    if client_app_processes:
                        for client_app_process in client_app_processes:
                            client_app_process.wait()
                            end_time = time.time()
                            if previous_end_time is not None:
                                interval = end_time - previous_end_time
                                intervals.append(interval)
                            previous_end_time = end_time

                    if previous_end_time is not None:
                        running_timer = time.time() - previous_end_time
                        if intervals:
                            average_interval = sum(intervals) / len(intervals)
                            if running_timer > 2 * average_interval:
                                self.logger.info("No new subprocess started within the expected time. Initiating shutdown...")
                                signal_handler(signal.SIGTERM, None)
                                return

                    time.sleep(1)

            monitor_thread = threading.Thread(target=monitor_subprocesses)
            monitor_thread.start()

        self.logger.info("Press CTRL+C to stop the server and supernode process.")
        
        try:
            while not termination_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            signal_handler(signal.SIGINT, None)

        if monitor_thread is not None:
            monitor_thread.join()

        self.logger.info("Exiting Task Runner")