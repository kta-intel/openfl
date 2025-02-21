import threading
import grpc
from concurrent.futures import ThreadPoolExecutor
from flwr.proto import grpcadapter_pb2_grpc
from multiprocessing import cpu_count
from openfl.federated.task.runner import TaskRunner
import subprocess
from logging import getLogger
import signal
import psutil
import time
import os
import numpy as np
from pathlib import Path

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
    def __init__(self, **kwargs):
        """
        Initializes the FlowerTaskRunner.

        Args:
            auto_shutdown (bool): Whether to enable automatic shutdown based on subprocess activity.
                Default is True. Set to False for long-lived components.
            **kwargs: Additional parameters to pass to the functions.
        """
        super().__init__(**kwargs)
        self.model = None
        self.logger = getLogger(__name__)
        self.num_partitions = self.data_loader.get_node_configs()[0]
        self.partition_id = self.data_loader.get_node_configs()[1]

        base_port = 5000
        # Only necessary to local runs in order to avoid port conflicts
        self.client_port = base_port + self.partition_id

        self.patch = kwargs.get('patch')
        self.shutdown_requested = False # Flag signal shutdown

    def start_client_adapter(self, local_grpc_server, **kwargs):
        """
        Starts the local gRPC server and the Flower SuperNode.
        """
        local_server_port = kwargs.get('local_server_port')

        # Only necessary to local runs in order to avoid port conflicts
        local_server_port = local_server_port - self.partition_id

        def message_callback():
            self.shutdown_requested = True

        # TODO: Can we isolate the local_grpc_server from the task runner?
        local_grpc_server.set_end_experiment_callback(message_callback)
        local_grpc_server.start_server(local_server_port)

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
        local_grpc_server.handle_signals(supernode_process)

        self.logger.info("Press CTRL+C to stop the server and SuperNode process.")
        
        try:
            while not local_grpc_server.termination_event.is_set():
                if self.shutdown_requested:
                    local_grpc_server.terminate_supernode_process(supernode_process)
                    local_grpc_server.stop_server()
                time.sleep(0.1)
        except KeyboardInterrupt:
            local_grpc_server.terminate_supernode_process(supernode_process)
            local_grpc_server.stop_server()

    def set_tensor_dict(self, tensor_dict, with_opt_vars=False):
        """Set the tensor dictionary.
        To be framework agnostic, this method will not attempt to load the weights into the model
        and save out the native format. Instead, it will load and save the dictionary directly

        Args:
            tensor_dict (dict): The tensor dictionary.
            with_opt_vars (bool): This argument is inherited from the parent class
                but is not used in the FlowerTaskRunner.
        """
        self.tensor_dict = tensor_dict

    def save_native(
        self,
        filepath,
        **kwargs,
    ):
        """
        Save model weights in a .npz file specified by the filepath.
        The model weights are stored as a dictionary of np.ndarray

        Args:
            filepath (str): Path to the .npz file to be created by np.savez().
            **kwargs: Additional parameters (currently not used).

        Returns:
            None

        Raises:
            AssertionError: If the file extension is not '.npz'.
        """
        # Ensure the file extension is .npz
        if isinstance(filepath, Path):
            filepath = str(filepath)

        # Ensure the file extension is .npz
        assert filepath.endswith('.npz'), "Currently, only '.npz' file type is supported."

        # Save the tensor dictionary to a .npz file
        np.savez(filepath, **self.tensor_dict)
