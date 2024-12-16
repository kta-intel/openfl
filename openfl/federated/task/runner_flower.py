import grpc
from concurrent.futures import ThreadPoolExecutor
from flwr.proto import grpcadapter_pb2_grpc
from multiprocessing import cpu_count
from openfl.federated.task.runner import TaskRunner
from openfl.transport import AggregatorGRPCClient
from openfl.transport.grpc.fim.flower.local_grpc_server import LocalGRPCServer
import subprocess


class FlowerTaskRunner(TaskRunner):
    """
    FlowerTaskRunner is a task runner that executes Flower SuperNode
    to initialize the experiment from the client side
    """

    def __init__(self, **kwargs):
        """
        Initializes.

        Args:
            **kwargs: Additional parameters to pass to the functions.
        """
        super().__init__(**kwargs)
        self.num_partitions = self.data_loader.get_node_configs()[0]
        self.partition_id = self.data_loader.get_node_configs()[1]

        # Define a base port number
        base_port = 5000

        # Calculate the client port by adding the partition ID to the base port
        self.client_port = base_port + self.partition_id

    def start_client_adapter(self, openfl_client, collaborator_name, **kwargs):
        """
        Starts the local gRPC server and the Flower SuperNode.

        Args:
            openfl_client: The OpenFL client instance used to communicate with the OpenFL server.
            collaborator_name: The name of the collaborator.
            **kwargs: Additional parameters, including 'local_server_port'.
        """
        local_server_port = kwargs['local_server_port']
        # local_server_port = 9092 # note [kta-intel]: a direct connection to flower superlink

        # Start the local gRPC server
        server = grpc.server(ThreadPoolExecutor(max_workers=cpu_count()))
        grpcadapter_pb2_grpc.add_GrpcAdapterServicer_to_server(LocalGRPCServer(openfl_client, collaborator_name), server)
        
        # TODO: add restrictions
        server.add_insecure_port(f'[::]:{local_server_port}')
        server.start()
        print(f"OpenFL local gRPC server started, listening on port {local_server_port}.")

        # Start the Flower SuperNode in a subprocess
        command = [
            "flower-supernode",
            "--insecure",
            "--grpc-adapter",
            "--superlink", f"127.0.0.1:{local_server_port}", #  note [kta-intel]: this connects to local gRPC server
            # TODO: you must specify separate client ports when running multiple super nodes
            # on a single machine (i.e. a local poc). We need to add ability to automatically
            # set separate ports for each client if it is set as a local poc, otherwise it can be
            # whatever is automatically set by the system. Or we can add option to set port manually
            # or let it be automatically set
            "--clientappio-api-address", f"127.0.0.1:{self.client_port}",
            "--node-config", f"num-partitions={self.num_partitions} partition-id={self.partition_id}"
        ]
        # Start the subprocess
        supernode_process = subprocess.Popen(command, shell=False)

        server.wait_for_termination()

        supernode_process.terminate()
        supernode_process.wait()