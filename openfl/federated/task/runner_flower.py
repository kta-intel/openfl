import grpc
from concurrent.futures import ThreadPoolExecutor
from flwr.proto import grpcadapter_pb2_grpc
from multiprocessing import cpu_count
from openfl.federated.task.runner import TaskRunner
from openfl.transport.grpc.flex.flower.local_grpc_server import LocalGRPCServer
import subprocess
from logging import getLogger


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
        self.logger = getLogger(__name__)
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

        # Start the local gRPC server
        server = grpc.server(ThreadPoolExecutor(max_workers=cpu_count()))
        grpcadapter_pb2_grpc.add_GrpcAdapterServicer_to_server(LocalGRPCServer(openfl_client, collaborator_name), server)
        
        # TODO: add restrictions
        server.add_insecure_port(f'[::]:{local_server_port}')
        server.start()
        self.logger.info(f"OpenFL local gRPC server started, listening on port {local_server_port}.")

        server.stop(0)
        self.logger.info(f"OpenFL local gRPC server stopped.")

        # # Start the Flower SuperNode in a subprocess
        # command = [
        #     "flower-supernode",
        #     "--insecure",
        #     "--grpc-adapter",
        #     "--superlink", f"127.0.0.1:{local_server_port}", #  note [kta-intel]: this connects to local gRPC server
        #     "--clientappio-api-address", f"127.0.0.1:{self.client_port}",
        #     "--node-config", f"num-partitions={self.num_partitions} partition-id={self.partition_id}"
        # ]
        # # Start the subprocess
        # supernode_process = subprocess.Popen(command, shell=False)

        # import pdb; pdb.set_trace()
        # server.wait_for_termination()

        # supernode_process.terminate()
        # supernode_process.wait()
        # try:
        #     supernode_process.wait(timeout=5)
        # except subprocess.TimeoutExpired:
        #     supernode_process.kill()