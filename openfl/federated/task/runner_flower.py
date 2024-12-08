import grpc
from concurrent.futures import ThreadPoolExecutor
from flwr.proto import grpcadapter_pb2_grpc
from multiprocessing import cpu_count
from openfl.federated.task.runner import TaskRunner
from openfl.transport import AggregatorGRPCClient
from openfl.transport.grpc.fim.flower.local_grpc_server import LocalGRPCServer
import subprocess


class FlowerTaskRunner(TaskRunner):
    def __init__(self, **kwargs):
        """Initializes the FlowerTaskRunner object.

        Args:
            **kwargs: Additional parameters to pass to the functions.
        """
        super().__init__(**kwargs)
        self.num_partitions = self.data_loader.get_node_configs()[0]
        self.partition_id = self.data_loader.get_node_configs()[1]
   
    def start_client_adapter(self, openfl_client, collaborator_name, **kwargs):
        local_server_port = kwargs['local_server_port']

        # Start the local gRPC server
        server = grpc.server(ThreadPoolExecutor(max_workers=cpu_count()))
        grpcadapter_pb2_grpc.add_GrpcAdapterServicer_to_server(LocalGRPCServer(openfl_client, collaborator_name), server)
        
        # TODO: add restrictions
        server.add_insecure_port(f'[::]:{local_server_port}')
        server.start()
        print(f"OpenFL local gRPC server started, listening on port {local_server_port}.")

        # Start the Flower supernode in a subprocess
        command = [
            "flower-supernode",
            "--insecure",
            "--grpc-adapter",
            "--superlink", f"127.0.0.1:{local_server_port}", # This should connect to local gRPC server
            # TODO: you must specify separate client ports when running multiple super nodes
            # on a single machine (i.e. a local poc). We need to add ability to automatically
            # set separate ports for each client if it is set as a local poc, otherwise it can be
            # whatever is automatically set by the system. Or we can add option to set port manually
            # or let it be automatically set
            # TODO: temporarilty add client port to a collaborator unique yaml (i.e. data)
            "--clientappio-api-address", f"127.0.0.1:{self.client_port}",
            "--node-config", f"num-partitions={self.num_partitions} partition-id={self.partition_id}"
        ]
        # Start the subprocess
        supernode_process = subprocess.Popen(command, shell=False)

        server.wait_for_termination()

        supernode_process.terminate()
        supernode_process.wait()