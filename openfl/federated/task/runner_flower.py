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
        # import pdb; pdb.set_trace()
        command = [
            "flower-supernode",
            kwargs.get('app_path', './app-pytorch'),
            "--insecure",
            "--grpc-adapter",
            "--superlink", f"127.0.0.1:{local_server_port}",
            "--node-config", f"num-partitions={kwargs.get('num_partitions', 1)} partition-id={kwargs.get('partition_id', 0)}"
        ]
        # Start the subprocess
        supernode_process = subprocess.Popen(command, shell=False)

        server.wait_for_termination()

        supernode_process.terminate()
        supernode_process.wait()