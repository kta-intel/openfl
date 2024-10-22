import grpc
from concurrent.futures import ThreadPoolExecutor
from flwr.proto import grpcadapter_pb2_grpc
from multiprocessing import cpu_count
from openfl.protocols import aggregator_pb2, aggregator_pb2_grpc
from openfl.transport import AggregatorGRPCClient
from openfl.transport.grpc.fim.flower.message_conversion import flower_to_openfl_message, openfl_to_flower_message


class LocalGRPCServer(grpcadapter_pb2_grpc.GrpcAdapterServicer):
    def __init__(self, openfl_client, collaborator_name):
        self.openfl_client = openfl_client
        self.collaborator_name = collaborator_name

    def SendReceive(self, request, context):
        # print(f"Received message from Flower client, sending through OpenFL client to OpenFL server: {request.grpc_message_name}")
        # Forward the incoming message to the OpenFL client
        flower_response = self.openfl_client.send_message_to_server(request, self.collaborator_name)
        # Sending the response back to the Flower client
        # print(f"Received message from OpenFL server, sending response through OpenFL client back to Flower client: {flower_response.grpc_message_name}")
        return flower_response