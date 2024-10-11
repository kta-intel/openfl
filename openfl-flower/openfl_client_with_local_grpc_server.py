import grpc
from concurrent import futures
from flwr.proto import grpcadapter_pb2_grpc
from openfl.proto import openfl_pb2_grpc, openfl_pb2
from message_conversion import flower_to_openfl_message, openfl_to_flower_message

from message_logger import MessageLogger
message_logger = MessageLogger(log_file='OpenFL_Client_messages.log')

class OpenFLClient:
    def __init__(self, openfl_server_address):
        self.channel = grpc.insecure_channel(openfl_server_address)
        self.stub = openfl_pb2_grpc.FederatedServiceStub(self.channel)

    def send_message_to_server(self, flower_message):
        # Convert Flower message to OpenFL message
        metadata = dict(flower_message.metadata)
        message_logger.log_binary_message(
            metadata,
            metadata.get('grpc-message-module', ''),
            metadata.get('grpc-message-qualname', ''),
            flower_message.grpc_message_content,
            "Received from Flower Client"
        )
        openfl_message = flower_to_openfl_message(flower_message)
        message_logger.log_message(
            f"message_type: \"{openfl_message.message_type}\" "
            f"payload (Base64): \"{openfl_message.payload}\" "  # Assuming payload is binary and needs encoding
            f"headers: {dict(openfl_message.headers)}",
            "Sending to OpenFL Server"
        )

        # Send the OpenFL message to the OpenFL server and get a response
        openfl_response = self.stub.Exchange(openfl_message)
        # Convert the OpenFL response back to a Flower message
        flower_response = openfl_to_flower_message(openfl_response)
        return flower_response


class OpenFLLocalGRPCServer(grpcadapter_pb2_grpc.GrpcAdapterServicer):
    def __init__(self, openfl_client):
        self.openfl_client = openfl_client

    def SendReceive(self, request, context):
        # Received a message from the Flower client
        print(f"Received message from Flower client, sending to OpenFL server: {request.grpc_message_name}")
        # Forward the incoming message to the OpenFL client
        flower_response = self.openfl_client.send_message_to_server(request)
        # Sending the response back to the Flower client
        print(f"Received message from OpenFL server, sending response back to Flower client: {flower_response.grpc_message_name}")
        return flower_response

def serve(openfl_server_address, local_server_port):
    # Start the OpenFL client
    openfl_client = OpenFLClient(openfl_server_address)

    # Start the local gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    grpcadapter_pb2_grpc.add_GrpcAdapterServicer_to_server(OpenFLLocalGRPCServer(openfl_client), server)
    server.add_insecure_port(f'[::]:{local_server_port}')
    server.start()
    print(f"OpenFL local gRPC server started, listening on port {local_server_port}.")
    server.wait_for_termination()

if __name__ == '__main__':
    openfl_server_address = '127.0.0.1:9095'  # The OpenFL server's IP address and port
    local_server_port = '9092'  # The port the local gRPC server will listen on
    serve(openfl_server_address, local_server_port)