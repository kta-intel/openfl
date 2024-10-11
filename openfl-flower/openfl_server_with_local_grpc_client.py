import grpc
from concurrent import futures
from flwr.proto import grpcadapter_pb2_grpc
from openfl.proto import openfl_pb2_grpc, openfl_pb2
from message_conversion import flower_to_openfl_message, openfl_to_flower_message

from message_logger import MessageLogger
message_logger = MessageLogger(log_file='OpenFL_Server_messages.log')

class LocalGRPCClient:
    def __init__(self, superlink_address):
        self.superlink_channel = grpc.insecure_channel(superlink_address)
        self.superlink_stub = grpcadapter_pb2_grpc.GrpcAdapterStub(self.superlink_channel)

    def send_receive(self, openfl_message):
        # Convert OpenFL message to Flower message
        flower_message = openfl_to_flower_message(openfl_message)
        # Send the Flower message to the Flower server and get a response
        flower_response = self.superlink_stub.SendReceive(flower_message)
        metadata = dict(flower_response.metadata)
        # Log the message using the new metadata fields
        message_logger.log_message(
            f"metadata: {metadata} "
            f"grpc_message_module: \"{metadata.get('grpc-message-module')}\" "
            f"grpc_message_qualname: \"{metadata.get('grpc-message-qualname')}\" "
            f"grpc_message_content: \"{flower_response.grpc_message_content}\"",
            "Received from Flower Server"
        )
        # Convert Flower response to OpenFL response
        openfl_response = flower_to_openfl_message(flower_response)
        return openfl_response

class OpenFLServer(openfl_pb2_grpc.FederatedServiceServicer):
    def __init__(self, local_grpc_client):
        self.local_grpc_client = local_grpc_client

    def Exchange(self, request, context):
        # Forward the incoming OpenFL message to the local gRPC client
        print(f"Received message from OpenFL client, sending message to Flower server: {request.message_type}")
        openfl_response = self.local_grpc_client.send_receive(request)
        # Log the OpenFL response message
        message_logger.log_message(
            f"message_type: \"{openfl_response.message_type}\" "
            f"payload: \"{openfl_response.payload}\" "
            f"headers: {dict(openfl_response.headers)}",
            "Sending to OpenFL Client"
        )
        print(f"Received message from Flower server, sending response back to OpenFL client: {openfl_response.message_type}")
        return openfl_response

def serve(superlink_address, openfl_server_port):
    # Start the local gRPC client
    local_grpc_client = LocalGRPCClient(superlink_address)

    # Start the OpenFL server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    openfl_pb2_grpc.add_FederatedServiceServicer_to_server(OpenFLServer(local_grpc_client), server)
    server.add_insecure_port(f'[::]:{openfl_server_port}')
    server.start()
    print(f"OpenFL server started with local gRPC client. Listening on port {openfl_server_port}.")
    server.wait_for_termination()

if __name__ == '__main__':
    superlink_address = '127.0.0.1:9093'  # The Flower superlink address
    openfl_server_port = '9095'  # The port the OpenFL server will listen on
    serve(superlink_address, openfl_server_port)