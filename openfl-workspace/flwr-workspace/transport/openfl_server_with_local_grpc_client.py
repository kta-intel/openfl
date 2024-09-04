import grpc
from concurrent import futures
from flwr.proto import grpcadapter_pb2_grpc
from openfl.proto import openfl_pb2_grpc, openfl_pb2
from message_conversion import flower_to_openfl_message, openfl_to_flower_message

class LocalGRPCClient(grpcadapter_pb2_grpc.GrpcAdapterServicer):
    def __init__(self, superlink_address):
        self.superlink_address = superlink_address
        self.superlink_channel = grpc.insecure_channel(superlink_address)
        self.superlink_stub = grpcadapter_pb2_grpc.GrpcAdapterStub(self.superlink_channel)

    def SendReceive(self, request, context):
        # Convert Flower message to OpenFL message
        openfl_message = flower_to_openfl_message(request)
        # Forward the OpenFL message to the Flower superlink
        print(f"Received message from OpenFL server: {openfl_message.message_type}")
        flower_response = self.superlink_stub.SendReceive(request)
        # Convert Flower response to OpenFL response
        openfl_response = flower_to_openfl_message(flower_response)
        print(f"Received response from Flower superlink: {openfl_response.message_type}")
        return openfl_response

class OpenFLServer(openfl_pb2_grpc.FederatedServiceServicer):
    def __init__(self, local_grpc_client):
        self.local_grpc_client = local_grpc_client

    def Exchange(self, request, context):
        # Forward the incoming OpenFL message to the local gRPC client
        print(f"Received message from OpenFL client: {request.message_type}")
        flower_message = openfl_to_flower_message(request)
        flower_response = self.local_grpc_client.SendReceive(flower_message, context)
        # Convert the Flower response back to an OpenFL message
        openfl_response = flower_to_openfl_message(flower_response)
        print(f"Received response from local gRPC client: {openfl_response.message_type}")
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
