import grpc
from concurrent import futures
from flwr.proto import grpcadapter_pb2_grpc
from openfl.proto import openfl_pb2_grpc, openfl_pb2
from message_conversion import flower_to_openfl_message, openfl_to_flower_message

class OpenFLClient(openfl_pb2_grpc.FederatedServiceStub):
    def __init__(self, openfl_server_address):
        self.channel = grpc.insecure_channel(openfl_server_address)
        super().__init__(self.channel)

    def send_message_to_server(self, flower_message):
        # Convert Flower message to OpenFL message
        openfl_message = flower_to_openfl_message(flower_message)
        # Send the OpenFL message to the OpenFL server and get a response
        openfl_response = self.Exchange(openfl_message)
        # Convert the OpenFL response back to a Flower message
        flower_response = openfl_to_flower_message(openfl_response)
        return flower_response

class LocalGRPCServer(grpcadapter_pb2_grpc.GrpcAdapterServicer):
    def __init__(self, openfl_client):
        self.openfl_client = openfl_client

    def SendReceive(self, request, context):
        # Forward the incoming message to the OpenFL client
        print(f"Received message from Flower client-app: {request.grpc_message_name}")
        flower_response = self.openfl_client.send_message_to_server(request)
        print(f"Received response from OpenFL server: {flower_response.grpc_message_name}")
        return flower_response

def serve(openfl_server_address, local_server_port):
    # Start the OpenFL client
    openfl_client = OpenFLClient(openfl_server_address)

    # Start the local gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    grpcadapter_pb2_grpc.add_GrpcAdapterServicer_to_server(LocalGRPCServer(openfl_client), server)
    server.add_insecure_port(f'[::]:{local_server_port}')
    server.start()
    print(f"Client site OpenFL local gRPC server started with OpenFL client. Listening on port {local_server_port}.")
    server.wait_for_termination()

if __name__ == '__main__':
    openfl_server_address = '127.0.0.1:9095'  # Replace with the OpenFL server's IP address and port
    local_server_port = '9092'  # The port the local gRPC server will listen on
    serve(openfl_server_address, local_server_port)
