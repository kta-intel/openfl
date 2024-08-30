# local_grpc_client.py (Server)
import grpc
from concurrent import futures
from flwr.proto import grpcadapter_pb2_grpc

class LocalGRPCClient(grpcadapter_pb2_grpc.GrpcAdapterServicer):
    def __init__(self, superlink_address):
        self.superlink_address = superlink_address
        self.superlink_channel = grpc.insecure_channel(superlink_address)
        self.superlink_stub = grpcadapter_pb2_grpc.GrpcAdapterStub(self.superlink_channel)

    def SendReceive(self, request, context):
        # Forward the incoming message to the Flower superlink
        print(f"Received message from OpenFL local gRPC server at client site: {request.grpc_message_name}")
        response = self.superlink_stub.SendReceive(request)
        print(f"Received response from Flower superlink: {response.grpc_message_name}")
        return response

def serve(superlink_address, local_server_port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    grpcadapter_pb2_grpc.add_GrpcAdapterServicer_to_server(LocalGRPCClient(superlink_address), server)
    server.add_insecure_port(f'[::]:{local_server_port}')
    server.start()
    print(f"Server local gRPC client started. Listening for messages from client site on port {local_server_port}.")
    server.wait_for_termination()

if __name__ == '__main__':
    superlink_address = '127.0.0.1:9093'  # The Flower superlink address
    local_server_port = '9094'  # The port the local gRPC client will listen on
    serve(superlink_address, local_server_port)
