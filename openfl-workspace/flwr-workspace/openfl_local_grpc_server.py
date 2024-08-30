# openfl_local_grpc_server.py (Client Site)
import grpc
from concurrent import futures
from flwr.proto import grpcadapter_pb2_grpc

class OpenFLLocalGRPCServer(grpcadapter_pb2_grpc.GrpcAdapterServicer):
    def __init__(self, server_address):
        self.server_address = server_address
        self.channel = grpc.insecure_channel(server_address)
        self.stub = grpcadapter_pb2_grpc.GrpcAdapterStub(self.channel)

    def SendReceive(self, request, context):
        # Forward the incoming message to the OpenFL local gRPC client on the server
        print(f"Received message from Flower client-app: {request.grpc_message_name}")
        response = self.stub.SendReceive(request)
        print(f"Received response from OpenFL local gRPC client on the server: {response.grpc_message_name}")
        return response

def serve(client_server_address):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    grpcadapter_pb2_grpc.add_GrpcAdapterServicer_to_server(OpenFLLocalGRPCServer(client_server_address), server)
    server.add_insecure_port('[::]:9092')
    server.start()
    print("Client site OpenFL local gRPC server started. Listening on port 9092.")
    server.wait_for_termination()

if __name__ == '__main__':
    client_server_address = '127.0.0.1:9094'  # Replace with the server's IP address and port
    serve(client_server_address)
