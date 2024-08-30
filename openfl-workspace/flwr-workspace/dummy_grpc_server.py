import grpc
from concurrent import futures
from flwr.proto import grpcadapter_pb2_grpc

class DummyGRPCServer(grpcadapter_pb2_grpc.GrpcAdapterServicer):
    def SendReceive(self, request, context):
        # Echo the incoming message back to the client
        print(f"Dummy server received message: {request.grpc_message_name}")
        return request

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    grpcadapter_pb2_grpc.add_GrpcAdapterServicer_to_server(DummyGRPCServer(), server)
    server.add_insecure_port('[::]:9093')  # Use a different port for the dummy server
    server.start()
    print("Dummy server started. Listening on port 9093.")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
