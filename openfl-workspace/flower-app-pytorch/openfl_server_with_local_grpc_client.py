import grpc
from concurrent import futures
from flwr.proto import grpcadapter_pb2_grpc
from openfl.protocols import aggregator_pb2, aggregator_pb2_grpc
from message_conversion import flower_to_openfl_message, openfl_to_flower_message

# from message_logger import log_message

class LocalGRPCClient:
    def __init__(self, superlink_address):
        self.superlink_channel = grpc.insecure_channel(superlink_address)
        self.superlink_stub = grpcadapter_pb2_grpc.GrpcAdapterStub(self.superlink_channel)

    def send_receive(self, openfl_message, client_id):
        # Convert OpenFL message to Flower message
        flower_message = openfl_to_flower_message(openfl_message)
        # log_message(flower_message, 'sent', client_id)
        # Send the Flower message to the Flower server and get a response
        flower_response = self.superlink_stub.SendReceive(flower_message)
        # log_message(flower_message, 'received', client_id)
        # Convert Flower response to OpenFL response
        print(f"Received message from Flower server, sending response back to OpenFL client: {flower_response.grpc_message_name}")
        openfl_response = flower_to_openfl_message(flower_response, sender='Flower Server', receiver='OpenFL Server')
        return openfl_response

class OpenFLServer(aggregator_pb2_grpc.AggregatorServicer):
    def __init__(self, local_grpc_client):
        self.local_grpc_client = local_grpc_client

    def PelicanDrop(self, request, context):
        """
        Args:
            request (aggregator_pb2.DropPod): The request
                from the collaborator.
            context (grpc.ServicerContext): The context of the request.

        Returns:
            aggregator_pb2.DropPod: The response to the
                request.
        """
        client_id = context.peer()
        # Forward the incoming OpenFL message to the local gRPC client
        print(f"Received message from OpenFL client, sending message to Flower server")#: {request.message_type}")
        openfl_response = self.local_grpc_client.send_receive(request, client_id)
        # print(f"Received message from Flower server, sending response back to OpenFL client: {openfl_response.message_type}")
        return openfl_response

def serve(superlink_address, openfl_server_port):
    # Start the local gRPC client
    local_grpc_client = LocalGRPCClient(superlink_address)

    # Start the OpenFL server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    aggregator_pb2_grpc.add_AggregatorServicer_to_server(OpenFLServer(local_grpc_client), server)
    server.add_insecure_port(f'[::]:{openfl_server_port}')
    server.start()
    print(f"OpenFL server started with local gRPC client. Listening on port {openfl_server_port}.")
    server.wait_for_termination()

if __name__ == '__main__':
    superlink_address = '127.0.0.1:9093'  # The Flower superlink address
    openfl_server_port = '9095'  # The port the OpenFL server will listen on
    serve(superlink_address, openfl_server_port)