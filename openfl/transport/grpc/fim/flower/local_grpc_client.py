import grpc
from flwr.proto import grpcadapter_pb2_grpc
from openfl.transport.grpc.fim.flower.message_conversion import flower_to_openfl_message, openfl_to_flower_message

class LocalGRPCClient:
    def __init__(self, superlink_address):
        self.superlink_channel = grpc.insecure_channel(superlink_address)
        self.superlink_stub = grpcadapter_pb2_grpc.GrpcAdapterStub(self.superlink_channel)

    def send_receive(self, openfl_message, header):
        # TODO: verification step for messages coming from Flower server
        collaborator_name = openfl_message.header.sender

        flower_message = openfl_to_flower_message(openfl_message)
        flower_response = self.superlink_stub.SendReceive(flower_message)
        # print(f"Received message from Flower server, sending response through OpenFL server back to OpenFL client: {flower_response.grpc_message_name}")
        openfl_response = flower_to_openfl_message(flower_response, header=header)
        return openfl_response
