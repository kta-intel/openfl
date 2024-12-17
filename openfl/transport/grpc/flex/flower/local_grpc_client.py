import grpc
from flwr.proto import grpcadapter_pb2_grpc
from openfl.transport.grpc.flex.flower.message_conversion import flower_to_openfl_message, openfl_to_flower_message

class LocalGRPCClient:
    """
    LocalGRPCClient facilitates communication between the Flower SuperLink
    and the OpenFL Server. It converts messages between OpenFL and Flower formats
    and handles the send-receive communication with the Flower SuperNode using gRPC.
    """
    def __init__(self, superlink_address):
        """
        Initialize.

        Args:
            superlink_address: The address the Flower SuperLink will listen on
        """
        self.superlink_channel = grpc.insecure_channel(superlink_address)
        self.superlink_stub = grpcadapter_pb2_grpc.GrpcAdapterStub(self.superlink_channel)

    def send_receive(self, openfl_message, header):
        """
        Sends a message to the Flower SuperLink and receives the response.

        Args:
            openfl_message: converted Flower SuperNode request sent by OpenFL server
            header: OpenFL header information to be included in the message.

        Returns:
            The response from the Flower SuperLink, converted back to OpenFL format.
        """
        # TODO: Add verification steps for messages coming from OpenFL transport
        flower_message = openfl_to_flower_message(openfl_message)
        flower_response = self.superlink_stub.SendReceive(flower_message)
        openfl_response = flower_to_openfl_message(flower_response, header=header)
        # TODO: Add verification steps for messages coming from Flower server
        return openfl_response
