import grpc
import subprocess
import json
from flwr.proto import grpcadapter_pb2_grpc
from openfl.transport.grpc.connector.flower.message_conversion import flower_to_openfl_message, openfl_to_flower_message
from openfl.transport.grpc.connector.flower.deserialize_message import deserialize_flower_message
from logging import getLogger

class LocalGRPCClient:
    """
    LocalGRPCClient facilitates communication between the Flower SuperLink
    and the OpenFL Server. It converts messages between OpenFL and Flower formats
    and handles the send-receive communication with the Flower SuperNode using gRPC.
    """
    def __init__(self, superlink_address, automatic_shutdown=False, is_flwr_serverapp_running_callback=None):
        """
        Initialize.

        Args:
            superlink_address: The address the Flower SuperLink will listen on
        """
        self.superlink_channel = grpc.insecure_channel(superlink_address)
        self.superlink_stub = grpcadapter_pb2_grpc.GrpcAdapterStub(self.superlink_channel)

        self.automatic_shutdown = automatic_shutdown
        self.end_experiment = False
        self.is_flwr_serverapp_running_callback = is_flwr_serverapp_running_callback
        self.round_number = 0

        self.logger = getLogger(__name__)

    def send_receive(self, openfl_message, header):
        """
        Sends a message to the Flower SuperLink and receives the response.

        Args:
            openfl_message: converted Flower SuperNode request sent by OpenFL server
            header: OpenFL header information to be included in the message.

        Returns:
            The response from the Flower SuperLink, converted back to OpenFL format.
        """
        flower_message = openfl_to_flower_message(openfl_message)

        deserialized_message = deserialize_flower_message(flower_message)
        if hasattr(deserialized_message, 'messages_list'):
            for message in deserialized_message.messages_list:
                self.round_number = message.metadata.group_id

        flower_response = self.superlink_stub.SendReceive(flower_message)

        if self.automatic_shutdown:
            self.end_experiment = not self.is_flwr_serverapp_running_callback()

        openfl_response = flower_to_openfl_message(flower_response, header=header, end_experiment=self.end_experiment)
        return openfl_response
