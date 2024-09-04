from flwr.proto import grpcadapter_pb2
from openfl.proto import openfl_pb2

def flower_to_openfl_message(flower_message):
    """Convert a Flower MessageContainer to an OpenFL OpenFLMessage."""
    if isinstance(flower_message, openfl_pb2.OpenFLMessage):
        # If the input is already an OpenFL message, return it as-is
        return flower_message
    else:
        openfl_message = openfl_pb2.OpenFLMessage()
        openfl_message.message_type = flower_message.grpc_message_name
        openfl_message.payload = flower_message.grpc_message_content
        openfl_message.headers.update(flower_message.metadata)
        return openfl_message

def openfl_to_flower_message(openfl_message):
    """Convert an OpenFL OpenFLMessage to a Flower MessageContainer."""
    if isinstance(openfl_message, grpcadapter_pb2.MessageContainer):
        # If the input is already a Flower message, return it as-is
        return openfl_message
    else:
        flower_message = grpcadapter_pb2.MessageContainer()
        flower_message.grpc_message_name = openfl_message.message_type
        flower_message.grpc_message_content = openfl_message.payload
        flower_message.metadata.update(openfl_message.headers)
        return flower_message