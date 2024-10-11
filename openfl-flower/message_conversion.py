from flwr.proto import grpcadapter_pb2
from openfl.proto import openfl_pb2

def flower_to_openfl_message(flower_message):
    """Convert a Flower MessageContainer to an OpenFL OpenFLMessage."""
    if isinstance(flower_message, openfl_pb2.OpenFLMessage):
        # If the input is already an OpenFL message, return it as-is
        return flower_message
    else:
        openfl_message = openfl_pb2.OpenFLMessage()
        metadata = dict(flower_message.metadata)
        # Use the fully qualified name of the message class for message_type
        openfl_message.message_type = metadata.get('grpc-message-qualname')
        openfl_message.payload = flower_message.grpc_message_content
        # Copy metadata, excluding deprecated and empty fields
        for key, value in metadata.items():
            if key not in ['flower-version', ''] and value not in ['', None]:
                openfl_message.headers[key] = value
        return openfl_message

def openfl_to_flower_message(openfl_message):
    """Convert an OpenFL OpenFLMessage to a Flower MessageContainer."""
    if isinstance(openfl_message, grpcadapter_pb2.MessageContainer):
        # If the input is already a Flower message, return it as-is
        return openfl_message
    else:
        flower_message = grpcadapter_pb2.MessageContainer()
        # Use the message_type as the grpc_message_name
        flower_message.grpc_message_name = openfl_message.message_type
        flower_message.grpc_message_content = openfl_message.payload
        # Copy headers to metadata, excluding deprecated and empty fields
        for key, value in openfl_message.headers.items():
            if key not in ['flower-version', ''] and value not in ['', None]:
                flower_message.metadata[key] = value
        return flower_message