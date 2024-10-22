from flwr.proto import grpcadapter_pb2
from openfl.protocols  import aggregator_pb2
# from deserialize_message import deserialize_flower_message

def flower_to_openfl_message(flower_message, header):
    """Convert a Flower MessageContainer to an OpenFL OpenFLMessage."""
    if isinstance(flower_message, aggregator_pb2.DropPod()):
        # If the input is already an OpenFL message, return it as-is
        return flower_message
    else:
        """Convert a Flower MessageContainer to an OpenFL message."""
        # Create the OpenFL message
        openfl_message = aggregator_pb2.DropPod()
        # Set the MessageHeader fields based on the provided sender and receiver
        openfl_message.header.CopyFrom(header)
        # openfl_message.message_type = flower_message.metadata['grpc-message-qualname']
        serialized_flower_message = flower_message.SerializeToString()
        openfl_message.message.npbytes = serialized_flower_message
        openfl_message.message.size = len(serialized_flower_message)

        return openfl_message

def openfl_to_flower_message(openfl_message):
    """Convert an OpenFL OpenFLMessage to a Flower MessageContainer."""
    if isinstance(openfl_message, grpcadapter_pb2.MessageContainer):
        # If the input is already a Flower message, return it as-is
        return openfl_message
    else:
    # Deserialize the Flower message from the DataStream npbytes field
        flower_message = grpcadapter_pb2.MessageContainer()
        flower_message.ParseFromString(openfl_message.message.npbytes)
        bytes_parsed = openfl_message.message.npbytes
        # import pdb; pdb.set_trace()
        return flower_message