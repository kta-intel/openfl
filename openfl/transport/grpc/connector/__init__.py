from importlib import util

from openfl.transport.grpc.connector.utils import get_local_grpc_server
from openfl.transport.grpc.connector.message_handler.message_handler import MessageHandler

if util.find_spec("flwr") is not None:
    from openfl.transport.grpc.connector.message_handler.message_handler_flower import MessageHandlerFlower


# __all__ = ['get_local_grpc_server']