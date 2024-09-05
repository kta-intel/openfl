# # message_logger.py

# import logging
# from google.protobuf.json_format import MessageToJson

# # Import the protobuf classes for deserialization
# from flwr.proto import fleet_pb2, run_pb2
# # Add other necessary imports for protobuf classes

# def get_message_class_by_name(message_name):
#     message_classes = {
#         "CreateNodeRequest": fleet_pb2.CreateNodeRequest,
#         "CreateNodeResponse": fleet_pb2.CreateNodeResponse,
#         "PingRequest": fleet_pb2.PingRequest,
#         "PingResponse": fleet_pb2.PingResponse,
#         "PullTaskInsRequest": fleet_pb2.PullTaskInsRequest,
#         "PullTaskInsResponse": fleet_pb2.PullTaskInsResponse,
#         "PushTaskResRequest": fleet_pb2.PushTaskResRequest,
#         "PushTaskResResponse": fleet_pb2.PushTaskResResponse,
#         "GetRunRequest": run_pb2.GetRunRequest,
#         "GetRunResponse": run_pb2.GetRunResponse,
#         # Add other message mappings here
#     }
#     return message_classes.get(message_name)

# class MessageLogger:
#     def __init__(self, log_file):
#         self.log_file = log_file
#         logging.basicConfig(filename=self.log_file, level=logging.INFO, 
#                             format='%(asctime)s - %(levelname)s - %(message)s')

#     def log_message(self, message_name, message_content, direction, headers=None):
#         message_class = get_message_class_by_name(message_name)
#         if message_class:
#             message = message_class()
#             message.ParseFromString(message_content)
#             message_dict = {
#                 "message_type": message_name,
#                 "payload": MessageToJson(message),  # Serialize the message content to JSON
#                 "headers": headers or {}
#             }
#             # If the message has a 'metadata' field, we assume it's a Flower message
#             if hasattr(message, 'metadata'):
#                 message_dict = {
#                     "metadata": {md.key: md.value for md in message.metadata},
#                     "grpc_message_name": message_name,
#                     "grpc_message_content": MessageToJson(message)  # Serialize the message content to JSON
#                 }
#             logging.info(f"{direction} - {message_name}: {message_dict}")
#         else:
#             logging.warning(f"Unknown message name: {message_name}")

import logging

class MessageLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        # Configure logging to write to the specified log file
        logging.basicConfig(filename=self.log_file, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

    def log_message(self, message, direction):
        # Log the message with the specified direction
        logging.info(f"{direction} - {message}")