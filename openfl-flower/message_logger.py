# import logging
# from deserialize_message import deserialize_flower_message

# # Configure logging
# logging.basicConfig(filename='flower_messages.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# def log_flower_message(flower_message, message_type):
#     """
#     Log a Flower message or response with deserialized content.

#     Args:
#         flower_message: The Flower message or response to be logged.
#         message_type: A string indicating the type of message ('sent' or 'received').
#     """
#     # Deserialize the grpc_message_content
#     deserialized_content = deserialize_flower_message(flower_message)

#     # Prepare the log entry
#     message_str = f"Flower message {message_type}:\n{flower_message}"
#     if deserialized_content is not None:
#         message_str += f"\nDeserialized content:\n{deserialized_content}"
#     else:
#         message_str += "\nDeserialization failed"

#     # Add separator
#     message_str += f"\n{'=' * 40}\n"

#     # Log the message with deserialized content and separator
#     logging.info(message_str)

# # This function can be used to log messages from other parts of your application
# def log_message(flower_message, message_type):
#     """
#     Public function to log a Flower message or response.

#     Args:
#         flower_message: The Flower message or response to be logged.
#         message_type: A string indicating the type of message ('sent' or 'received').
#     """
#     log_flower_message(flower_message, message_type)

import logging
import os
from deserialize_message import deserialize_flower_message

def get_logger(client_id):
    """
    Get a logger for a specific client ID.

    Args:
        client_id: A unique identifier for the client.

    Returns:
        A logging.Logger instance for the client.
    """
    # Create a directory for client logs if it doesn't exist
    log_dir = 'client_logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Configure logging for the client
    log_filename = os.path.join(log_dir, f'flower_messages_{client_id}.log')
    logger = logging.getLogger(f'client_{client_id}')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        # Add a file handler if it doesn't already exist
        file_handler = logging.FileHandler(log_filename)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

def log_flower_message(flower_message, message_type, client_id):
    """
    Log a Flower message or response with deserialized content for a specific client.

    Args:
        flower_message: The Flower message or response to be logged.
        message_type: A string indicating the type of message ('sent' or 'received').
        client_id: A unique identifier for the client.
    """
    # Deserialize the grpc_message_content
    deserialized_content = deserialize_flower_message(flower_message)

    # Prepare the log entry
    message_str = f"Flower message {message_type}:\n{flower_message}"
    if deserialized_content is not None:
        message_str += f"\nDeserialized content:\n{deserialized_content}"
    else:
        message_str += "\nDeserialization failed"

    # Add separator
    message_str += f"\n{'=' * 40}\n"

    # Get the logger for the client and log the message with deserialized content and separator
    logger = get_logger(client_id)
    logger.info(message_str)

# This function can be used to log messages from other parts of your application
def log_message(flower_message, message_type, client_id):
    """
    Public function to log a Flower message or response for a specific client.

    Args:
        flower_message: The Flower message or response to be logged.
        message_type: A string indicating the type of message ('sent' or 'received').
        client_id: A unique identifier for the client.
    """
    log_flower_message(flower_message, message_type, client_id)