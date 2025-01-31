import importlib
# import os
from google.protobuf.message import DecodeError

# def get_next_log_filename(log_dir='./logs'):
#     """
#     Get the next log filename based on the existing log files in the directory.

#     Args:
#         log_dir: The directory where log files are stored.

#     Returns:
#         The next log filename.
#     """
#     if not os.path.exists(log_dir):
#         os.makedirs(log_dir)

#     existing_logs = [f for f in os.listdir(log_dir) if f.endswith('.log')]
#     if not existing_logs:
#         return os.path.join(log_dir, '1.log')

#     existing_numbers = [int(f.split('.')[0]) for f in existing_logs]
#     next_number = max(existing_numbers) + 1
#     return os.path.join(log_dir, f'{next_number}.log')

# def save_message_to_log(message, log_filename):
#     """
#     Save the message to a log file.

#     Args:
#         message: The message object to save.
#         log_filename: The filename of the log file.
#     """
#     with open(log_filename, 'w') as log_file:
#         log_file.write(str(message))

def deserialize_flower_message(flower_message):
    """
    Deserialize the grpc_message_content of a Flower message using the module and class name
    specified in the metadata.

    Args:
        flower_message: The Flower message containing the metadata and binary content.

    Returns:
        The deserialized message object, or None if deserialization fails.
    """
    # Access metadata directly
    metadata = flower_message.metadata
    module_name = metadata.get('grpc-message-module')
    qualname = metadata.get('grpc-message-qualname')

    # Import the module
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        print(f"Failed to import module: {module_name}. Error: {e}")
        return None

    # Get the message class
    try:
        message_class = getattr(module, qualname)
    except AttributeError as e:
        print(f"Failed to get message class '{qualname}' from module '{module_name}'. Error: {e}")
        return None

    # Deserialize the content
    try:
        message = message_class.FromString(flower_message.grpc_message_content)
    except DecodeError as e:
        print(f"Failed to deserialize message content. Error: {e}")
        return None

    # Save the message to a log file
    # log_filename = get_next_log_filename()
    # save_message_to_log(message, log_filename)
    # print(f"Message saved to {log_filename}")
    return message