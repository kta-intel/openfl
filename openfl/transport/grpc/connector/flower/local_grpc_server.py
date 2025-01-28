import threading
import queue
import grpc
from flwr.proto import grpcadapter_pb2_grpc
from openfl.transport.grpc.connector.flower.message_conversion import flower_to_openfl_message, openfl_to_flower_message
from openfl.transport.grpc.connector.flower.deserialize_message import deserialize_flower_message

class LocalGRPCServer(grpcadapter_pb2_grpc.GrpcAdapterServicer):
    """ 
    LocalGRPCServer is a gRPC server that handles requests from the Flower SuperNode
    and forwards them to the OpenFL Client. It uses a queue-based system to
    ensure that requests are processed sequentially, preventing concurrent
    request handling issues.
    """

    def __init__(self, openfl_client, collaborator_name, message_callback):
        """
        Initialize.

        Args:
            openfl_client: An instance of the OpenFL Client.
            collaborator_name: The name of the collaborator.
            message_callback: A callback function to be called when a specific message is received.
        """
        self.openfl_client = openfl_client
        self.collaborator_name = collaborator_name
        self.message_callback = message_callback
        self.request_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self.process_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        self.shutting_down = False  # Flag to indicate if the server is shutting down

    def SendReceive(self, request, context):
        """ Handles incoming gRPC requests by putting them into the request queue and waiting for the response.
        Args:
            request: The incoming gRPC request.
            context: The gRPC context.
        Returns:
            The response from the OpenFL server.
        """
        response_queue = queue.Queue()
        self.request_queue.put((request, response_queue))
        return response_queue.get()

    def process_queue(self):
        """
        Continuously processes requests from the request queue. Each request is
        sent to the OpenFL server, and the response is put into the corresponding
        response queue.
        """
        while True:
            request, response_queue = self.request_queue.get()
            deserialized_message = deserialize_flower_message(request)
            openfl_request = flower_to_openfl_message(request, header=None)

            # Send request to the OpenFL server
            openfl_response = self.openfl_client.send_message_to_server(openfl_request, self.collaborator_name)
            # Send response to Flower client
            flower_response = openfl_to_flower_message(openfl_response)
            # Check for the specific conditions
            if hasattr(deserialized_message, 'task_res_list'):
                for task_res in deserialized_message.task_res_list:
                    # TODO: this needs to be able to be set by the plan or the toml, not hard coded in the local grpc server
                    if task_res.group_id == "3" and task_res.task.task_type == "evaluate":
                        self.message_callback()
            response_queue.put(flower_response)
            self.request_queue.task_done()