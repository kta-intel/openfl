import threading
import queue
from flwr.proto import grpcadapter_pb2_grpc
from openfl.transport.grpc.connector.flower.message_conversion import flower_to_openfl_message, openfl_to_flower_message

class LocalGRPCServer(grpcadapter_pb2_grpc.GrpcAdapterServicer):
    """
    LocalGRPCServer is a gRPC server that handles requests from the Flower SuperNode
    and forwards them to the OpenFL Client. It uses a queue-based system to
    ensure that requests are processed sequentially, preventing concurrent
    request handling issues.
    """

    def __init__(self, openfl_client, collaborator_name):
        """
        Initialize.

        Args:
            openfl_client: An instance of the OpenFL Client.
            collaborator_name: The name of the collaborator.
        """
        self.openfl_client = openfl_client
        self.collaborator_name = collaborator_name
        self.request_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self.process_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def SendReceive(self, request, context):
        """
        Handles incoming gRPC requests by putting them into the request queue
        and waiting for the response.

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
            request = flower_to_openfl_message(request, header=None)

            # Send request to the OpenFL server
            openfl_response = self.openfl_client.send_message_to_server(request, self.collaborator_name)

            # Send response to Flower client
            flower_response = openfl_to_flower_message(openfl_response)
            response_queue.put(flower_response)
            self.request_queue.task_done()