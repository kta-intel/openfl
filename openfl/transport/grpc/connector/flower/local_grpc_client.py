import grpc
import subprocess
import json
from flwr.proto import grpcadapter_pb2_grpc
from openfl.transport.grpc.connector.flower.message_conversion import flower_to_openfl_message, openfl_to_flower_message
from openfl.transport.grpc.connector.flower.deserialize_message import deserialize_flower_message

class LocalGRPCClient:
    """
    LocalGRPCClient facilitates communication between the Flower SuperLink
    and the OpenFL Server. It converts messages between OpenFL and Flower formats
    and handles the send-receive communication with the Flower SuperNode using gRPC.
    """
    def __init__(self, superlink_address, automatic_shutdown=False):
        """
        Initialize.

        Args:
            superlink_address: The address the Flower SuperLink will listen on
        """
        self.superlink_channel = grpc.insecure_channel(superlink_address)
        self.superlink_stub = grpcadapter_pb2_grpc.GrpcAdapterStub(self.superlink_channel)

        self.automatic_shutdown = automatic_shutdown
        self.end_experiment = False

        self.run_id = None
        self.flwr_ls_command = None

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
                self.round = message.metadata.group_id

        # # Check if clients completes the evaluation task for the final server round
        # if hasattr(deserialized_message, 'messages_list'):
        #     self.end_experiment = any(
        #         message.metadata.group_id == str(self.num_server_rounds) and message.metadata.message_type == "evaluate"
        #         for message in deserialized_message.messages_list
        #     )
        flower_response = self.superlink_stub.SendReceive(flower_message)

        if self.automatic_shutdown:
            self.end_experiment = self.monitor_server_app()
            print(self.end_experiment)

        openfl_response = flower_to_openfl_message(flower_response, header=header, end_experiment=self.end_experiment)
        return openfl_response
    
    def set_run_id(self, run_id, flwr_app_name):
        """
        Set the run ID for the Flower application and build the flwr_ls_command.

        Args:
            run_id: The run ID of the Flower application
            flwr_app_name: The name of the Flower application
        """
        self.run_id = run_id
        self.flwr_ls_command = ["flwr", "ls", f"./src/{flwr_app_name}", "--format", "json", "--run-id", str(self.run_id)]

    def monitor_server_app(self) -> bool:
        """
        Run the `flwr ls` command to monitor the Flower application.
        
        Returns:
            bool: True if the experiment has ended, False otherwise.
        """
        print(self.flwr_ls_command)
        flwr_ls_process = subprocess.run(self.flwr_ls_command, stdout=subprocess.PIPE, text=True)
        print(flwr_ls_process)
        print(flwr_ls_process.stdout)
        flwr_ls_output = json.loads(flwr_ls_process.stdout)

        for run in flwr_ls_output["runs"]:
            if "finished" in run["status"]:
                return True
        return False