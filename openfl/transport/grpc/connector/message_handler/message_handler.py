from abc import ABC, abstractmethod

class MessageHandler(ABC):
    def __init__(self, deserialized_message, origin=None):
        """
        Initialize the MessageHandler with a deserialized message.

        Args:
            deserialized_message: The deserialized message object.
        """
        self.deserialized_message = deserialized_message
        self.origin=origin

    @abstractmethod
    def extract_round_number(self):
        """Extract the round number from the deserialized message."""
        pass

    @abstractmethod
    def extract_model_dict(self):
        """Extract the model dictionary from the deserialized message."""
        pass

    @abstractmethod
    def extract_metrics(self):
        """Extract metrics from the deserialized message."""
        pass