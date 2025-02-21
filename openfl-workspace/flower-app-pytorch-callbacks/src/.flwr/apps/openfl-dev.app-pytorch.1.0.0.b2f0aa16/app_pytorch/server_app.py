"""app-pytorch: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from app_pytorch.task import Net, get_weights


####################################################################################
# TODO: Consider moving this to a separate file and importing SaveModelStrategy

from openfl.protocols import utils
from openfl.pipelines import NoCompressionPipeline
def save_model(tensor_dict, round_number, file_path):
    model = utils.construct_model_proto(
                tensor_dict, round_number, NoCompressionPipeline()
            )
    utils.dump_proto(model, file_path)


# from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, Scalar, Parameters, parameters_to_ndarrays, Metrics
from typing import Optional, Union, OrderedDict, List, Tuple
import numpy as np

net = Net()

class SaveModelStrategy(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.largest_loss = 1e9

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `list[np.ndarray]`
            aggregated_ndarrays: list[np.ndarray] = parameters_to_ndarrays(
                aggregated_parameters
            )


            params_dict =  OrderedDict(zip(net.state_dict().keys(), aggregated_ndarrays))

            # # Save the model to disk
            # save_model(params_dict, server_round, './save/last.pbuf')

            # if aggregated_metrics["train_loss"] < self.largest_loss:
            #     self.largest_loss = aggregated_metrics["train_loss"]
            #     save_model(params_dict, server_round, './save/best.pbuf')

        return aggregated_parameters, aggregated_metrics


# def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
#     # Multiply accuracy of each client by number of examples used

#     print(metrics)
#     losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
#     examples = [num_examples for num_examples, _ in metrics]

#     # Aggregate and return custom metric (weighted average)
#     return {"train_loss": sum(losses) / sum(examples)}

##################################################################################### 


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = SaveModelStrategy(
        # fit_metrics_aggregation_fn=weighted_average,
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
