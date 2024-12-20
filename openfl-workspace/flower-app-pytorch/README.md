# Open(FL)ower

This workspace demonstrates a new functionality in OpenFL to interoperate with [Flower](https://flower.ai/). In particular, a user can now use the Flower API to run on an OpenFL infrastructure. OpenFL will act as an intermediary step between the Flower SuperLink and Flower SuperNode to relay messages across the network using OpenFL's transport mechanisms while Flower manages the experiment.

## Overview

In this repository, you'll notice a directory called `./app-pytorch`. This is effectively a Flower PyTorch app created using Flower's `flwr new` command that has been modified to run a local federation. The client and server apps dictate what will be run by the client and server respectively. `Task.py` defines the logic that will be executed by each app, such as the model definition, train/test tasks, etc.

## Execution Methods

There are two ways to execute this:

1. Run `flwr run` as a sub-process of the aggregator alongside the superlink. (default)
2. Run `flwr run` as a [separate process](#invoke-flower-experiment-as-a-separate-command)  after initializing the `SuperLink` and `SuperNode` at the aggregator and collaborators respectively.

In addition, there are options to run the `SuperLink` and `SuperNode` as [long-lived components](#long-lived-superlink-and-supernode) that will indefinitely wait for new runs or, by default, as a short-lived component (similar to OpenFL's task runner) that terminates at the end of the experiment.

## Getting Started

### Create a Workspace

Start by creating a workspace:

```sh
fx workspace create --template flower-app-pytorch --prefix my_workspace
cd my_workspace
```

This will create a workspace in your current working directory called `./my_workspace` as well as install the Flower app defined in `./app-pytorch.` This will be where the experiment takes place.

### Configure the Experiment
Notice under `./plan`, you will find the familiar OpenFL YAML files to configure the experiment. `col.yaml` and `data.yaml` will be populated by the collaborators that will run the Flower client app and the respective data shard or directory they will perform their training and testing on.
plan.yaml configures the experiment itself. The Open-Flower integration makes a few key changes to the `plan.yaml`:

1. Introduction of a new top-level key (`flex`) to configure a newly introduced component called "FLEX (Federated Learning EXchange)". Specifically, the Flower integration uses a `FLEX` subclass called `FLEXFlower`. This component is run by the aggregator and is responsible for initializing the Flower SuperLink and connecting to the OpenFL server. The superlink parameters can be configured using `flex.settings.superlink_params`. If nothing is supplied, it will simply run `flower-superlink --insecure` with the command's default settings as dictated by Flower. It also includes the option to run the flwr run command via `flex.settings.flwr_run_params`. Without setting these commands, the aggregator will not invoke `flwr run` and it will be up to the user to run this process separately to start a Flower experiment.

```yaml
flex:
  defaults: plan/defaults/flex.yaml
  template: openfl.component.FLEXFlower
  settings:
    superlink_params:
      insecure: True
      serverappio-api-address: 127.0.0.1:9091
      fleet-api-address: 127.0.0.1:9092
      exec-api-address: 127.0.0.1:9093
    flwr_run_params:
      flwr_app_name: "app-pytorch"
      federation_name: "local-poc"
```

2. `FLEXAssigner` and tasks designed to explicitly run `start_client_adapter` task for every authorized collaborator, which is defined by the Task Runner.

```yaml
assigner:
  defaults: plan/defaults/assigner.yaml
  template: openfl.component.FLEXAssigner
  settings:
    task_groups:
      - name: FLEX_Flower
        tasks:
          - start_client_adapter
```

3. `FlowerTaskRunner` which will execute the `start_client_adapter` task. This task starts the Flower SuperNode and makes a connection to the OpenFL client. Additionally, the `FlowerTaskRunner` has an additional setting `FlowerTaskRunner.settings.auto_shutdown` which is default set to `True`. When set to `True`, the task runner will shut the SuperNode at the completion of an experiment, otherwise, it will run continuously.

```yaml
task_runner:
  defaults: plan/defaults/task_runner.yaml
  template: openfl.federated.task.runner_flower.FlowerTaskRunner
  settings:
    auto_shutdown: True
```
3. `FlowerDataLoader` with similar high-level functionality to other dataloaders.

**IMPORTANT NOTE**: `aggregator.settings.rounds_to_train` is set to 1. __Do not edit this__. The actual number of rounds for the experiment is controlled by Flower logic inside of `./app-pytorch/pyproject.toml`. The entirety of the Flower experiment will run in a single OpenFL round. The aggregator round is there to stop the OpenFL components at the completion of the experiment.

## Running the Workspace
Run the workspace as normal (certify the workspace, initialize the plan, register the collaborators, etc.):

```SH
# Generate a Certificate Signing Request (CSR) for the Aggregator
fx aggregator generate-cert-request

# The CA signs the aggregator's request, which is now available in the workspace
fx aggregator certify --silent

# Initialize FL Plan and Model Weights for the Federation
fx plan initialize

################################
# Setup Collaborator 1 
################################

# Create a collaborator named "collaborator1" that will use shard "0"
fx collaborator create -n collaborator1 -d 0

# Generate a CSR for collaborator1
fx collaborator generate-cert-request -n collaborator1

# The CA signs collaborator1's certificate
fx collaborator certify -n collaborator1 --silent

################################
# Setup Collaborator 2 
################################

# Create a collaborator named "collaborator2" that will use shard "1"
fx collaborator create -n collaborator2 -d 1

# Generate a CSR for collaborator2
fx collaborator generate-cert-request -n collaborator2

# The CA signs collaborator2's certificate
fx collaborator certify -n collaborator2 --silent

##############################
# Start to Run the Federation
##############################

# Run the Aggregator
fx aggregator start
```

This will prepare the workspace and start the OpenFL aggregator, Flower superlink, and Flower serverapp. You should see something like:

```SH
INFO     🧿 Starting the Aggregator Service.                                                                                         aggregator.py:70
INFO     Building `openfl.component.FLEXAssigner` Module.                                                                                 plan.py:226
INFO     Building `openfl.pipelines.NoCompressionPipeline` Module.                                                                        plan.py:226
INFO     Building `openfl.component.straggler_handling_functions.CutoffTimeBasedStragglerHandling` Module.                                plan.py:226
WARNING  CutoffTimeBasedStragglerHandling is disabled as straggler_cutoff_time is set to np.inf.           cutoff_time_based_straggler_handling.py:46
INFO     Building `openfl.component.FLEXFlower` Module.                                                                                   plan.py:226
INFO     Building `openfl.component.Aggregator` Module.                                                                                   plan.py:226
use_tls=True
INFO     [FLEX] Starting server process: flower-superlink --fleet-api-type grpc-adapter --insecure --serverappio-api-address               flex.py:28
         127.0.0.1:9091 --fleet-api-address 127.0.0.1:9092 --exec-api-address 127.0.0.1:9093                                                         
INFO     [FLEX] server process started with PID: 1972825                                                                                   flex.py:30
INFO     Starting Aggregator gRPC Server                                                                                     aggregator_server.py:389
INFO :      Starting Flower SuperLink
WARNING :   Option `--insecure` was set. Starting insecure HTTP server.
INFO :      Flower Deployment Engine: Starting Exec API on 127.0.0.1:9093
INFO :      Flower ECE: Starting ServerAppIo API (gRPC-rere) on 127.0.0.1:9091
INFO :      Flower ECE: Starting Fleet API (GrpcAdapter) on 127.0.0.1:9092
```

### Start Collaborators
Open 2 additional terminals for collaborators.
For collaborator 1's terminal, run:
```SH
fx collaborator start -n collaborator1
```
For collaborator 2's terminal, run:
```SH
fx collaborator start -n collaborator2
```
This will start the collaborator nodes, the Flower `SuperNode`, and Flower `ClientApp`, and begin running the Flower experiment. You should see something like:

```SH
INFO     🧿 Starting a Collaborator Service.                                                                                       collaborator.py:85
INFO     Building `openfl.federated.data.loader_flower.FlowerDataLoader` Module.                                                          plan.py:226
INFO     Building `openfl.federated.task.runner_flower.FlowerTaskRunner` Module.                                                          plan.py:226
INFO     Building `openfl.pipelines.NoCompressionPipeline` Module.                                                                        plan.py:226
INFO     Building `openfl.component.Collaborator` Module.                                                                                 plan.py:226
INFO     Waiting for tasks...                                                                                                     collaborator.py:222
INFO     Received the following tasks: [name: "start_client_adapter"                                                              collaborator.py:172
         ]                                                                                                                                           
INFO     OpenFL local gRPC server started, listening on port 9090.                                                                runner_flower.py:61
INFO     Automatic shutdown enabled. Monitoring subprocess activity...                                                           runner_flower.py:105
INFO     Press CTRL+C to stop the server and supernode process.                                                                  runner_flower.py:139
INFO :      Starting Flower SuperNode
WARNING :   Option `--insecure` was set. Starting insecure HTTP channel to 127.0.0.1:9090.
INFO :      Starting Flower ClientAppIo gRPC server on 127.0.0.1:5000
```
### Completion of the Experiment
Upon the completion of the experiment, on the `aggregator` terminal, the Flower components should send an experiment summary as the `SuperLink `continues to receive requests from the supernode:
```SH
INFO :      [SUMMARY]
INFO :      Run finished 3 round(s) in 93.29s
INFO :          History (loss, distributed):
INFO :                  round 1: 2.0937052175497555
INFO :                  round 2: 1.8027011854633406
INFO :                  round 3: 1.6812996898487116
INFO :      GrpcAdapter.PullTaskIns
INFO :      GrpcAdapter.PullTaskIns
INFO :      GrpcAdapter.PullTaskIns
```
If `autoshutdown` is enabled, this will be shortly followed by the OpenFL `aggregator` receiving "results" from the `collaborator` and subsequently shutting down:

```SH
INFO     Collaborator collaborator1 is sending task results for start_client_adapter, round 0                                       aggregator.py:633
INFO     Round: 0, Collaborators that have completed all tasks: ['collaborator1']                                                  aggregator.py:1095
INFO :      GrpcAdapter.DeleteNode
INFO     Collaborator collaborator2 is sending task results for start_client_adapter, round 0                                       aggregator.py:633
INFO     Round: 0, Collaborators that have completed all tasks: ['collaborator1', 'collaborator2']                                 aggregator.py:1095
INFO     Experiment Completed. Cleaning up...                                                                                      aggregator.py:1053
INFO     Sending signal to collaborator collaborator2 to shutdown...                                                                aggregator.py:360
INFO     Sending signal to collaborator collaborator1 to shutdown...                                                                aggregator.py:360
INFO     [FLEX] Stopping server process with PID: 1963348...                                                                               flex.py:39
INFO     [FLEX] Stopping server subprocess  with PID: 1964099...                                                                           flex.py:44
INFO     [FLEX] Server process stopped.  
```    
Upon the completion of the experiment, on the `collaborator` terminals, the Flower components should be outputting the information about the run:

```SH
INFO :      [RUN ..., ROUND 3]
INFO :      Received: evaluate message 53e1ad1c-ffeb-41cc-9857-3d1b83273bd9
INFO :      Starting Flower ClientApp
INFO :      Pulling ClientAppInputs for token ...
INFO :      Pushing ClientAppOutputs for token ...
```

If `autoshutdown` is enabled, this will be shortly followed by the OpenFL `collaborator` shutting down:

```SH
INFO :      Disconnect and shut down
INFO     Supernode process terminated. Shutting down gRPC server...                                                               runner_flower.py:96
INFO     gRPC server stopped.                                                                                                     runner_flower.py:98
INFO     Waiting for tasks...                                                                                                     collaborator.py:222
INFO     End of Federation reached. Exiting...  
``` 
Congratulations, you have run a Flower experiment through OpenFL's task runner!

## Advanced Usage
### Long-lived SuperLink and SuperNode
If `autoshutdown` is not enabled, Flower's `ServerApp` and `ClientApp` will shut down at the completion of the Flower experiment, but the `SuperLink` and `SuperNode` will continue to run. As a result, on the `aggregator` terminal, you will see a constant request coming from the `SuperNode`:
```SH
INFO :      GrpcAdapter.PullTaskIns
INFO :      GrpcAdapter.PullTaskIns
INFO :      GrpcAdapter.PullTaskIns
```
You can run another experiment by opening another terminal, navigating to this workspace, and running:
```SH
flwr run ./app-pytorch
```
It will run another experiment. Once you are done, you can manually shut down OpenFL's `collaborator` and Flower's `SuperNode` with `CTRL+C`. This will trigger a task-completion by the task runner that'll subsequently begin the graceful shutdown process of the OpenFL and Flower components.

### Invoke Flower experiment as a separate command
If you did not set `flwr_run_params` in the `plan.yaml`, the OpenFL `FLEX` will not automatically start a Flower experiment. Instead, you should open a terminal, navigate to this workspace, and run 
```SH
flwr run ./app-pytorch
```
separately to begin the experiment.