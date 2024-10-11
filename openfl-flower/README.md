# OpenFL and Flower

This rudimentary integration is a proof-of-concept test to explore communication between OpenFL and Flower using gRPC.

## Version Compatibility

| OpenFL Version | Flower Version         |
|----------------|------------------------|
| 1.6 (dev)      | 1.12.0.dev20241010 (nightly) |

## Setup Instructions

Create proto:
```bash
cd openfl/proto
python -m grpc_tools.protoc -I=. --python_out=. --grpc_python_out=. openfl.proto
```

To set up the integration, follow these steps:

1. Open 6 terminals on your machine to run the different components of the system.

2. Terminal 1: Start the Flower superlink.
```bash
flower-superlink --insecure --fleet-api-type grpc-adapter --fleet-api-address 127.0.0.1:9093 --driver-api-address 127.0.0.1:9091
```

3. Terminal 2: Start the Flower server application.
```bash
flower-server-app ./app-pytorch --insecure --superlink 127.0.0.1:9091
```

4. Terminal 3: Start the OpenFL server with a local gRPC client, connecting to the Flower superlink and OpenFL client.
```bash
python openfl_server_with_local_grpc_client.py
```

5. Terminal 4: Start the OpenFL client with a local gRPC server, connecting to the OpenFL server and Flower superlink.
```bash
python openfl_client_with_local_grpc_server.py
```

6. Terminal 5: Start the first Flower supernode and client application.
```bash
flower-supernode ./app-pytorch --insecure --grpc-adapter --superlink 127.0.0.1:9092 --node-config "num-partitions=2 partition-id=0"
```

7. Terminal 6: Start the second Flower supernode and client application.
```bash
flower-supernode ./app-pytorch --insecure --grpc-adapter --superlink 127.0.0.1:9092 --node-config "num-partitions=2 partition-id=1"
```

## Overview
The integration setup involves running multiple components that facilitate federated learning across the OpenFL and Flower frameworks:
- The **Flower superlink** acts as a bridge between the Flower server application and the OpenFL server, allowing for message passing and coordination.
- The **Flower server** application manages the federated learning process, coordinating with the supernodes to train models on distributed data.
- The **OpenFL server with local gRPC client** connects to the Flower superlink and serves as an intermediary for the OpenFL client, translating and forwarding messages.
- The **OpenFL client with local gRPC server** communicates with the OpenFL server and forwards messages to the Flower supernodes.
- The **Flower supernodes** represent the distributed clients in the federated learning setup, each responsible for a portion of the data and computation.