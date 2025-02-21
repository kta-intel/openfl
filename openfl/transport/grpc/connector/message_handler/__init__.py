# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from importlib import util
from openfl.transport.grpc.connector.message_handler.message_handler import MessageHandler

if util.find_spec("flwr") is not None:
    from openfl.transport.grpc.connector.message_handler.message_handler_flower import MessageHandlerFlower
