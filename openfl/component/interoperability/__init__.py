# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from importlib import util
from openfl.component.interoperability.flex import FederatedLearningExchange

if util.find_spec("flwr") is not None:
    from openfl.component.interoperability.flex_flower import FLEXFlower
