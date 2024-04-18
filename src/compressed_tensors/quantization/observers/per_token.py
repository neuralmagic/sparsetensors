# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Tuple

import torch
from compressed_tensors.quantization.observers.base import Observer
from compressed_tensors.quantization.observers.helpers import calculate_qparams
from compressed_tensors.quantization.quant_args import QuantizationArgs
from torch import FloatTensor, IntTensor, Tensor


__all__ = ["PerTokenObserver"]


@Observer.register("per_token", alias="per_token_dynamic")
class PerTokenObserver(Observer):
    """
    Values targted for a dyanmic observer do not require calibration,
    this observer will persist in the model through the lifecycle, calculating
    the quantization parameters on the fly for each observed Tensor.

    This base dynamic observer uses the `calculate_qparams` from MemorylessObserver
    where each scale and zero point is based solely on the currently observed
    Tensor.

    :param axis: axis that token dimension is expected to be in
    """

    def __init__(self, quantization_args: QuantizationArgs, axis: int = 1):
        super().__init__(quantization_args=quantization_args)

        self.axis = 1

    DYNAMIC = True

    def calculate_qparams(self, observed: Tensor) -> Tuple[FloatTensor, IntTensor]:
        """
        :param observed: observed tensor to calculate quantization parameters for
        :return: tuple of scale and zero point derived from the observed tensor
        """
        # reduce every dimension except token dimension
        reduce_dims = [idx for idx in range(observed.dim()) if idx != self.axis]

        # return shape will be [1, ..., num_tokens, 1, ...] with same num dims
        min_vals = observed.amin(dim=reduce_dims, keepdim=True)
        max_vals = observed.amax(dim=reduce_dims, keepdim=True)

        # ensure zero is in the range
        min_vals = torch.min(min_vals, torch.zeros_like(min_vals))
        max_vals = torch.max(max_vals, torch.zeros_like(max_vals))

        # returned shape will match the min/max vals shape
        # since keepdim=True, the reduced dims will have their dims set to 1
        # so scales and zero points should broadcast correctly along the
        # token axis
        # TODO: add test for the broadcast mentioned above
        return calculate_qparams(min_vals, max_vals, self.quantization_args)
