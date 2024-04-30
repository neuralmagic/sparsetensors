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

from typing import Dict

from compressed_tensors.compressors.exllama.transformations import (
    GPTQ_EXLLAMA_TRANSFORMATIONS,
)
from torch import Tensor


def translate_state_dict_to_exllama_4bit(
    model_state: Dict[str, Tensor]
) -> Dict[str, Tensor]:
    """
    Translate the state dict to the Exllama format.

    Changes made to quantized params in the passed state_dict:
    - weight tensor renamed to qweight, and the corresponding tensor
        value of shape [x, 8y] will be repacked to [x, y]
    - scale tensor renamed to scales, and the corresponding tensor
        value of shape [8x] will be reshaped to [1, 8x] and
        then repacked to [1, x]
    - zero_point tensor renamed to qzeros, and the corresponding tensor
        value of shape [x] will be reshaped to [1, x]
    - A g_idx tensor of shape [num_channels] will be added to the
        state_dict, this tensor will be filled with zeros
    - All fake quantization parameters will be removed from the state_dict




    :param state_dict: The model state dict to be translated.
    :return: The translated state dict compatible with Exllama.
    """

    state_dict_copy = {}
    for transformation in GPTQ_EXLLAMA_TRANSFORMATIONS:
        state_dict_copy: Dict[str, Tensor] = transformation(
            state_dict=state_dict_copy or model_state
        )
    return state_dict_copy
