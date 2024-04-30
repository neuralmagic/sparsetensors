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

"""
Contains the ExLLAMA compressor class.
"""
from typing import Dict, Generator, Tuple

from compressed_tensors.compressors import ModelCompressor
from compressed_tensors.compressors.exllama.helpers import (
    translate_state_dict_to_exllama_4bit,
)
from compressed_tensors.config import CompressionFormat
from torch import Tensor


@ModelCompressor.register(name=CompressionFormat.exllama_4bit.value)
class Exllama4BitCompressor(ModelCompressor):
    """
    Exllama 4-bit model compressor.

    Converts a 4 bit quantized state dict to be exllama compatible.

    Transformations include:

    - Convert state dict keys to exllama format
        - weight --> qweight
        - bias --> qbias
        - scales --> qscale
        - zero_points --> qzero_point
    - Adds missing key value pairs in the state dict
        - Adds g_idx tensor to the state dict
    - Converts all other values to fp16
    """

    def compress(self, model_state: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return translate_state_dict_to_exllama_4bit(model_state=model_state)

    def decompress(
        self, path_to_model_or_tensors: str, device: str = "cpu"
    ) -> Generator[Tuple[str, Tensor], None, None]:
        return iter([])
