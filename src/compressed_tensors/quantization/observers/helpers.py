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
from compressed_tensors.quantization.quant_args import QuantizationArgs, QuantizationType
from torch import FloatTensor, IntTensor, Tensor


__all__ = ["calculate_qparams", "calculate_range"]


def calculate_qparams(
    min_vals: Tensor, max_vals: Tensor, quantization_args: QuantizationArgs
) -> Tuple[FloatTensor, IntTensor]:
    """
    :param min_vals: tensor of min value(s) to caluclate scale(s) and zero point(s)
        from
    :param max_vals: tensor of max value(s) to caluclate scale(s) and zero point(s)
        from
    :param quantization_args: settings to quantization
    :return: tuple of the calculated scale(s) and zero point(s)
    """
    min_vals = torch.min(min_vals, torch.zeros_like(min_vals))
    max_vals = torch.max(max_vals, torch.zeros_like(max_vals))
    device = min_vals.device

    bit_min, bit_max = calculate_range(quantization_args, device)
    bit_range = bit_max - bit_min

    if quantization_args.type == QuantizationType.FLOAT:
        #TODO: don't assume symmetric
        max_val_pos = torch.max(-min_vals, max_vals)
        scales = (bit_max / max_val_pos.clamp(min=1e-12)).float().reciprocal()
        zero_points = torch.zeros(scales.shape, device=device, dtype=torch.float8_e4m3fn)
    elif quantization_args.symmetric:
        max_val_pos = torch.max(-min_vals, max_vals)
        scales = max_val_pos / (float(bit_range) / 2)
        scales = torch.clamp(scales, min=torch.finfo(torch.float32).eps)
        zero_points = torch.zeros(scales.shape, device=device, dtype=torch.int8)
    else:
        scales = (max_vals - min_vals) / float(bit_range)
        scales = torch.clamp(scales, min=torch.finfo(torch.float32).eps)
        zero_points = bit_min - torch.round(min_vals / scales)
        zero_points = torch.clamp(zero_points, bit_min, bit_max).to(torch.int8)

    return scales, zero_points

def calculate_range(quantization_args: QuantizationArgs, device: str) -> Tuple:
    """
    """
    if quantization_args.type == QuantizationType.INT:
        bit_range = 2**quantization_args.num_bits
        q_max = torch.tensor(bit_range / 2 - 1, device=device)
        q_min = torch.tensor(-bit_range / 2, device=device)
    else: # QuantizationType.FLOAT
        if quantization_args.num_bits != 8:
            raise ValueError(
                "Floating point quantization is only supported for 8 bits,"
                f"got {quantization_args.num_bits}"
            )
        fp_range_info = torch.finfo(torch.float8_e4m3fn)
        q_max = torch.tensor(fp_range_info.max, device=device)
        q_min = torch.tensor(fp_range_info.min, device=device)

    return q_min, q_max
