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

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, validator


__all__ = ["QuantizationType", "QuantizationStrategy", "QuantizationArgs"]


class QuantizationType(str, Enum):
    """
    Enum storing quantization type options
    """

    INT = "int"
    FLOAT = "float"


class QuantizationStrategy(str, Enum):
    """
    Enum storing quantization strategy options
    """

    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"
    BLOCK = "block"


class QuantizationArgs(BaseModel):
    """
    User facing arguments used to define a quantization config for weights or
    activations

    :param num_bits: quantization bit depth
    :param type: dtype to quantized to, either int or float
    :param symmetric: whether or not quantization scale is symmetric about zero-point
    :param strategy: string id determining the scope of scale/zero-point to apply
    :param group_size: group length to use for the group strategy
    :param block_structure: 2d block structure to use for the block strategy, must be
    of the format "2x4", "8x16", etc.
    :param dynamic: set True to perform dynamic quantization - values will not be
        calibrated during calibration phase, instead during inference new quantization
        ranges will be observed with every sample. Defaults to False for static
        quantization. Note that enabling dynamic quantization will change the default
        observer to a memoryless one
    """

    num_bits: int = 8
    type: QuantizationType = QuantizationType.INT
    symmetric: bool = True
    strategy: QuantizationStrategy = QuantizationStrategy.TENSOR
    group_size: Optional[int] = None
    block_structure: Optional[str] = None
    dynamic: bool = False
    observer: str = Field(
        default="minmax",
        description=(
            "The class to use to compute the quantization param - "
            "scale and zero-point'"
        ),
    )
    observer_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "optional dict of kwargs to be passed directly to torch quantization "
            "Observers constructor excluding quantization range or symmetry"
        ),
    )

    def get_observer(self):
        """
        :return: torch quantization FakeQuantize built based on these QuantizationArgs
        """
        from compressed_tensors.quantization.observers.base import Observer

        if self.observer == "minmax" and self.dynamic:
            # override defualt observer for dynamic, you never want minmax which
            # keeps state across samples for dynamic
            self.observer = "memoryless"

        return Observer.load_from_registry(self.observer, quantization_args=self)

    @validator("strategy", pre=True)
    def validate_strategy(cls, value, values):
        group_size = values.get("group_size")
        if group_size is not None:
            if group_size > 0:
                if value != QuantizationStrategy.GROUP:
                    raise ValueError(
                        f"group_size={group_size} with strategy {value} is invald. "
                        "Please set strategy to 'group'"
                    )
                return QuantizationStrategy.GROUP

            elif group_size == -1:
                if value != QuantizationStrategy.CHANNEL:
                    raise ValueError(
                        f"group_size={group_size} with strategy {value} is invald. "
                        "Please set strategy to 'channel'"
                    )
                return QuantizationStrategy.CHANNEL

            else:
                raise ValueError(
                    f"group_size={group_size} with strategy {value} is invald. "
                    "group_size > 0 for strategy='group' and "
                    "group_size = -1 for 'channel'"
                )

        return value
