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


from compressed_tensors.quantization.observers.base import Observer
from compressed_tensors.quantization.observers.memoryless import MemorylessObserver


__all__ = ["DynamicObserver"]


@Observer.register("dynamic")
class DynamicObserver(MemorylessObserver):
    """
    Values targted for a dyanmic observer do not require calibration,
    this observer will persist in the model through the lifecycle, calculating
    the quantization parameters on the fly for each observed Tensor.

    This base dynamic observer uses the `calculate_qparams` from MemorylessObserver
    where each scale and zero point is based solely on the currently observed
    Tensor.
    """

    DYNAMIC = True
