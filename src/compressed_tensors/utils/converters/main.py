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


from compressed_tensors.utils.converters.converters import BaseConverter, ConverterNames


__all__ = ["convert_autogptq_checkpoint"]


def convert_autogptq_checkpoint(
    old_checkpoint_path, new_checkpoint_path, **kwargs
) -> str:
    """
    Convert an autogptq checkpoint to a compressed tensor checkpoint

    :param old_checkpoint_path: the path to the autogptq checkpoint
    :param new_checkpoint_path: the path to save the converted compressed
        tensor checkpoint
    :param kwargs: additional arguments to pass to the transformations
    :return: the path to the new checkpoint
    """
    converter: BaseConverter = BaseConverter.load_from_registry(
        ConverterNames.AutoGPTQConverter
    )
    checkpoint_path = converter.convert_from_safetensors(
        old_checkpoint_path, new_checkpoint_path, **kwargs
    )
    return checkpoint_path
