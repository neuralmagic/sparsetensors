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

import copy
import logging
import shutil
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, Tuple, Union

import torch
from compressed_tensors.registry.registry import RegistryMixin
from compressed_tensors.utils.converters.transformations import (
    remove_unused_tensors,
    transform_autogptq_weights_and_reshape_tensors,
    transform_exllama_names,
)
from compressed_tensors.utils.safetensors_load import validate_safetensors_file_path
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm


StateDictType = Union[Dict[str, torch.Tensor], str, Path]
TransformationType = Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]

_LOGGER: logging.Logger = logging.getLogger(__name__)


class ConverterNames(str, Enum):
    AutoGPTQConverter: str = "exllama_to_compressed_tensor"


class BaseConverter(ABC, RegistryMixin):
    @classmethod
    def translate(cls, state_dict: StateDictType, **kwargs) -> StateDictType:
        """
        Applies transformations to the state_dict

        :param state_dict: The state_dict to apply transformations to
        :param kwargs: Additional arguments to pass to the transformations
        :return: The transformed state_dict
        """
        _LOGGER.info("Applying transformations...")
        new_state_dict = copy.copy(state_dict)
        for transformation in cls.transformations():
            new_state_dict = transformation(new_state_dict, **kwargs)
        return new_state_dict

    @classmethod
    def convert_from_safetensors(
        cls, filepath: str, save_dir: str = None, **kwargs
    ) -> str:
        """
        Convert a .safetensors file or directory of .safetensors files, applying
        transformations to the state_dict and saving the new state_dict to a new
        directory

        :param filepath: The file path to the .safetensors file or directory
            containing .safetensors files to convert
        :param save_dir: The directory to save the converted state_dict to
        :return: The directory where the converted state_dict was saved
        """
        validate_safetensors_file_path(filepath)

        filepath_: Path = Path(filepath)
        if not save_dir:
            save_dir: str = "compressed_tensors_model"

        save_dir_: Path = Path(save_dir)
        save_dir_.mkdir(exist_ok=True, parents=True)

        metadata = {"format": "pt", "source": "Created by SparseML"}
        # transform and save the state_dict
        if filepath_.is_dir():
            tqdm.write(f"Converting directory: {filepath}")
            tqdm.write(
                f"Found: {len(list(filepath_.glob('*.safetensors')))} "
                ".safetensors files"
            )
            for file in filepath_.glob("*.safetensors"):
                tqdm.write(f"Converting file: {file.name}")
                new_state_dict = {}
                state_dict: Iterable[StateDictType] = load_safetensors_state_dict(
                    file, by_layers=True
                )
                layer_progress_bar = tqdm(
                    state_dict, total=layer_count(file), desc="Converting layers"
                )
                for layer_state_dict in layer_progress_bar:
                    layer_name = list(layer_state_dict.keys())[0][
                        : len("model.layers.0")
                    ]
                    layer_progress_bar.set_description(f"Converting layer {layer_name}")
                    layer_progress_bar.update()
                    new_state_dict.update(
                        cls.translate(state_dict=layer_state_dict, **kwargs)
                    )

                if new_state_dict:
                    # compress before saving
                    # compressor = Compressor.load_from_registry(
                    # name=CompressionFormat.pack_quantized.value
                    # )
                    # new_state_dict = compressor.compress(new_state_dict)
                    save_file(
                        new_state_dict,
                        filename=save_dir_ / file.name,
                        metadata=metadata,
                    )
            _copy_non_safetensor_files_(filepath_, save_dir_)
            # _update_quantization_config(filepath_, save_dir_)

        elif filepath_.is_file():
            new_state_dict = {}
            state_dict: Iterable[StateDictType] = load_safetensors_state_dict(
                file, by_layers=True
            )
            for layer_state_dict in state_dict:
                new_state_dict.update(cls.translate(state_dict=layer_state_dict))

            save_file(
                new_state_dict, save_path=save_dir_ / filepath_.name, metadata=metadata
            )

        return str(save_dir_)

    @classmethod
    @abstractmethod
    def transformations(cls) -> Iterable[TransformationType]:
        """
        Returns an iterable of transformations that are applied in the converter,
        each transformation should be a callable that takes a state_dict and returns
        a transformed state_dict
        """
        raise NotImplementedError()


@BaseConverter.register(name=ConverterNames.AutoGPTQConverter)
class AutoGPTQConverter(BaseConverter):
    """
    A converter that applies transformations to the state_dict of a autogptq
    quantized model to convert it to a compressed tensor model

    Transformations made:

    -> Unpack autogptq 4 bit weight packing
    -> Translate exllama names to compressed tensor names
    -> Pack 4 bit weights with compressed tensor format
    -> Remove unused tensors
    -> Update quantization config in config.json file
    """

    @classmethod
    def transformations(cls):
        return (
            transform_autogptq_weights_and_reshape_tensors,
            transform_exllama_names,
            remove_unused_tensors,
        )


def _copy_non_safetensor_files_(source_dir: Path, dest_dir: Path):
    """
    A helper function to copy all auxillary files in a directory that are
    not .safetensors files, for example (config.json, recipe.yaml, ...)

    :param source_dir: The directory to copy files from
    :param dest_dir: The directory to copy files to
    """
    for file in source_dir.glob("*"):
        if file.suffix != ".safetensors" and file.name != "config.json":
            _LOGGER.info(f"Copying file: {file} to {dest_dir}")
            shutil.copy(file, dest_dir / file.name)


def _update_quantization_config(source_dir: Path, dest_dir: Path):
    """
    Updates config.json file in the destination directory by removing the
    quantization_config attribute

    :param source_dir: The directory containing the original config.json file
    :param dest_dir: The directory to save the updated config.json file
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(source_dir)

    if hasattr(config, "quantization_config"):
        _LOGGER.info("Updating quantization config...")
        quantization_config = config.quantization_config
        config.quantization_config = _convert_to_compressed_tensors_config(
            quantization_config
        )
    config.save_pretrained(dest_dir)


def _convert_to_compressed_tensors_config(quantization_config):
    """
    Converts the quantization_config attribute from a config.json file
    to a dictionary

    :param quantization_config: The quantization_config
        attribute from a config.json file
    :return: The quantization_config as a dictionary
    """
    compressed_tensor_config = ...
    return compressed_tensor_config


def layer_count(file_path: str) -> int:
    """
    Count the number of layers in a safetensors file

    :param file_path: path to the safetensors file
    :return: number of layers in the safetensors file
    """
    with safe_open(file_path, framework="pt", device="cpu") as f:
        keys = sorted(f.keys())

        last_layer_name = None
        layer_count = 0
        for key in keys:
            layer_name = key[: len("model.layers.0")]
            if layer_name != last_layer_name:
                last_layer_name = layer_name
                layer_count += 1
        return layer_count


def load_safetensors_state_dict(
    file_path: str, by_layers: bool = True
) -> Iterator[Tuple[str, Dict[str, torch.Tensor]]]:
    """
    Load a safetensors file from disk

    :param file_path: path to the safetensors file
    :param by_layers: if True, return a iterator with dictionary of safetensors
        data by layers. Default is True
    :return: Iterator of dictionary of safetensors data or iterator of
        dictionaries by layers
    """
    with safe_open(file_path, framework="pt", device="cpu") as f:
        if by_layers:
            current_layer = None
            layer_data = {}
            for key in sorted(f.keys()):
                layer_name = key[: len("model.layers.0")]
                if current_layer is None:
                    current_layer = layer_name
                elif layer_name != current_layer:
                    yield layer_data
                    current_layer = layer_name
                    layer_data = {}
                layer_data[key] = f.get_tensor(key)
            if layer_data:
                yield layer_data
        else:
            yield {key: f.get_tensor(key) for key in f.keys()}
