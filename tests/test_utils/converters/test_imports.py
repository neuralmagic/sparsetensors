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

import pytest


@pytest.fixture(scope="module")
def conversion_function():
    from compressed_tensors.utils.converters import convert_autogptq_checkpoint

    return convert_autogptq_checkpoint


@pytest.mark.parametrize(
    "parent_module, function_name",
    [
        ("compressed_tensors.utils", "convert_autogptq_checkpoint"),
        ("compressed_tensors.utils.main", "convert_autogptq_checkpoint"),
    ],
)
def test_convert_function_is_importable(parent_module, function_name):
    import importlib

    module = importlib.import_module(parent_module)
    assert hasattr(
        module, function_name
    ), f"{function_name} is not found in {parent_module}"


def test_conversion_function_accepts_correct_arguments(conversion_function):
    import inspect

    sig = inspect.signature(conversion_function)
    params = sig.parameters
    assert (
        "old_checkpoint_path" in params
    ), "Function does not accept 'old_checkpoint_path' argument"
    assert (
        "new_checkpoint_path" in params
    ), "Function does not accept 'new_checkpoint_path' argument"

    # check keyword arguments are also accepted
    # (might be needed to configure specific transformations)
    assert any(
        param.kind == param.VAR_KEYWORD for param in params.values()
    ), "Function does not accept **kwargs"
