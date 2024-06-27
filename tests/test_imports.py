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


def test_util_imports_correct_module():
    """
    Test that the utils module can be imported from the compressed_tensors package
    does not get imported from the compressors package
    """
    import compressed_tensors.compressors.utils as compressor_utils
    import compressed_tensors.utils

    assert compressed_tensors.utils is not compressor_utils, (
        "compressed_tensors.utils should not be the same as "
        "compressed_tensors.compressors.utils"
    )

    # Note: this could be any module in the utils directory
    module_name = "safetensors_load"
    assert hasattr(
        compressed_tensors.utils, module_name
    ), f"{module_name} should be importable from compressed_tensors.utils"
    import compressed_tensors.utils.safetensors_load as actual_module
    from compressed_tensors.utils import safetensors_load as expected_module

    assert (
        actual_module == expected_module
    ), "The two import methods should import the same module"
