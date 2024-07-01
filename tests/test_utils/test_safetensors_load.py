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
from compressed_tensors.utils.safetensors_load import validate_safetensors_file_path


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path / "subdirectory"


@pytest.fixture
def safetensors_file(temp_dir):
    temp_dir.mkdir(exists_ok=True)
    safetensors_filepath = temp_dir / "test.safetensors"
    safetensors_filepath.write_text("content")
    return safetensors_filepath


@pytest.fixture
def non_safetensors_file(temp_dir):
    temp_dir.mkdir(exists_ok=True)
    non_safetensors_filepath = temp_dir / "test.txt"
    non_safetensors_filepath.write_text("content")
    return non_safetensors_filepath


def test_validate_safetensors_file_path_file_not_found():
    with pytest.raises(FileNotFoundError):
        validate_safetensors_file_path("nonexistent_file.safetensors")


def test_validate_safetensors_file_path_no_safetensors_files_in_directory(temp_dir):
    temp_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        validate_safetensors_file_path(str(temp_dir))


def test_validate_safetensors_file_path_file_is_not_safetensors(non_safetensors_file):
    with pytest.raises(ValueError):
        validate_safetensors_file_path(str(non_safetensors_file))


def test_validate_safetensors_file_path_valid_safetensors_file(safetensors_file):
    validate_safetensors_file_path(str(safetensors_file))


def test_validate_safetensors_file_path_valid_directory_with_safetensors_files(
    temp_dir, safetensors_file
):
    validate_safetensors_file_path(str(temp_dir))
