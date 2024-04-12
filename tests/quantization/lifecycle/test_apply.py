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


from sparsetensors.quantization.lifecycle import apply_quantization_config
from sparsetensors.quantization.quant_config import QuantizationConfig
from transformers import AutoModelForCausalLM


def test_apply_quantization_config_tinyllama():
    quant_config = get_sample_tinyllama_quant_config()
    model = get_tinyllama_model()

    # check that model is not already quantized
    for module in model.modules():
        _test_layer_quantization_status(module, inputs=False, weights=False)

    # apply quant config to model
    apply_quantization_config(model, quant_config)

    # check for correct application of quant config
    num_linears = 0
    num_embeddings = 0
    num_rotary_embeddings = 0
    for module in model.modules():
        module_type = module.__class__.__name__
        if module_type == "Linear":
            num_linears += 1
            _test_layer_quantization_status(module, inputs=True, weights=True)
        elif module_type == "Embedding":
            num_embeddings += 1
            _test_layer_quantization_status(module, inputs=False, weights=True)
        elif module_type == "LlamaRotaryEmbedding":
            num_rotary_embeddings += 1
            _test_layer_quantization_status(module, inputs=False, weights=False)

    # sanity check correct number of layers targeted
    assert num_linears == 155
    assert num_embeddings == 1
    assert num_rotary_embeddings == 22


def _test_layer_quantization_status(module, inputs: bool, weights: bool):
    # check if quantization is applied at all (true if inputs or weights targeted)
    quantized = inputs or weights
    assert hasattr(module, "quantization_scheme") == quantized
    assert hasattr(module, "quantization_status") == quantized

    # check for inputs
    assert hasattr(module, "input_scale") == inputs
    assert hasattr(module, "input_zero_point") == inputs
    assert hasattr(module, "weight_scale") == weights
    assert hasattr(module, "weight_zero_point") == weights


def get_tinyllama_model():
    return AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    )


def get_sample_tinyllama_quant_config():
    config_dict = {
        "quant_method": "sparseml",
        "format": "fakequant",
        "quantization_status": "frozen",
        "global_compression_ratio": None,
        "config_groups": {
            "group_1": {
                "weights": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "tensor",
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "tensor",
                },
                "targets": ["Linear"],
            },
            "group_2": {
                "weights": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": False,
                    "strategy": "tensor",
                },
                "input_activations": None,
                "targets": ["Embedding"],
            },
        },
        "ignore": ["LlamaRotaryEmbedding"],
    }
    return QuantizationConfig.parse_obj(config_dict)


test_apply_quantization_config_tinyllama()
