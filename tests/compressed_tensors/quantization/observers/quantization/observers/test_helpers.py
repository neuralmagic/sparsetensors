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

from compressed_tensors.quantization import (
    QuantizationConfig,
    apply_quantization_config,
)
from compressed_tensors.quantization.observers.helpers import get_observer_token_count
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_get_observer_token_count():
    model = AutoModelForCausalLM.from_pretrained("Isotonic/TinyMixtral-4x248M-MoE")
    tokenizer = AutoTokenizer.from_pretrained("Isotonic/TinyMixtral-4x248M-MoE")
    model.eval()
    config = QuantizationConfig(
        format="fakequant",
        quantization_status="calibration",
        config_groups={
            "group_1": {
                "input_activations": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": False,
                    "strategy": "tensor",
                },
                "targets": ["Linear"],
            },
        },
    )
    apply_quantization_config(model, config)

    # start calibration
    calib_str = "I am a string that is used for calibration so that your model is quantized properly."  # noqa
    calibration_tokens = tokenizer(calib_str, return_tensors="pt")
    model(**calibration_tokens)
    counter = get_observer_token_count(model)

    # filter out the None values
    # (tokens, in the appropriate format, that were not observed by the model)
    counter = {k: v for k, v in counter.items() if v is not None}

    # iterate over all the layers in the model where the token count in the proper
    # format is has been observed
    for i in range(model.config.num_hidden_layers):
        # fetch the tokens observed by the router
        tokens_observed_by_router = counter.pop(
            f"model.layers.{i}.block_sparse_moe.gate"
        )
        assert tokens_observed_by_router == len(calibration_tokens.input_ids.flatten())

        # fetch the sum of tokens observed by all the experts
        sum_tokens_observed_by_experts = 0
        keys_for_this_layer = [
            k
            for k in counter.keys()
            if f"model.layers.{i}.block_sparse_moe.experts" in k
        ]
        for key in keys_for_this_layer:
            sum_tokens_observed_by_experts += counter.pop(key)

        # each Mixtral expert is comprised of 3 linear layers,
        # so we need to multiply by 3
        assert (
            sum_tokens_observed_by_experts
            == len(calibration_tokens.input_ids.flatten())
            * model.config.num_experts_per_tok
            * 3
        )

    # there are no more information in the counter
    assert len(counter) == 0
