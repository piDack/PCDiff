# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
from itertools import zip_longest

import numpy
import paddle

from utils import compare_tensor,log
from yaml_loader import global_yaml_loader as yamls

def process_each_weight(process, layer, module, options):
    """
    Apply process for each pair of parameters in layer(paddle) and module(torch)

    Args:
        process (function): process applied to parameters
        layer (paddle.nn.Layer): input paddle layer
        module (torch.nn.Module): input torch module
        layer_mapping (dict, optional): manually map paddle layer and torch module.
        options (dict, optional):
            atol, rtol, compare_mode, single_step
    """

    def _process_runner(
        process,
        paddle_sublayer,
        torch_submodule,
        param_name,
        paddle_param,
        torch_param,
    ):
        try:
            settings = yamls.get_weight_settings(paddle_sublayer, torch_submodule, param_name)
        except Exception as e:
            p_model_log = os.path.join(sys.path[0], "diff_log", "paddle_model_struct.log")
            t_model_log = os.path.join(sys.path[0], "diff_log", "paddle_cpu_model_struct.log")
            with open(p_model_log, "w") as log:
                log.write(str(layer))
            with open(t_model_log, "w") as log:
                log.write(str(module))
            raise e

        process(
            paddle_sublayer,
            torch_submodule,
            param_name,
            paddle_param,
            torch_param,
            settings,
        )

    layers = [layer]
    modules = [module]

    for paddle_sublayer, torch_submodule in zip_longest(layers, modules, fillvalue=None):
        if paddle_sublayer is None or torch_submodule is None:
            raise RuntimeError("Torch and Paddle return difference number of sublayers. Check your model.")
        for (name, paddle_param), (name,torch_param) in zip(
            paddle_sublayer.named_parameters(prefix="", include_sublayers=False),
            torch_submodule.named_parameters(prefix="", include_sublayers=False),
        ):
            _process_runner(
                process,
                paddle_sublayer,
                torch_submodule,
                name,
                paddle_param,
                torch_param,
            )

def _assign_weight(
    paddle_sublayer,
    paddle_cpu_sublayer,
    param_name,
    paddle_param,
    paddle_cpu_param,
    settings,
):
    _shape_check(
        paddle_sublayer,
        torch_submodule,
        param_name,
        paddle_param,
        torch_param,
        settings,
    )
    np_value = paddle.randn(paddle_param.shape).numpy()
    paddle.assign(paddle.to_tensor(np_value), paddle_param)
    paddle.assign(paddle.to_tensor(np_value), paddle_cpu_param)

def assign_weight(layer, cpu_layer, options):
    """
    Init weights of layer and cpu_layer with same value

    Args:
        layer (paddle.nn.Layer): input paddle custom device layer
        cpu_layer (paddle.nn.Layer): input paddle cpu layer
    """
    process_each_weight(_assign_weight, layer, cpu_layer, options)


def _shape_check(
    paddle_sublayer,
    paddle_cpu_sublayer,
    param_name,
    paddle_param,
    paddle_cpu_param,
    settings,
):
    p_shape = list(paddle_param.shape)
    pc_shape = list(paddle_cpu_param.shape)
    assert p_shape == pc_shape, (
        "Shape of param `{}` is not the same. {} vs {}\n"
        "Hint: \n"
        "      1. check whether your paddle model definition and torch model definition are corresponding.\n"
        "      2. check the weight shape of paddle:`{}` and torch:`{}` is the same.\n"
    ).format(param_name, p_shape, pc_shape, paddle_sublayer,paddle_cpu_sublayer)

def check_weight_grad(layer,cpu_layer, options):
    """
    Compare weights and grads between layer(paddle) 

    Args:
        layer (paddle.nn.Layer): input paddle layer
        cpu_layer (paddle.nn.Layer): input torch module
        layer_mapping (dict, optional): manually map paddle layer and torch module.
        options (dict, optional):
            atol, compare_mode
    """
    if options["diff_phase"] == "forward":
        log("Diff_phase is `forward`. Weight and grad check skipped.")
        return True, True

    _weight_check = True
    _grad_check = True
    def _check_weight_grad(
        paddle_sublayer,
        paddle_cpu_sublayer,
        param_name,
        paddle_param,
        paddle_cpu_param,
        settings,
    ):
        nonlocal _weight_check, _grad_check

def check_weight_grad(layer, module, options):
    """
    Compare weights and grads between layer(paddle) and module(torch)

    Args:
        layer (paddle.nn.Layer): input paddle layer
        module (torch.nn.Module): input torch module
        layer_mapping (dict, optional): manually map paddle layer and torch module.
        options (dict, optional):
            atol, compare_mode
    """
    if options["diff_phase"] == "forward":
        log("Diff_phase is `forward`. Weight and grad check skipped.")
        return True, True

    _weight_check = True
    _grad_check = True

    def _check_weight_grad(
        paddle_sublayer,
        paddle_cpu_sublayer,
        param_name,
        paddle_param,
        paddle_cpu_param,
        settings,
    ):
        nonlocal _weight_check, _grad_check
        _shape_check(
            paddle_sublayer,
            paddle_cpu_sublayer,
            param_name,
            paddle_param,
            paddle_cpu_param,
            settings,
        )
        p_param = paddle_param.numpy()
        pc_param = paddle_cpu_param.numpy()
        p_grad = paddle_param.grad.numpy() if paddle_param.grad is not None else None
        pc_grad = paddle_cpu_param.grad.numpy() if paddle_cpu_param.grad is not None else None

        weight_log_path = os.path.join(sys.path[0], "diff_log", "weight_diff.log")
        grad_log_path = os.path.join(sys.path[0], "diff_log", "grad_diff.log")

        _weight_check = compare_tensor(
            p_param,
            pc_param,
            atol=settings["atol"],
            rtol=settings["rtol"],
            compare_mode=settings["compare_mode"],
        )
        _grad_check = compare_tensor(
            p_grad, pc_grad, atol=settings["atol"], rtol=settings["rtol"], compare_mode=settings["compare_mode"]
        )

        if _weight_check is False:
            with open(weight_log_path, "a") as f:
                f.write(
                    "After training, weight value is different for param `{}`.\n"
                    "paddle: `{}` with value:\n{}\n"
                    "torch: `{}` with value:\n{}\n\n".format(
                        param_name, paddle_sublayer, p_param, paddle_cpu_sublayer,pc_param
                    )
                )

        if _grad_check is False:
            with open(grad_log_path, "a") as f:
                f.write(
                    "After training, grad value is different for param `{}`.\n"
                    "paddle: `{}` with value\n{}\n"
                    "torch: `{}` with value\n{}\n\n".format(
                        param_name, paddle_sublayer, p_grad, paddle_cpu_sublayer, pc_grad
                    )
                )

    process_each_weight(_check_weight_grad, layer, module, options)

    if _weight_check and _grad_check:
        log("weight and weight.grad is compared.")
    else:
        diff_log_path = os.path.join(sys.path[0], "diff_log")
        log("Differences in weight or grad !!!")
        log("Check reports at `{}`\n".format(diff_log_path))

    return _weight_check, _grad_check

