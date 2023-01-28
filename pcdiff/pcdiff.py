import sys
import paddle
import contextlib
from weight import assign_weight, check_weight_grad
from functools import partial
from utils import (
    reset_log_dir,
    init_options,
    log,
    for_each_grad_tensor,
    tensors_mean,
    max_diff
)
from stack_info import *
from reporter import Report,check_forward_and_backward, current_paddle_report,current_paddle_cpu_report,report_guard
from yaml_loader import global_yaml_loader as yamls
import numpy as np

def auto_layer_diff(layer,input,label=None,auto_weights=True,options={}):
    """
    Given example inputs, automatically find the first layer with precision diff.

    Args:
        layer (paddle.nn.Layer): paddle layer that needs compare
        example_inp (paddle_input, torch_input): input data for paddle layer and torch module.
            paddle_input and torch_input should be dict and send into net like `module(**input)`.
        auto_weights (boolean, optional): uniformly init the parameters of models
        options (dict, optional):
            atol, compare_mode
    Returns:
        True for success, False for failed.
    """
    assert isinstance(layer, paddle.nn.Layer), "Invalid Argument."
    log("Start auto_layer_diff, may need a while to generate reports...")
    paddle_cpu_report = Report("paddle-CPU")
    paddle_xpu_report = Report("paddle-XPU")
    reset_log_dir()
    init_options(options)
    yamls.options = options
    #CPU
    paddle.set_device("cpu")
    cpu_layer = layer
    cpu_layer.to(device='cpu',blocking=True)
    paddle_input_cpu = paddle.to_tensor(input)
    with report_guard(paddle_cpu_report):
        with _register_paddle_hooker(cpu_layer,options):
            try:
                paddle_cpu_output = cpu_layer(paddle_input_cpu)
                loss = tensors_mean(paddle_cpu_output, "paddle")
                if options["diff_phase"] == "both":
                    loss.backward()
            except Exception as e:
                raise RuntimeError(
                    "Exception is thrown while running forward of paddle_layer, please check the legality of layer.\n{}".format(
                        str(e)
                    )
                )
    #CUSTOM DEVICE
    paddle.set_device(options["plat"])
    xpu_layer = layer
    cpu_layer.to(device='npu:0',blocking=True)
    paddle_input_xpu = paddle.to_tensor(input)
    with report_guard(paddle_xpu_report):
        with _register_paddle_hooker(xpu_layer,options):
            try:
                paddle_xpu_output = xpu_layer(paddle_input_xpu)
                loss = tensors_mean(paddle_xpu_output, "paddle")
                if options["diff_phase"] == "both":
                    loss.backward()
            except Exception as e:
                raise RuntimeError(
                    "Exception is thrown while running forward of paddle_layer, please check the legality of layer.\n{}".format(
                        str(e)
                    )
                )
    log("place {} vs {} \n".format(paddle_cpu_output.place,paddle_xpu_output.place))
    diff=max_diff(paddle_cpu_output,paddle_xpu_output)
    log("Max elementwise output diff is {}\n".format(diff))
    
    weight_check, grad_check = check_weight_grad(cpu_layer,xpu_layer, options=options)
    ret = check_forward_and_backward(paddle_cpu_report, paddle_xpu_report, options)
    ret = ret and weight_check and grad_check
    return ret,paddle_cpu_output

def tensor_hook(x_grad, bwd_item, nth_tensor):
    bwd_item.set_input_grads(nth_tensor, x_grad)
    return x_grad

def paddle_layer_hook(module, input, output, idx, options):
    p_rep = current_paddle_report()
    frame_info, frames = extract_frame_summary()
    fwd_item = p_rep.put_item("forward", input, output, module, idx, frame_info, frames)
    bwd_item = p_rep.put_item("backward", input, output, module, idx, frame_info, frames)
    bwd_item.set_forward(fwd_item)
    for i, (t,) in enumerate(for_each_grad_tensor(input)):
        t.register_hook(partial(tensor_hook, bwd_item=bwd_item, nth_tensor=i))
    return None

def traversal_layers(layers, cur_net, layer_map):
    for child in cur_net.children():
        if not (isinstance(child, paddle.nn.Sequential)):
            layers.append(child)
        if child.__class__.__name__ not in layer_map.keys() and child.__class__.__name__ not in layer_map.values():
            traversal_layers(layers, child, layer_map)

@contextlib.contextmanager
def _register_paddle_hooker(layer, options):
    remove_handles = []
    # TODO(xiongkun): duplicate layer is not support, implement custom generator to support (different net_id is ok).
    idx = 0
    layers = [layer]
    traversal_layers(layers,layer,{})
    for mod in layers:
        handle = mod.register_forward_post_hook(partial(paddle_layer_hook, idx=idx, options=options))
        remove_handles.append(handle)
        idx += 1
    yield
    for h in remove_handles:
        h.remove()


##************test**********************************##

class SimpleLayer(paddle.nn.Layer):
    def __init__(self):
        super(SimpleLayer, self).__init__()
        self.linear1 = paddle.nn.Linear(100, 100)
        self.linear2 = paddle.nn.Linear(100, 10)
        self.act = paddle.nn.ReLU()

    def forward(self, x):
        resdual = x
        x = self.linear1(x)
        x = self.act(x)
        x = x + resdual
        x = self.linear2(x)
        return x

if __name__ == "__main__":
    paddle.set_device("cpu")
    layer=SimpleLayer()
    inp = paddle.rand((100, 100))
    ret,pred=auto_layer_diff(layer, inp,auto_weights=True, options={'atol': 1e-4, 'rtol':0, 'plat':'npu','compare_mode': 'strict', 'single_step':False})