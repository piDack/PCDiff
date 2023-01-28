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

import contextlib
from actions import get_action
from stack_info import print_frames
from utils import (
    TableView,
    TreeView,
    clone_structure,
    for_each_grad_tensor,
    for_each_tensor,
    log,
)

class Counter:
    def __init__(self):
        self.clear()

    def clear(self):
        self.id = 0

    def get_id(self):
        ret = self.id
        self.id += 1
        return ret
    
class ReportItem:
    def __init__(self, type, step, input, output, net, net_id, frame_info, frames):
        assert type in [
            "forward",
            "backward",
        ], "type can only be one of ['forward', 'backward']"
        self.type = type
        self.step = step
        """
        self.input is a tuple: (tensor, ...)
        """
        # self.input = clone_tensors(input)
        self.input = input
        if self.type == "forward":
            # we only clone output in forward step.
            self.output = clone_structure(output)
        else:
            self.output = output
        self.net = net
        self.net_id = net_id
        self.fwd_item = None
        self.bwd_item = None
        self.frame_info = frame_info
        self.frames = frames
        self.input_grads = self._gen_input_grads()

    def set_forward(self, fwd):
        assert self.type == "backward", "can't set forward for non-backward item."
        fwd.bwd_item = self
        self.fwd_item = fwd

    def _gen_input_grads(self):
        if self.type == "forward":
            return None
        assert self.input is not None, "Backward while input is None, not expected."

        return [None for i in for_each_grad_tensor(self.input)]

    def set_input_grads(self, nth, value):
        assert nth < len(self.input_grads)
        self.input_grads[nth] = value

    def print_stacks(self):
        print_frames(self.frames)

    def stacks(self):
        return self.frames

    def compare_tensors(self):
        if self.type == "forward":
            return for_each_tensor(self.output)
        if self.type == "backward":
            return for_each_tensor(self.input_grads)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        strings = []
        strings.append("ReportItem: \n    type={}".format(self.type))
        strings.append("    step_idx: {}".format(self.step))
        return "\n".join(strings)

class Report:
    def __init__(self, name):
        self.name = name
        self.items = []
        self.counter = None
    
    def put_item(self, type, input, output, net, net_id, frame_info, frames):
        step = self.counter.get_id()
        self.items.append(
            ReportItem(
                type=type,
                step=step,
                input=input,
                output=output,
                net=net,
                net_id=net_id,
                frame_info=frame_info,
                frames=frames,
            )
        )
        return self.items[-1]

    def get_fwd_items(self):
        sorted(self.items, key=lambda x: x.step)
        return list(filter(lambda x: x.type == "forward", self.items))

    def get_bwd_items(self):
        sorted(self.items, key=lambda x: x.step)
        return list(filter(lambda x: x.type == "backward", self.items))

    def find_item(self, p_report, net_id):
        tlist = list(filter(lambda x: x.type == "forward" and x.net_id == net_id, self.items))
        plist = list(filter(lambda x: x.type == "forward" and x.net_id == net_id, p_report.items))
        return tlist[len(plist) - 1]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        sorted(self.items, key=lambda x: x.step)
        strings = []
        strings.append("Report name is: " + self.name)
        for item in self.items:
            strings.append("    " + str(item.step) + ": [{}]".format(type(item.net)))
        return "\n".join(strings)
    

global_paddle_report = None
global_paddle_counter = Counter()

@contextlib.contextmanager
def report_guard(paddle_report):
    global global_paddle_report
    old_p = global_paddle_report
    try:
        global_paddle_report = paddle_report
        paddle_report.counter = global_paddle_counter
        paddle_report.counter.clear()
        yield
    finally:
        global_paddle_report = old_p
        paddle_report.counter = None

def current_paddle_report():
    if global_paddle_report is None:
        raise RuntimeError(
            "Please call `current_paddle_report()` within contextmanager `report_guard(Report(), Report())`."
        )
    return global_paddle_report

def current_paddle_cpu_report():
    if global_paddle_cpu_report is None:
        raise RuntimeError(
            "Please call `current_paddle_cpu_report()` within contextmanager `report_guard(Report(), Report())`."
        )
    return global_paddle_cpu_report


def print_info(paddle_item, exc, step_idx, grad=False):
    log("FAILED !!!")
    if grad:
        log(
            "    Diff found in `Backward Stage` in step: {}, net_id is {}".format(
                step_idx, paddle_item.net_id
            )
        )
    else:
        log(
            "    Diff found in `Forward  Stage` in step: {}, net_id is {}".format(
                step_idx, paddle_item.net_id 
            )
        )
    log("    Type of layer is  : {}".format(type(paddle_item.net)))
    print(str(exc))

    print("\n\nPaddle Stacks:")
    print("=========================")
    paddle_item.print_stacks()

def check_forward_and_backward(torch_rep, paddle_rep, cfg):
    torch_fwd_items = torch_rep.get_fwd_items()
    paddle_fwd_items = paddle_rep.get_fwd_items()
    print(paddle_fwd_items)
    paddle_bwd_items = paddle_rep.get_bwd_items()
    torch_tree_view = TreeView(torch_fwd_items)
    paddle_tree_view = TreeView(paddle_fwd_items)

    backward_items = []
    # forward check
    for idx, paddle_item in enumerate(paddle_fwd_items[::-1]):
        print(paddle_item.step)
        torch_item = torch_fwd_items[paddle_item.net_id]
        paddle_item = paddle_fwd_items[paddle_item.net_id]
        assert torch_item.type == "forward" and paddle_item.type == "forward"
        act = get_action(torch_item.net, paddle_item.net)
        try:
            backward_items.append([torch_item.bwd_item,paddle_item.bwd_item])
            act(torch_item, paddle_item, cfg)
        except Exception as e:
            if cfg["single_step"]:
                log("Under single_step mode:")
            print_info(paddle_item, e, idx, grad=False)
            return False

    log("forward {} steps compared.".format(len(paddle_fwd_items)))

    if cfg["diff_phase"] == "forward":
        log("Diff_phase is `forward`. Backward compare skipped.")
        log("SUCCESS !!!")
        return True

    # backward check
    # backward_map map from id(paddle_backward_item) to torch_backward_item
    backward_map = TableView(backward_items, lambda x: x[1].step)
    """
    TODO(xiongkun): the order is problematic because we consider the tree structure as a chain structure.
          so, always the root layer is calculated first. but we want the first layer with diff.

    """
    for idx, paddle_item in enumerate(paddle_bwd_items[::-1]):
        torch_item,paddle_item = backward_map[paddle_item.step]
        assert torch_item.type == "backward" and paddle_item.type == "backward"
        act = get_action(torch_item.net, paddle_item.net)
        try:
            act(torch_item, paddle_item, cfg)
        except Exception as e:
            print_info(paddle_item, e, idx, grad=True)
            return False

    log("bacward {} steps compared.".format(len(backward_items)))

    # total status
    log("SUCCESS !!!")
    return True