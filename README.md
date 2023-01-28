# PCDiff
**P**addle **C**ustomDevice **Diff** precision toolkits which is forked from [PaDiff](https://github.com/PaddlePaddle/PaDiff) repo.


## 简介
PCDiff是基于PaddlePaddle专用于不同硬件平台的模型精度对齐工具。传入Paddle模型，PCDiff将对不同硬件训练过程中的所有中间结果以及训练后的模型权重进行对齐检查，并以调用栈的形式提示模型第一次出现精度diff的位置。


## 安装
```
pip install pcdiff
```

尝鲜版或开发者推荐如下命令安装：
```
pip install -e .
```
## 使用说明

### auto_layer_diff 使用接口与参数说明

接口函数签名：`auto_layer_diff(layer, example_inp, auto_weights=True, options={})`

-   layer：传入paddle模型

-   example_inp：传入输入的样例数据，样例数据为paddle tensor类型。

-   auto_weights: 是否使用随机数值统一初始化paddle custom device与cpu模型，默认为True

-   options：一个传递参数的字典

       -   "atol": 绝对精度误差上限，默认值为 `0`

       -   "rtol": 相对精度误差上限，默认值为 `1e-7`

       -   "plat": 精度对比平台，默认值为 `npu`

       -   "diff_phase": `"both"|"forward"`默认为`"both"`。设置为`"both"`时，工具将比较前反向的diff；当设置为`forward`时，仅比较前向diff，且会跳过模型的backward计算过程。

       -   "compare_mode": `"mean"|"strict"`默认为`"mean"`。`"mean"`表示使用Tensor间误差的均值作为对齐标准；`"strict"`表示对Tensor进行逐数据（Elementwise）的对齐检查。

       -   "single_step": `True|False` 默认为 `False`。设置为`True`开启单步对齐模式，paddle模型中每一个sublayer的input由cpu paddle模型中对应的input对齐，可以避免层间误差累积。注意：开启single_step后将不会触发backward过程，"diff_phase"参数将被强制设置为`"forward"`。

### 注意事项与用例代码：

-   在使用auto_layer_diff时，需要传入paddle模型，在模型定义时，需要将forward中所使用的子模型在`__init__`函数中定义，并保证其中的子模型定义顺序一致，具体可见下方示例代码

```py
from pcdiff import auto_layer_diff
import paddle

# 使用paddle定义模型: SimpleLayer 
# 样例模型结构为:
#       x -> linear1 -> x -> relu -> x -> add -> linear2 -> output
#       |                                  |
#       |----------------------------------|

# 注意：两个模型定义顺序都是 linear1 linear2 ReLU，顺序必须对齐，submodule内部的定义也是一样。
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

layer = SimpleLayer()
inp = paddle.rand((100, 100)).numpy().astype("float32")
inp = paddle.to_tensor(inp)  ## <-- 注意顺序，paddle_input, torch_input 的形式。
auto_layer_diff(layer, inp, auto_weights=True, options={'atol': 1e-4, 'rtol':0,'plat':'npu', 'compare_mode': 'strict', 'single_step':False})
```

## 输出信息示例

-   正确对齐时的输出信息：
    auto_diff将输出paddle与cpu-paddle模型输出结果之间的最大diff值

       ```
       [AutoDiff] Start auto_diff, may need a while to generate reports...
       [AutoDiff] Max output diff is 6.103515625e-05

       [AutoDiff] weight and weight.grad is compared.
       [AutoDiff] forward 4 steps compared.
       [AutoDiff] bacward 4 steps compared.
       [AutoDiff] SUCCESS !!!
       ```

-   模型对齐失败时的输出信息：

       -   训练后，模型权重以及梯度的对齐情况，具体信息将记录在当前路径的diff_log文件夹下
       -   注意，每次调用auto_diff后，diff_log下的报告会被覆盖
       -   在训练过程中首先出现diff的位置（在forward过程或backward过程）
       -   paddle的调用栈，可以追溯到第一次出现不对齐的代码位置

       ```
       [AutoDiff] Start auto_diff, may need a while to generate reports...
       [AutoDiff] Max output diff is 3.0571913719177246

       [AutoDiff] Differences in weight or grad !!!
       [AutoDiff] Check reports at `/workspace/diff_log`

       [AutoDiff] FAILED !!!
       [AutoDiff]     Diff found in `Forward  Stagy` in step: 0, net_id is 1 vs 1
       [AutoDiff]     Type of layer is  : <class 'torch.nn.modules.linear.Linear'> vs <class 'paddle.nn.layer.common.Linear'>


       Paddle Stacks:
       =========================
              ...
              File /workspace/env/env3.7/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py: 1022    __call__
                     return self._dygraph_call_func(*inputs, **kwargs)
              File pptest.py: 37    forward
                     x = self.linear1(x)
              ...
       ```

-   模型对齐失败且失败位置在反向过程时：
    结合输出文本 " [AutoDiff]     Diff found in `Backward Stagy` in step: 0, net_id is 2 vs 2 "，可知模型前向能够对齐，但是反向过程出现diff。结合调用栈信息可以发现，diff出现在linear2对应的反向环节出现diff


       ```
       [AutoDiff] Start auto_diff, may need a while to generate reports...
       [AutoDiff] Max output diff is 1.71661376953125e-05

       [AutoDiff] Differences in weight or grad !!!
       [AutoDiff] Check reports at `/workspace/diff_log`

       [AutoDiff] forward 4 steps compared.
       [AutoDiff] FAILED !!!
       [AutoDiff]     Diff found in `Backward Stagy` in step: 0, net_id is 2 vs 2
       [AutoDiff]     Type of layer is  : <class 'torch.nn.modules.linear.Linear'> vs <class 'paddle.nn.layer.common.Linear'>


       Paddle Stacks:
       =========================
              ...
              File /workspace/env/env3.7/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py: 1022    __call__
                     return self._dygraph_call_func(*inputs, **kwargs)
              File pptest.py: 52    forward
                     x3 = self.linear2(x)
              ...
       ```
## 调试建议

如果遇到了 auto_layer_diff 函数提示某个 layer 没有对齐，可以考虑如下几个 debug 建议：

- 如果不是上述的问题，那么可以考虑进行debug，比如构造最小复现样例或者是pdb调试等等。

- 如果上述无法解决您的问题，或者认为找不到问题，可以考虑给本仓库提一个Issue。
