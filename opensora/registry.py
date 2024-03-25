from copy import deepcopy

import torch.nn as nn
from mmengine.registry import Registry


def build_module(module, builder, **kwargs):
    """Build module from config or return the module itself.

    Args:
        module (Union[dict, nn.Module]): The module to build.
        builder (Registry): The registry to build module.
        *args, **kwargs: Arguments passed to build function.

    Returns:
        Any: The built module.
    """
    if isinstance(module, dict):
        cfg = deepcopy(module)  # 拷贝模型的配置
        for k, v in kwargs.items():
            cfg[k] = v  # 合并输入解析器参数的配置
        return builder.build(cfg)  # 从cfg.vae.type的名称，映射/构建注册器对应注册的模块
    elif isinstance(module, nn.Module):
        return module
    elif module is None:
        return None
    else:
        raise TypeError(f"Only support dict and nn.Module, but got {type(module)}.")


MODELS = Registry(  # 注册模型
    "model",
    locations=["opensora.models"],
)

SCHEDULERS = Registry(  # 注册采样器
    "scheduler",
    locations=["opensora.schedulers"],
)
