import importlib

import torch.nn as nn
from torch.nn import functional as F

class ModelBuilder(nn.Module):
    def __init__(self, net_cfg):
        super(ModelBuilder, self).__init__()
        # self.model = self._build_model(net_cfg["model"])
        self.backbone = self._build_backbone(net_cfg["backbone"])
        self.framework = self._build_framework(net_cfg["framework"])

    def _build_backbone(self, backbone_cfg):
        backbone = self._build_module(backbone_cfg["type"], backbone_cfg["kwargs"])
        return backbone

    def _build_framework(self, framework_cfg):
        framework = self._build_module(framework_cfg["type"], framework_cfg["kwargs"])
        return framework

    def _build_module(self, mtype, kwargs):
        module_name, class_name = mtype.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls(**kwargs)

    def forward(self, x):
        h, w = x.shape[-2:]
        c1, c2, c3, c4 = self.backbone(x)
        outs = self.framework((c1, c2, c3, c4), h, w)
        return outs
