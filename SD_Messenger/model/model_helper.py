import importlib
import torch.nn as nn


class ModelBuilder(nn.Module):
    def __init__(self, net_cfg):
        super(ModelBuilder, self).__init__()

        self.backbone = self._build_backbone(net_cfg["backbone"])
        self.transformer = self._build_transformer(net_cfg["transformer"])

    def _build_backbone(self, backbone_cfg):
        backbone = self._build_module(backbone_cfg["type"], backbone_cfg["kwargs"])
        return backbone

    def _build_transformer(self, trans_cfg):
        transformer = self._build_module(trans_cfg["type"], trans_cfg["kwargs"])
        return transformer

    def _build_module(self, mtype, kwargs):
        module_name, class_name = mtype.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls(**kwargs)

    def forward(self, x):
        h, w = x.shape[-2:]
        c1, c2, c3, c4 = self.backbone(x)
        outs = self.transformer((c1, c2, c3, c4), h, w)
        return outs
