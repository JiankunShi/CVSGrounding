from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmdet3d.registry import HOOKS
import torch.nn as nn

@HOOKS.register_module()
class FreezeAndEvalHook(Hook):
    def __init__(self, frozen_modules=None):
        self.frozen_modules = frozen_modules or []

    def get_module_by_name(self, model, name: str):
        """递归获取模型中的子模块，支持 'a.b.c' 格式"""
        parts = name.split('.')
        if len(parts) == 1:
            for part in parts:
                model = getattr(model.module, part)
        else:
            model = getattr(model.module, parts[0])
            for part in parts[1:]:
                model = getattr(model, part)
        return model

    def before_train(self, runner):
        model = runner.model
        for name in self.frozen_modules:
            module = self.get_module_by_name(model, name)
            # 设置参数不可训练
            for param in module.parameters():
                param.requires_grad = False
            # 如果模块或其子模块包含 BN，则设为 eval 模式
            module.eval()
            for m in module.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    m.eval()
            print(f'🔒 Froze and eval mode: {name}')

        # # 特定参数层冻结（可选：你原来的特殊命名逻辑）
        # for name, param in model.named_parameters():
        #     if any(keyword in name for keyword in [
        #         'bbox_head.heatmap_head',
        #         'bbox_head.prediction_heads.0.heatmap',
        #         'bbox_head.fusion_layer',
        #         'bbox_head.prediction_heads',
        #         'bbox_head.decoder',
        #         'bbox_head.class_encoding',
        #         'bbox_head.shared_conv'
        #     ]):
        #         param.requires_grad = False
        #         print(f'🔒 Param frozen: {name}')
