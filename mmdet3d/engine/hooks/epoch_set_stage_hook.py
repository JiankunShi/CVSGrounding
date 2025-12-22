from mmengine.hooks import Hook
from mmdet3d.registry import HOOKS

@HOOKS.register_module()
class EpochStageSetterHook(Hook):
    def __init__(self, max_epochs):
        self.max_epochs = max_epochs

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model

        if hasattr(model, 'set_epoch'):
            model.set_epoch(epoch, self.max_epochs)
        else:
            # 若 set_epoch 是在子模块里（如 head），也可以往下传
            if hasattr(model, 'bbox_head') and hasattr(model.bbox_head, 'set_epoch'):
                model.bbox_head.set_epoch(epoch, self.max_epochs)
