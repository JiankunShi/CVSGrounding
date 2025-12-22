#为了调试，一开始就进行验证
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmdet3d.registry import HOOKS

@HOOKS.register_module()
class ValidateBeforeTrainHook(Hook):
  pass
    #
    # def before_train(self, runner: Runner):
    #     runner.logger.info('Running validation before training starts...')
    #     results = runner.val_loop.run()  # 调用验证循环
    #     runner.logger.info(f'Initial validation results: {results}')
