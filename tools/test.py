# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import os.path as osp

from mmengine.config import Config, ConfigDict, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet3d.utils import replace_ceph_backend
import torch
import time

# print("当前使用的 GPU 编号是:", torch.cuda.current_device())
# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet3D test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    # parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--ceph', action='store_true', help='Use ceph as data storage backend')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--score-thr', type=float, default=0.1, help='bbox score threshold')
    parser.add_argument(
        '--task',
        type=str,
        choices=[
            'mono_det', 'multi-view_det', 'lidar_det', 'lidar_seg',
            'multi-modality_det'
        ],
        help='Determine the visualization method depending on the task.')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used configs, the key-value pair '
        'in xxx=yyy format will be merged into configs file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--tta', action='store_true', help='Test time augmentation')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/test.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualization_hook['test_out_dir'] = args.show_dir
        all_task_choices = [
            'mono_det', 'multi-view_det', 'lidar_det', 'lidar_seg',
            'multi-modality_det'
        ]
        assert args.task in all_task_choices, 'You must set '\
            f"'--task' in {all_task_choices} in the command " \
            'if you want to use visualization hook'
        visualization_hook['vis_task'] = args.task
        visualization_hook['score_thr'] = args.score_thr
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg


def main():
    args = parse_args()

    # load configs
    cfg = Config.fromfile(args.config)

    # TODO: We will unify the ceph support approach with other OpenMMLab repos
    if args.ceph:
        cfg = replace_ceph_backend(cfg)

    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use configs filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:
        # Currently, we only support tta for 3D segmentation
        # TODO: Support tta for 3D detection
        assert 'tta_model' in cfg, 'Cannot find ``tta_model`` in configs.'
        assert 'tta_pipeline' in cfg, 'Cannot find ``tta_pipeline`` in configs.'
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)

    # build the runner from configs
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # groundingDINOSwinTPathName = "/data_volumn/model_sjk/groundingDINOModel/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth"
###将groundingDINO的预训练模型存Swin-T储到bevfusion里
    # state_dict = torch.load(groundingDINOSwinTPathName)
    # state_dict_bevfusion = torch.load('/data_volume_1/sjk_pretrain_model/bevfusionModel/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth')
    # runner.model.load_state_dict(state_dict_bevfusion['state_dict'])
    # runner_model_img_backbone_state_dict = {k[len("backbone."):]: v for k, v in state_dict['state_dict'].items() if k.startswith("backbone.")}
    # runner.model.img_backbone.load_state_dict(runner_model_img_backbone_state_dict)
    # torch.save(runner.model.state_dict(), '/data_volume_1/sjk_pretrain_model/bevfusionModel/bevfusion_lidar-cam_grounding_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth')
    # print('/data_volumn/model_sjk/groundingDINOModel/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth model save finish!')
###将groundingDINO的预训练模型Swin-T存储到/data_volume_1/sjk_pretrain_model/bevfusionModel/swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth，为finetune做准备
    # state_dict = torch.load(groundingDINOSwinTPathName)
    # runner_model_img_backbone_state_dict = {k[len("backbone."):]: v for k, v in state_dict['state_dict'].items() if k.startswith("backbone.")}
    # runner.model.img_backbone.load_state_dict(runner_model_img_backbone_state_dict)
    # torch.save(runner.model.img_backbone.state_dict(), '/data_volume_1/sjk_pretrain_model/bevfusionModel/swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth')
    # print('/data_volu del_sjk/groundingDINOModel/swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth model save finish!')

    # start testing
    runner.test()


if __name__ == '__main__':
    st_time = time.time()
    main()
    print(f"总花费时间为{time.time()-st_time}")
