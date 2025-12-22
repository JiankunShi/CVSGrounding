#这个代码是MSSG的配置文件，由bevfusion-lidar-only改写
_base_ = ['./_base_/default_runtime.py']
custom_imports = dict(
    imports=['projects.BEVFusion.bevfusion',
             'mmdet3d.models.backbones.bert_wrapper',
             'projects.BEVFusion.bevfusion.mssgfusion',
             'mmdet3d.models.data_preprocessors.SPNuscenes_data_preprocessor',
             'mmdet3d.engine.hooks.epoch_set_stage_hook'
             # 'mmdet3d.engine.hooks.validate_before_train_hook'
             ], allow_failed_imports=False)

# model settings
# Voxel size for voxel encoder
# Usually voxel size is changed consistently with the point cloud range
# If point cloud range is modified, do remember to change all related
# keys in the config.
voxel_size = [0.075, 0.075, 0.2]
point_cloud_range_minx = -54.0
point_cloud_range_maxx = 54.0
point_cloud_range_miny = -54.0
point_cloud_range_maxy = 54.0
point_cloud_range_minz = -5.0
point_cloud_range_maxz = 3.0
out_size_factor = 8 #这个值是当bev坐标和特征图大小不一致时，进行下采样的比例，bevfusion中是8
point_cloud_range = [point_cloud_range_minx, point_cloud_range_miny, point_cloud_range_minz, point_cloud_range_maxx, point_cloud_range_maxy, point_cloud_range_maxz]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

metainfo = dict(classes=class_names)
dataset_type = 'Talk2car3dDataset'
#dataset_type = 'SPNuscenesDataset'
if dataset_type == 'SPNuscenesDataset':
    train_ann_file_path = 'mmdet_all_train_data_map_caption_IOU03_yaw_egoCentre_multiView6p5.pkl'
    val_ann_file_path = 'mmdet_all_val_data_map_caption_IOU03_yaw_egoCentre_multiView6p5.pkl'
elif dataset_type == 'Talk2car3dDataset':
    train_ann_file_path = 'talk2car_dataset/train_commands_3d_lidarCentre.pkl'
    val_ann_file_path = 'talk2car_dataset/test_commands_3d_lidarCentre.pkl'
data_root = '/data_volume_3/nuscenes/v1_0/'
# data_prefix = dict(
#     pts='samples/LIDAR_TOP',
    # CAM_FRONT='samples/CAM_FRONT',
    # CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    # CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    # CAM_BACK='samples/CAM_BACK',
    # CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
    # CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
    # sweeps='sweeps/LIDAR_TOP')
input_modality = dict(use_lidar=True, use_camera=True)
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/nuscenes/':
#         's3://openmmlab/datasets/detection3d/nuscenes/',
#         'data/nuscenes/':
#         's3://openmmlab/datasets/detection3d/nuscenes/',
#         './data/nuscenes_mini/':
#         's3://openmmlab/datasets/detection3d/nuscenes/',
#         'data/nuscenes_mini/':
#         's3://openmmlab/datasets/detection3d/nuscenes/'
#     }))
backend_args = None

model = dict(
    type='MSSGFusion',
    data_preprocessor=dict(
        type='SPNuscenesDataPreprocessor',
        pad_size_divisor=32,
        voxelize_cfg=dict(
            max_num_points=10,
            point_cloud_range=[point_cloud_range_minx, point_cloud_range_miny, -5.0, point_cloud_range_maxx, point_cloud_range_maxy, 3.0],
            voxel_size=[0.075, 0.075, 0.2],
            max_voxels=[120000, 160000],
            voxelize_reduce=True)),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='BEVFusionSparseEncoder',
        in_channels=5,
        sparse_shape=[1440, 1440, 41],
        # sparse_shape=[int((point_cloud_range_maxx-point_cloud_range_minx)/voxel_size[0]), int((point_cloud_range_maxy-point_cloud_range_miny)/voxel_size[1]), 41],
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, (1, 1, 0)), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        # out_channels=[256, 512], # sjk modify
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        # in_channels=[256, 512], # sjk modify
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    # fusion_PC_Text=dict(
    #         type='ConvFusionPC_Text', in_channels=[768, 256], out_channels=256, use_wsa=True),
    fusion_BEV_Text=dict(
            type='ConvFusionBEV_Text', in_channels=[768, 512], out_channels=512, use_wsa=True),
    img_backbone=dict(
        type='mmdet.SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=[1, 2, 3],
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa: E501
        )),
    img_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[192, 384, 768],
        out_channels=256,
        start_level=0,
        num_outs=3,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        upsample_cfg=dict(mode='bilinear', align_corners=False)),
    view_transform=dict(
        type='DepthLSSTransform',
        in_channels=256,
        out_channels=80,
        image_size=[256, 704],
        feature_size=[32, 88],
        xbound=[-54.0, 54.0, 0.3],
        ybound=[-54.0, 54.0, 0.3],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[1.0, 60.0, 0.5],
        downsample=2),
    fusion_layer=dict(
        type='ConvFuser', in_channels=[80, 256], out_channels=256),
    bbox_head=dict(
        type='MSSGGroundingTransFusionHead',  # 自定义类名，确保你已经在项目中实现了它
        # num_proposals=200,
        # auxiliary=False,
        in_channels=512,
        hidden_channel=128,
        # num_classes=1,  # 只预测被语言指引的一个目标
        # nms_kernel_size=3,
        bn_momentum=0.1,
        lambda_reg = 0.25,
        num_decoder_layers=1,
        gamma=0.4,
        num_conv=2,
        pc_range = [point_cloud_range_minx, point_cloud_range_miny],
        voxel_size = 0.075,
        out_size_factor = out_size_factor,
        # decoder_layer=dict(
        #     type='TransformerDecoderLayer',
        #     self_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.1),
        #     cross_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.1),
        #     ffn_cfg=dict(
        #         embed_dims=128,
        #         feedforward_channels=256,
        #         num_fcs=2,
        #         ffn_drop=0.1,
        #         act_cfg=dict(type='ReLU', inplace=True),
        #     ),
        #     norm_cfg=dict(type='LN'),
        #     pos_encoding_cfg=dict(input_channel=2, num_pos_feats=128)
        # ),

        # 训练配置
        train_cfg=dict(
            dataset='SPNuscenes',
            point_cloud_range=[point_cloud_range_minx, point_cloud_range_miny, -5.0, point_cloud_range_maxx, point_cloud_range_maxy, 3.0],
            grid_size=[1440, 1440, 41],
            # grid_size=[int((point_cloud_range_maxx - point_cloud_range_minx) / voxel_size[0]),
            #            int((point_cloud_range_maxy - point_cloud_range_miny) / voxel_size[1]),
            #               41],
            voxel_size=[0.075, 0.075, 0.2],
            out_size_factor=out_size_factor,
            gaussian_overlap=0.1,
            min_radius=2,
            pos_weight=-1,
            lambda_reg=0.25,  # 加入总损失中的回归损失权重
            # assigner=dict(  # 可保留或简化，取决于你是否用 assigner
            #     type='HungarianAssigner3D',
            #     iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
            #     cls_cost=dict(
            #         type='mmdet.FocalLossCost',
            #         gamma=2.0,
            #         alpha=0.25,
            #         weight=0.15
            #     ),
            #     reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
            #     iou_cost=dict(type='IoU3DCost', weight=0.25)
            # )
        ),

        # 测试配置
        test_cfg=dict(
            dataset='SPNuscenes',
            grid_size=[1440, 1440, 41],
            # grid_size=[int((point_cloud_range_maxx - point_cloud_range_minx) / voxel_size[0]),
            #            int((point_cloud_range_maxy - point_cloud_range_miny) / voxel_size[1]),
            #               41],
            out_size_factor=out_size_factor,
            voxel_size=[0.075, 0.075],
            pc_range=[point_cloud_range_minx, point_cloud_range_miny],
            nms_type=None
        ),

        # 回归输出结构
        common_heads=dict(
            center=[2, 2],
            height=[1, 2],
            dim=[3, 2],
            rot=[2, 2]
            # vel 不用预测可删掉
        ),
        # # 编码器配置
        # bbox_coder=dict(
        #     type='TransFusionBBoxCoder',
        #     pc_range=[point_cloud_range_minx, point_cloud_range_miny],
        #     post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        #     score_threshold=0.0,
        #     out_size_factor=out_size_factor,
        #     voxel_size=[0.075, 0.075],
        #     code_size=10  # 包括 x, y, z, dx, dy, dz, sin(yaw), cos(yaw), optional vel
        # )
    ))

train_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    # dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='ImageAug3D',
        final_dim=[256, 704],
        resize_lim=[0.38, 0.55],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[-5.4, 5.4],
        rand_flip=True,
        is_train=True),
    dict(
        type='GlobalRotScaleTrans',
        scale_ratio_range=[0.9, 1.1],
        rot_range=[-0.78539816, 0.78539816],
        translation_std=0.5),
    dict(
        type='BEVFusionGlobalRotScaleTrans',
        scale_ratio_range=[0.9, 1.1],
        rot_range=[-0.78539816, 0.78539816],
        translation_std=0.5),
    dict(type='BEVFusionRandomFlip3D'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    # Actually, 'GridMask' is not used here
    dict(
        type='GridMask',
        use_h=True,
        use_w=True,
        max_epoch=6,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=1,
        prob=0.0,
        fixed_prob=True),
    dict(type='PointShuffle'),
    dict(
        type='BertTextEmbeddingTransform',
        pretrained='/data_volume_1/sjk_pretrain_model/BERT_model/bert-base-uncased',
        output_cls_token=True,  # 是否输出 [CLS] 向量
        freeze=True  # 是否冻结 BERT 参数
    ),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'nlp_desc_embedding', 'gt_3d_center', 'gt_3d_size', 'sin_yaw', 'cos_yaw', 'img'
        ],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'transformation_3d_flow', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans', 'img_aug_matrix',
            'lidar_aug_matrix', 'num_pts_feats'
        ])
]

test_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(
        type='ImageAug3D',
        final_dim=[256, 704],
        resize_lim=[0.48, 0.48],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=[point_cloud_range_minx, point_cloud_range_miny, -5.0, point_cloud_range_maxx, point_cloud_range_maxy, 3.0]),
    dict(
        type='BertTextEmbeddingTransform',
        pretrained='/data_volume_1/sjk_pretrain_model/BERT_model/bert-base-uncased',
        output_cls_token=True,  # 是否输出 [CLS] 向量
        freeze=True  # 是否冻结 BERT 参数
    ),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'nlp_desc_embedding', 'gt_3d_center', 'gt_3d_size', 'sin_yaw', 'cos_yaw', 'img'
        ],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'num_pts_feats'
        ])
]
batch_size = 6
train_dataloader = dict(
    # batch_size=4,
    # batch_size=3,
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    # sampler=dict(type='DefaultSampler', shuffle=False),#sjk modify
    dataset=dict(
        type=dataset_type,
        data_root='/data_volume_1/sjk_data/NuscenesGrounding/',
        ann_file=train_ann_file_path,
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        test_mode=False,
        # data_prefix=data_prefix,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'))
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root='/data_volume_1/sjk_data/NuscenesGrounding/',
        ann_file=val_ann_file_path,
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        # data_prefix=data_prefix,
        test_mode=True,
        box_type_3d='LiDAR',
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='SPNuscenesMetric',
    prefix='val',
    # data_root=data_root,
    # ann_file=data_root + 'mmdet_all_val_data_map_caption_IOU03_yaw.pkl',
    # metric='bbox',
    # backend_args=backend_args
)
test_evaluator = val_evaluator

# vis_backends = [dict(type='LocalVisBackend')]
vis_backends = [#dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# learning rate
lr = 0.0001
# param_scheduler = [
#     dict(
#         type='ExponentialLR',     # 使用指数衰减
#         gamma=0.9,               # 每个 epoch 学习率乘以 0.9
#         by_epoch=True,            # 每个 epoch 调度
#         begin=0,
#         end=20                    # 总共训练 20 个 epoch
#     )
# ]
param_scheduler = [
    # learning rate scheduler
    # During the first 8 epochs, learning rate increases from 0 to lr * 10
    # during the next 12 epochs, learning rate decreases from lr * 10 to
    # lr * 1e-4
    dict(
        type='CosineAnnealingLR',
        T_max=8,
        eta_min=lr * 10,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=12,
        eta_min=lr * 1e-4,
        begin=8,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True),
    # momentum scheduler
    # During the first 8 epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next 12 epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type='CosineAnnealingMomentum',
        T_max=8,
        eta_min=0.85 / 0.95,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=12,
        eta_min=1,
        begin=8,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=1, val_begin=0)
# train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=1, val_begin=0)
val_cfg = dict(type='ValLoop')
test_cfg = dict()

# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
#     clip_grad=dict(max_norm=35, norm_type=2))
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=lr,
        weight_decay=0.01
    ),clip_grad=dict(max_norm=30, norm_type=2)
)
# param_scheduler = [
#     dict(
#         type='OneCycleLR',
#         max_lr=1e-3,
#         pct_start=0.3,
#         div_factor=25,
#         final_div_factor=1e4,
#         base_momentum=0.85,
#         max_momentum=0.95,
#         by_epoch=False,
#         convert_to_iter_based=True,
#         epoch_length=250,
#         total_epochs=20           #必须加 total_epochs 才能计算 total_steps
#     )
# ]
# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (4 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=4) #当enable设置为True时，学习率自动调整（Auto-scaling Learning Rate）的功能将被启用。这个功能的目的是在批次大小变化时自动调整学习率，以保持训练的稳定性和效果。基本的调整原则通常遵循线性缩放规则，即实际批次大小与参考批次大小的比例被用来调整学习率
log_processor = dict(window_size=50)
find_unused_parameters=True
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=5))#每隔5个epoch保存一下模型
custom_hooks = [dict(type='DisableObjectSampleHook', disable_after_epoch=15),
                dict(type='ValidateBeforeTrainHook',  priority='HIGH'),
                dict(type='EpochStageSetterHook', max_epochs=20)]
