_base_ = [
    './conf_two_stage_grounding_by_bevfusion_lidar_only.py'
]
voxel_size = [0.075, 0.075, 0.2]
point_cloud_range_minx = -54.0
point_cloud_range_maxx = 54.0
point_cloud_range_miny = -54.0
point_cloud_range_maxy = 54.0
point_cloud_range_minz = -5.0
point_cloud_range_maxz = 3.0
out_size_factor = 8 #这个值是当bev坐标和特征图大小不一致时，进行下采样的比例，bevfusion中是8
point_cloud_range = [point_cloud_range_minx, point_cloud_range_miny, point_cloud_range_minz, point_cloud_range_maxx, point_cloud_range_maxy, point_cloud_range_maxz]
# point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
input_modality = dict(use_lidar=True, use_camera=True)
backend_args = None

model = dict(
    type='TwoStageGrounding',
    fusion_imgs_and_text = False,
    fusion_imgs_text_layer =dict(),
    frozen_modules=[
        'pts_voxel_encoder',
        'pts_middle_encoder',
        'pts_backbone',
        'pts_neck',
        'img_backbone',
        'img_neck',
        'view_transform',
        'fusion_layer',
        'bbox_head.heatmap_head',
        'bbox_head.prediction_heads.0.heatmap',
        # 'bbox_head.fusion_layer',
        'bbox_head.prediction_heads',
        'bbox_head.decoder',
        'bbox_head.class_encoding',
        'bbox_head.shared_conv'
    ],
    data_preprocessor=dict(
        type='SPNuscenesDataPreprocessor',
        pad_size_divisor=32,
        voxelize_cfg=dict(
            max_num_points=10,
            point_cloud_range=[point_cloud_range_minx, point_cloud_range_miny, -5.0, point_cloud_range_maxx,
                               point_cloud_range_maxy, 3.0],
            voxel_size=[0.075, 0.075, 0.2],
            max_voxels=[120000, 160000],
            voxelize_reduce=True)),
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
            # '/data_volume_1/sjk_pretrain_model/bevfusionModel/swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth'
            # '/data_volume_1/sjk_pretrain_model/groundingDINO/groundingdino_swinb_cogcoor.pth'
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
        type='ConvFuser', in_channels=[80, 256], out_channels=256))

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
    dict(
        type='ImageAug3D',
        final_dim=[256, 704],
        resize_lim=[0.38, 0.55],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[-5.4, 5.4],
        rand_flip=True,
        is_train=True),
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
        max_epoch=20,
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
            'points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes',
            'gt_labels'
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
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
    dict(
        type='BertTextEmbeddingTransform',
        pretrained='/data_volume_1/sjk_pretrain_model/BERT_model/bert-base-uncased',
        output_cls_token=True,  # 是否输出 [CLS] 向量
        freeze=True  # 是否冻结 BERT 参数
    ),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'nlp_desc_embedding', 'gt_3d_center', 'gt_3d_size', 'sin_yaw', 'cos_yaw', 'img', 'camera_rec_desc_embedding'
        ],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'num_pts_feats', 'category_name'
        ])
]

# train_dataloader = dict(
#     dataset=dict(
#         # dataset=dict(pipeline=train_pipeline, modality=input_modality)))
#         dataset=dict(pipeline=test_pipeline, modality=input_modality)))
train_dataloader = dict(
        batch_size=8,
        # dataset=dict(pipeline=train_pipeline, modality=input_modality)))
        dataset=dict(pipeline=test_pipeline, modality=input_modality))
val_dataloader = dict(
    batch_size=8,
    dataset=dict(pipeline=test_pipeline, modality=input_modality))
test_dataloader = val_dataloader

param_scheduler = [
    dict(
        type='ExponentialLR',     # 使用指数衰减
        gamma=0.9,               # 每个 epoch 学习率乘以 0.9
        by_epoch=True,            # 每个 epoch 调度
        begin=0,
        end=20                    # 总共训练 20 个 epoch
    )
]

# param_scheduler = [
#     dict(
#         type='LinearLR',
#         start_factor=0.33333333,
#         by_epoch=False,
#         begin=0,
#         end=500),
#     dict(
#         type='CosineAnnealingLR',
#         begin=0,
#         T_max=20,
#         end=20,
#         by_epoch=True,
#         eta_min_ratio=1e-4,
#         convert_to_iter_based=True),
#     # momentum scheduler
#     # During the first 8 epochs, momentum increases from 1 to 0.85 / 0.95
#     # during the next 12 epochs, momentum increases from 0.85 / 0.95 to 1
#     dict(
#         type='CosineAnnealingMomentum',
#         eta_min=0.85 / 0.95,
#         begin=0,
#         end=2.4,
#         by_epoch=True,
#         convert_to_iter_based=True),
#     dict(
#         type='CosineAnnealingMomentum',
#         eta_min=1,
#         begin=2.4,
#         end=20,
#         by_epoch=True,
#         convert_to_iter_based=True)
# ]

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=1)
val_cfg = dict()
test_cfg = dict()
find_unused_parameters=True
optim_wrapper = dict(
    type='OptimWrapper',
    # optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.01),
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.015), #sjk modify
    # clip_grad=dict(max_norm=35, norm_type=2))
    clip_grad=dict(max_norm=30, norm_type=2))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (4 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=32)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=5))#每隔5个epoch保存一下模型
custom_hooks = [dict(type='DisableObjectSampleHook', disable_after_epoch=15),
                dict(type='ValidateBeforeTrainHook',  priority='HIGH'),
                dict(type='EpochStageSetterHook', max_epochs=20),
                dict(
                        type='FreezeAndEvalHook',
                        frozen_modules=[
                            'pts_voxel_encoder',
                            'pts_middle_encoder',
                            'pts_backbone',
                            'pts_neck',
                            'img_backbone',
                            'img_neck',
                            'view_transform',
                            'fusion_layer',
                            'bbox_head.heatmap_head',
                            'bbox_head.prediction_heads.0.heatmap',
                            # 'bbox_head.fusion_layer',
                            'bbox_head.prediction_heads',
                            'bbox_head.decoder',
                            'bbox_head.class_encoding',
                            'bbox_head.shared_conv'
                        ]
                    )
                ]

del _base_.custom_hooks
