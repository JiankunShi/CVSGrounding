#这个代码是MSSG的配置文件，由bevfusion-lidar-only改写
_base_ = ['./_base_/default_runtime.py']
custom_imports = dict(
    imports=['projects.BEVFusion.bevfusion',
             'mmdet3d.models.backbones.bert_wrapper',
             'mmdet3d.models.backbones.grounding_dino_pretrain_backbone',
             'projects.BEVFusion.bevfusion.two_stage_grounding',
             'mmdet3d.models.data_preprocessors.SPNuscenes_data_preprocessor',
             'mmdet3d.engine.hooks.epoch_set_stage_hook',
             'projects.BEVFusion.bevfusion.two_stage_grounding_head'
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
dataset_type = 'SPNuscenesDataset'
# dataset_type = 'Talk2car3dDataset'
if dataset_type == 'SPNuscenesDataset':
    # train_ann_file_path = 'mmdet_all_train_data_map_caption_IOU03_yaw_egoCentre_multiView6p5.pkl'
    # val_ann_file_path = 'mmdet_all_val_data_map_caption_IOU03_yaw_egoCentre_multiView6p5.pkl'
    # train_ann_file_path = 'mmdet_all_train_data_map_caption_IOU03_yaw_egoCentre_multiView30_pro0P5.pkl'
    # val_ann_file_path = 'mmdet_all_val_data_map_caption_IOU03_yaw_egoCentre_multiView30_pro0p5.pkl'
    train_ann_file_path = 'mmdet_all_train_data_map_caption_IOU03_yaw_egoCentre_multiView30_pro0P5_concatDesc_lidar2CamIns2_addBehavior2_addDiffDimDesc.pkl'
    val_ann_file_path = 'mmdet_all_val_data_map_caption_IOU03_yaw_egoCentre_multiView30_pro0P5_concatDesc_lidar2CamIns2_addBehavior2_addDiffDimDesc.pkl'
elif dataset_type == 'Talk2car3dDataset':
    train_ann_file_path = 'talk2car_dataset/train_commands_3d_lidarCentre_lidar2CamIns2_addCategoryName.pkl'
    val_ann_file_path = 'talk2car_dataset/test_commands_3d_lidarCentre_lidar2CamIns2_addCategoryName.pkl'
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
find_unused_parameters=True  # 不设置这个，eval参数会报错
model = dict(
    type='FusionGroundingDetector',
    use_direction_select_img_flag = True,
    frozen_modules=[
                    'pts_voxel_encoder',
                    'pts_middle_encoder',
                    'pts_backbone',
                    'pts_neck',
                    'bbox_head.heatmap_head',
                    'bbox_head.prediction_heads.0.heatmap',
                    # 'bbox_head.fusion_layer',
                    'bbox_head.prediction_heads',
                    'bbox_head.decoder',
                    'bbox_head.class_encoding',
                    'bbox_head.shared_conv'
                ],
    data_preprocessor=dict(
        # type='SPNuscenesDetectionGroundingDataPreprocessor',
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
    gdino_ckpt = '/data_volume_1/sjk_pretrain_model/bevfusionModel/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth',
    gdino_encoder = dict(                      # ← 全贴进去
        _scope_='mmdet',                  # ★ 加这一行
        type='GroundingDINO',
            num_queries=900,
            with_box_refine=True,
            as_two_stage=True,
            data_preprocessor=dict(
                type='DetDataPreprocessor',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                bgr_to_rgb=True,
                pad_mask=False,
            ),
            language_model=dict(
                    type='BertModel',
                    name='/data_volume_1/sjk_pretrain_model/BERT_model/bert-base-uncased',
                    max_tokens=256,
                    pad_to_max=False,
                    use_sub_sentence_represent=True,
                    special_tokens_list=['[CLS]', '[SEP]', '.', '?'],
                    add_pooling_layer=False,
                ),
            backbone=dict(
                type='SwinTransformer',
                embed_dims=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.2,
                patch_norm=True,
                out_indices=(1, 2, 3),
                with_cp=True,
                convert_weights=True,
                frozen_stages=-1,
                init_cfg=dict(type='Pretrained', checkpoint='/data_volume_1/sjk_pretrain_model/bevfusionModel/swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth')),
            neck=dict(
                type='ChannelMapper',
                in_channels=[192, 384, 768],
                kernel_size=1,
                out_channels=256,
                act_cfg=None,
                bias=True,
                norm_cfg=dict(type='GN', num_groups=32),
                num_outs=4),
            encoder=dict(
                num_layers=6,
                num_cp=6,
                # visual layer config
                layer_cfg=dict(
                    self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
                    ffn_cfg=dict(
                        embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
                # text layer config
                text_layer_cfg=dict(
                    self_attn_cfg=dict(num_heads=4, embed_dims=256, dropout=0.0),
                    ffn_cfg=dict(
                        embed_dims=256, feedforward_channels=1024, ffn_drop=0.0)),
                # fusion layer config
                fusion_layer_cfg=dict(
                    v_dim=256,
                    l_dim=256,
                    embed_dim=1024,
                    num_heads=4,
                    init_values=1e-4),
            ),
            decoder=dict(
                num_layers=6,
                return_intermediate=True,
                layer_cfg=dict(
                    # query self attention layer
                    self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
                    # cross attention layer query to text
                    cross_attn_text_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
                    # cross attention layer query to image
                    cross_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
                    ffn_cfg=dict(
                        embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
                post_norm_cfg=None),
            positional_encoding=dict(
                num_feats=128, normalize=True, offset=0.0, temperature=20),
            bbox_head=dict(
                type='GroundingDINOHead',
                num_classes=256,
                sync_cls_avg_factor=True,
                contrastive_cfg=dict(max_text_len=256, log_scale='auto', bias=True),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),  # 2.0 in DeformDETR
                loss_bbox=dict(type='L1Loss', loss_weight=5.0)),
            dn_cfg=dict(  # TODO: Move to model.train_cfg ?
                label_noise_scale=0.5,
                box_noise_scale=1.0,  # 0.4 for DN-DETR
                group_cfg=dict(dynamic=True, num_groups=None,
                               num_dn_queries=100)),  # TODO: half num_dn_queries
            # training and testing settings
            train_cfg=dict(
                assigner=dict(
                    type='HungarianAssigner',
                    match_costs=[
                        dict(type='BinaryFocalLossCost', weight=2.0),
                        dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                        dict(type='IoUCost', iou_mode='giou', weight=2.0)
                    ])),
            test_cfg=dict(max_per_img=300),
            init_cfg=dict(
                    type='Pretrained',
                    checkpoint='/data_volume_1/sjk_pretrain_model/bevfusionModel/'
                               'grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth'
            )
    ),
    # fusion_PC_Text=dict(
    #         type='ConvFusionPC_Text', in_channels=[768, 512], out_channels=512, use_wsa=True),
    bbox_head=dict(
        type='FusionGroundingDetectorHead',
        text_dim = 768,
        fusion_method = 'cross_attention',   # 'concat' | 'cross_attention'
        fusion_hidden_dim = 256,
        num_proposals=200,
        topk = 128,
        iou_thr = 0.1, #只有IOU大于0.3的才算，防止产生伪标签
        # num_proposals=300, # sjk modify
        auxiliary=True,
        in_channels=512,
        hidden_channel=128,
        num_classes=10,
        nms_kernel_size=3,
        bn_momentum=0.1,
        num_decoder_layers=1,
        # num_decoder_layers=2, #sjk modify
        decoder_layer=dict(
            type='TransformerDecoderLayer',
            self_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.1),
            cross_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.1),
            ffn_cfg=dict(
                embed_dims=128,
                feedforward_channels=256,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True),
            ),
            norm_cfg=dict(type='LN'),
            pos_encoding_cfg=dict(input_channel=2, num_pos_feats=128)),
        train_cfg=dict(
            dataset='SPNuscenes',
            point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
            grid_size=[1440, 1440, 41],
            voxel_size=[0.075, 0.075, 0.2],
            out_size_factor=8,
            gaussian_overlap=0.1,
            min_radius=2,
            pos_weight=-1,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            assigner=dict(
                type='HungarianAssigner3D',
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(
                    type='mmdet.FocalLossCost',
                    gamma=2.0,
                    alpha=0.25,
                    weight=0.15),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                iou_cost=dict(type='IoU3DCost', weight=0.25))),
        test_cfg=dict(
            dataset='SPNuscenes',
            grid_size=[1440, 1440, 41],
            out_size_factor=8,
            voxel_size=[0.075, 0.075],
            pc_range=[-54.0, -54.0],
            nms_type=None),
        common_heads=dict(
            center=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2], vel=[2, 2]),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=[-54.0, -54.0],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=0.0,
            out_size_factor=8,
            voxel_size=[0.075, 0.075],
            code_size=10),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=1.0),
        loss_heatmap=dict(
            type='mmdet.GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.L1Loss', reduction='mean', loss_weight=0.25)
    ))

train_pipeline = [
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
        type='GlobalRotScaleTrans',
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
    dict(type='PointShuffle'),
    dict(
        type='BertTextEmbeddingTransform',
        pretrained='/data_volume_1/sjk_pretrain_model/BERT_model/bert-base-uncased',
        output_cls_token=True,  # 是否输出 [CLS] 向量
        freeze=True  # 是否冻结 BERT 参数
    ),
    # dict(type='GroundingDINOTextEmbeddingTransform',
    #      ckpt_path='/data_volume_1/sjk_pretrain_model/bevfusionModel/process_grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth'),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'nlp_desc_embedding', 'gt_3d_center', 'gt_3d_size' #, 'img'
            'sin_yaw', 'cos_yaw'
        ],
        meta_keys=[
            'box_type_3d', 'sample_idx', 'lidar_path',
            'pcd_rotation', 'pcd_scale_factor', 'pcd_trans', 'lidar_aug_matrix'
        ])
]

test_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        # to_float32=True,
        to_float32=False,#sjk modify
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
    # dict(
    #     type='ImageAug3D',
    #     final_dim=[256, 704],
    #     resize_lim=[0.48, 0.48],
    #     bot_pct_lim=[0.0, 0.0],
    #     rot_lim=[0.0, 0.0],
    #     rand_flip=False,
    #     is_train=False),
    dict(
        _scope_='mmdet',
        type='FixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'),
    # dict(type='ImageAug3D',
    #  final_dim=[800, 1333],
    #  resize_lim=[1.0, 1.0],               # 固定比例
    #  bot_pct_lim=[0.0, 0.0],
    #  rot_lim=[0.0, 0.0],
    #  rand_flip=False,
    #  is_train=False),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=[point_cloud_range_minx, point_cloud_range_miny, -5.0, point_cloud_range_maxx, point_cloud_range_maxy, 3.0]),
    dict(
        type='BertTextEmbeddingTransform',
        pretrained='/data_volume_1/sjk_pretrain_model/BERT_model/bert-base-uncased',
        output_cls_token=True,  # 是否输出 [CLS] 向量
        freeze=True  # 是否冻结 BERT 参数
    ),
    # dict(_scope_='mmdet',type='LoadAnnotations', with_bbox=True),
    # dict(
    #     _scope_='mmdet',
    #     type='PackDetInputs',
    #     meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
    #                'scale_factor', 'text', 'custom_entities',
    #                'tokens_positive')),
    # dict(type='GroundingDINOTextEmbeddingTransform',
    #      ckpt_path='/data_volume_1/sjk_pretrain_model/bevfusionModel/process_grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth'),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'nlp_desc_embedding', 'gt_3d_center', 'gt_3d_size', 'sin_yaw', 'cos_yaw' , 'img','camera_rec_desc_embedding'
        ],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar','scale_factor',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'num_pts_feats', 'camera_rec_desc', 'lidar2CamIns',
            'proj_bbox', 'direction_desc_num', 'nlp_desc', 'bbox_token', 'category_name'
        ])
]

train_dataloader = dict(
    # batch_size=4,
    # batch_size=3,
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    #GPT说如果在多卡下使用 DefaultSampler：所有 GPU 会加载同样的数据 → 重复训练，shuffle=True 也不会保证多卡之间同步 → 很容易死锁或训练无效，要用DistributedSampler
    # sampler=dict(type='DistributedSampler', shuffle=True),#sjk modify 这个会报错DistributedSampler is not in the mmdet3d::data sampler registry.
    dataset=dict(
        type=dataset_type,
        data_root='/data_volume_1/sjk_data/NuscenesGrounding/',
        ann_file=train_ann_file_path,
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        test_mode=False,
        # data_prefix=data_prefix,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'))
val_dataloader = dict(
    batch_size=16,
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
    # data_root='/data_volume_1/sjk_data/NuscenesGrounding/',
    # ann_file='/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_val_data_map_caption_IOU03_yaw_egoCentre_multiView6p5.pkl',
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
lr = 0.001
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
#     # learning rate scheduler
#     # During the first 8 epochs, learning rate increases from 0 to lr * 10
#     # during the next 12 epochs, learning rate decreases from lr * 10 to
#     # lr * 1e-4
#     dict(
#         type='CosineAnnealingLR',
#         T_max=8,
#         eta_min=lr * 10,
#         begin=0,
#         end=8,
#         by_epoch=True,
#         convert_to_iter_based=True),
#     dict(
#         type='CosineAnnealingLR',
#         T_max=12,
#         eta_min=lr * 1e-4,
#         begin=8,
#         end=20,
#         by_epoch=True,
#         convert_to_iter_based=True),
#     # momentum scheduler
#     # During the first 8 epochs, momentum increases from 0 to 0.85 / 0.95
#     # during the next 12 epochs, momentum increases from 0.85 / 0.95 to 1
#     dict(
#         type='CosineAnnealingMomentum',
#         T_max=8,
#         eta_min=0.85 / 0.95,
#         begin=0,
#         end=8,
#         by_epoch=True,
#         convert_to_iter_based=True),
#     dict(
#         type='CosineAnnealingMomentum',
#         T_max=12,
#         eta_min=1,
#         begin=8,
#         end=20,
#         by_epoch=True,
#         convert_to_iter_based=True)
# ]
ddp_wrapper=dict(find_unused_parameters=True)
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
        lr=1e-4,
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

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1))#每隔1个epoch保存一下模型
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
