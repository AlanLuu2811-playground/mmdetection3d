_base_ = [
    '../_base_/datasets/adam-mono3d.py', '../_base_/models/fcos3d.py',
    '../_base_/schedules/schedule-3x.py', '../_base_/default_runtime.py'
]


# model settings
model = dict(
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    bbox_head=dict(
        num_classes=1))

backend_args = None

albu_train_transforms = [
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[-0.3, 0.3],
        contrast_limit=[-0.3, 0.3],
        brightness_by_max=True,
        ensure_safe_range=True,
        p=0.6),
    dict(
        type='MotionBlur',
        blur_limit=[5, 8],
        allow_shifted=False,
        angle_range=[0, 0],
        direction_range=[0, 0],
        p=0.5),
]

train_pipeline = [
    dict(type='LoadImageFromFileMono3D', backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox=True,
        with_label=True,
        with_attr_label=True,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True),
    dict(type='mmdet.Resize', scale=(640, 400), keep_ratio=True),
    dict(
        type='mmdet.Albu',
        transforms=albu_train_transforms,
        keymap={
            'img': 'image',
            'images': 'images_info'
        }),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'img', 'gt_bboxes', 'gt_bboxes_labels', 'attr_labels',
            'gt_bboxes_3d', 'gt_labels_3d', 'centers_2d', 'depths'
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFileMono3D', backend_args=backend_args),
    dict(type='mmdet.Resize', scale_factor=1.0),
    dict(type='Pack3DDetInputs', keys=['img'])
]

train_dataloader = dict(
    batch_size=8, num_workers=2, dataset=dict(pipeline=train_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))

# optimizer
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(lr=1e-4, type='AdamW', weight_decay=0.01),
    type='OptimWrapper')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

train_cfg = dict(val_interval=1)

default_hooks = dict(
    checkpoint=dict(interval=2, max_keep_ckpts=1, save_best='NuScenes metric/pred_instances_3d_NuScenes/alcon_cart_AP_dist_0.5', rule='greater'))
