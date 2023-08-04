_base_ = [
    '../coco/cascade_internimage_l_fpn_3x_coco.py',
]

model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0))
        ],
        mask_head=dict(
            num_classes=1),
    )
)

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('blood_vessel',)
data_root='/internimage/detection/hubmap_hv'
fold = 0

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        data_root=data_root,
        img_prefix='hubmap-hacking-the-human-vasculature/train',
        classes=classes,
        ann_file=f'repos/coco_label/bv_only_dataset12_5fold/train_fold{fold}.json',
        ),
    val=dict(
        data_root=data_root,
        img_prefix='hubmap-hacking-the-human-vasculature/train',
        classes=classes,
        ann_file=f'repos/coco_label/bv_only_dataset12_5fold/val_fold{fold}.json',
        # pipeline=test_pipeline,
        ),
    test=dict(
        data_root=data_root,
        img_prefix='hubmap-hacking-the-human-vasculature/train',
        classes=classes,
        ann_file=f'repos/coco_label/bv_only_dataset12_5fold/val_fold{fold}.json',
        # pipeline=test_pipeline,
    )
)

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = f'{data_root}/pretrained/intern/cascade_internimage_l_fpn_3x_coco.pth'