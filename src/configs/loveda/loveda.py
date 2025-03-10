from albumentations import Compose, OneOf, Normalize
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, RandomCrop
import ever as er

data = dict(
    train=dict(
        type='LoveDALoader',
        params=dict(
            image_dir=[
                './loveda/Train/Urban/images_png/',
                './loveda/Train/Rural/images_png/',
            ],
            mask_dir=[
                './loveda/Train/Urban/masks_png/',
                './loveda/Train/Rural/masks_png/',
            ],
            
            transforms=Compose([
                RandomCrop(512, 512),
                OneOf([
                    HorizontalFlip(True),
                    VerticalFlip(True),
                    RandomRotate90(True)
                ], p=0.75),
                Normalize(mean=(123.675, 116.28, 103.53),
                          std=(58.395, 57.12, 57.375),
                          max_pixel_value=1, always_apply=True),
                er.preprocess.albu.ToTensor()

            ]),
            CV=dict(k=10, i=-1),
            training=True,
            batch_size=16,
            num_workers=2,
            num_classes=7,
        ),
    ),
    test=dict(
        type='LoveDALoader',
        params=dict(
            image_dir=[
                './loveda/Val/Urban/images_png/',
                './loveda/Val/Rural/images_png/',
            ],
            mask_dir=[
                './loveda/Val/Urban/masks_png/',
                './loveda/Val/Rural/masks_png/',
            ],
            transforms=Compose([
                Normalize(mean=(123.675, 116.28, 103.53),
                          std=(58.395, 57.12, 57.375),
                          max_pixel_value=1, always_apply=True),
                er.preprocess.albu.ToTensor()

            ]),
            CV=dict(k=10, i=-1),
            training=False,
            batch_size=4,
            num_workers=0,
            num_classes=7,
        ),
    ),
)
optimizer = dict(
    type='sgd',
    params=dict(
        momentum=0.9,
        weight_decay=0.0001
    ),
    grad_clip=dict(
        max_norm=35,
        norm_type=2,
    )
)
learning_rate = dict(
    type='poly',
    params=dict(
        base_lr=0.01,
        power=0.9,
        max_iters=15000,
    ))
train = dict(
    forward_times=1,
    num_epochs=10,
    eval_per_epoch=True,
    summary_grads=False,
    summary_weights=False,
    distributed=True,
    apex_sync_bn=True,
    sync_bn=True,
    eval_after_train=True,
    log_interval_step=50,
    save_ckpt_interval_epoch=1000,
    eval_interval_epoch=20,
)

test = dict(

)