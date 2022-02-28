from timm.data import create_transform, Mixup


def CIFAR100_augmentation(is_training: bool):
    transform = create_transform(
        input_size=32,
        is_training=is_training,
        auto_augment='rand-m2-n1-mstd0.5',
        interpolation='bicubic',
        re_prob=0.5,
        re_mode='pixel',
        re_count=1,
        mean=(0.413, 0.482, 0.446),
        std=(0.247, 0.243, 0.261)
    )
    if is_training:
        mixup = Mixup(
            mixup_alpha=0.8, 
            cutmix_alpha=1.0, 
            prob=1.0, 
            switch_prob=0.5, 
            mode='batch',
            label_smoothing=0.1, 
            num_classes=100
        )
        return transform, mixup
    else:
        return transform