from cv2 import transform
from timm.data import create_transform, Mixup

def CIFAR100_augmentation(
                input_size= 32,
                is_training= True,
                color_jitter= 0.4,
                auto_augment= 'rand-m9-mstd0.5-inc1',
                train_interpolation= 'bicubic',
                re_prob= 0.5,
                re_mode= 'pixel',
                re_count= 1,
                mixup_alpha= 0.8,
                cutmix_alpha= 1.0, 
                prob= 1.0, 
                switch_prob= 0.5, 
                mode='batch', 
                label_smoothing=0.1, 
                num_classes=100
                ):
    assert mode in ['batch','pair','elem'], 'mode must be batch, pair or elem'
    transform = create_transform(
                    input_size=input_size,
                    is_training=is_training,
                    color_jitter=color_jitter,
                    auto_augment=auto_augment,
                    train_interpolation=train_interpolation,
                    re_prob=re_prob,
                    re_mode=re_mode,
                    re_count=re_count,
                    mean=(0.413, 0.482, 0.446),
                    std=(0.247, 0.243, 0.261),
                )
    mixup = Mixup(
                mixup_alpha=mixup_alpha, 
                cutmix_alpha=cutmix_alpha, 
                prob=prob, 
                switch_prob=switch_prob, 
                mode=mode,
                label_smoothing=label_smoothing, 
                num_classes=num_classes
                )
    return transform, mixup