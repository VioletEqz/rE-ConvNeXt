from timm.data import create_transform, Mixup


CIFAR100_MEAN = (0.413, 0.482, 0.446)
CIFAR100_STD = (0.247, 0.243, 0.261)

def get_transform(input_size= 32,
                 is_training= True,
                 color_jitter= 0.4,
                 auto_augment= 'rand-m9-mstd0.5-inc1',
                 train_interpolation= 'bicubic',
                 re_prob= 0.5,
                 re_mode= 'pixel',
                 re_count= 1,
                 mean= CIFAR100_MEAN,
                 std= CIFAR100_STD):
    # Still 50-50 about this one, but it's a good idea to use it.
    # The original paper used timm's implementation of AA, so we will too.
    return create_transform(
                    input_size=input_size,
                    is_training=is_training,
                    color_jitter=color_jitter,
                    auto_augment=auto_augment,
                    train_interpolation=train_interpolation,
                    re_prob=re_prob,
                    re_mode=re_mode,
                    re_count=re_count,
                    mean=mean,
                    std=std)
    )
    

def get_mixup(mixup_alpha= 0.8,
             cutmix_alpha= 1.0, 
             prob= 1.0, 
             switch_prob= 0.5, 
             mode='batch', 
             label_smoothing=0.1, 
             num_classes=100):
    assert mode in ['batch', 'epoch'], 'Mode must be either batch or epoch.'
    
    return Mixup(
                mixup_alpha= mixup_alpha, 
                cutmix_alpha= cutmix_alpha, 
                prob= prob, 
                switch_prob= switch_prob, 
                mode=mode,
                label_smoothing=label_smoothing, 
                num_classes=num_classes)