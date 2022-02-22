import torchvision
from timm.data import Mixup
def get_transform():
    # Maybe use timm's implementation instead of torchvision ?
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandAugment(magnitude=9), # No further info on this one
        torchvision.transforms.RandomErasing(p = 0.25),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.413, 0.482, 0.446),
                                               std=(0.247, 0.243, 0.261))
    ])
    return transform

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