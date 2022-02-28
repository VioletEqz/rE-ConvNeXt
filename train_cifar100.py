import time
import os
import torch
import torchvision

import models

from utils.general import AverageMeter, get_model_factory
from utils.metrics import accuracy
from utils.augmentations import CIFAR100_augmentation

def train(opt):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    MODEL_NAME = opt['model_name']
    BATCH_SIZE = opt['batch_size']
    EPOCHS = opt['epochs']
    CLASSES = opt['classes']

    transform, mixup = CIFAR100_augmentation(is_training=True)
    t_transform, _ = CIFAR100_augmentation(is_training=False)
    
    cifar_train = torchvision.datasets.CIFAR100(
        root='./data/cifar100', train=True, transform=transform, 
        target_transform=None, download=True)
        
    cifar_test = torchvision.datasets.CIFAR100(
        root='./data/cifar100', train=False, transform=t_transform,
        target_transform=None, download=True)

    train_loader = torch.utils.data.DataLoader(
        cifar_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    test_loader = torch.utils.data.DataLoader(
        cifar_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Define model
    model = get_model_factory(models, MODEL_NAME)
    model = model(num_classes=CLASSES).to(device)

    # Softmax loss
    criterion = torch.nn.CrossEntropyLoss()

    # AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Grad scaler for AMP
    scaler = torch.cuda.amp.GradScaler(init_scale=2**14)   # Prevents early overflow

    # Output directory
    output_dir = './output/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # To-do: some more checks for the output directory
    # to support storing multiple results from training.

    # Training
    best_val_loss = float('inf')
    print('Epoch\tTrain Loss\tTrain Acc\tVal Loss\tVal Acc\t\tLearning Rate\tTime')
    

    for t in range(EPOCHS):
        # Begin timer
        start_time = time.monotonic()
        # Initialize loss and accuracy
        train_loss, train_acc = AverageMeter(), AverageMeter()

        # Train on train set
        model.train()
        for batchX, batchY in train_loader:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                batchX = batchX.to(device)
                batchY = batchY.to(device)
                # Get mixup sample
                if mixup is not None:
                    batchX, batchY = mixup(batchX, batchY)
                pred = model(batchX)
                loss = criterion(pred, batchY)

            with torch.no_grad():
                acc = accuracy(pred, batchY)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad(set_to_none=True)

            train_loss.update(loss.item())
            train_acc.update(acc.item())
        
        # Evaluate on test set
        model.eval()
        preds = []
        targets = []
        for batchX, batchY in test_loader:
            with torch.no_grad():
                batchX = batchX.to(device)
                batchY = batchY.to(device)

                with torch.cuda.amp.autocast():
                    pred = model(batchX)
                preds.append(pred)
                targets.append(batchY)
        
        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)
        val_loss = criterion(preds, targets).item()
        val_acc = accuracy(preds, targets).item()
        
        # End timer
        end_time = time.monotonic()
        elapsed_time = end_time - start_time

        # Check if validation loss is best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = t
            # Marking the epoch as best
            print('*', end='')
            # Save the best weight seperately
            torch.save(model.state_dict(), output_dir + '/best.pt')

        # LR scheduler
        scheduler.step()
        # Print results
        print('%d\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%f\t%2.2f' %
                (t+1, train_loss.avg, train_acc.avg*100, val_loss, val_acc*100, scheduler.get_last_lr()[0],elapsed_time))
        # Saving current epoch's weight
        torch.save(model.state_dict(), output_dir + '/last.pt')
    return best_epoch

  

if __name__ == "__main__":
    options = {
        'model_name': 'resnet18',
        'batch_size': 256,
        'epochs': 50,
        'classes': 100,
    }

    best_epoch = train(options)
    print('Best epoch: %d' % best_epoch)