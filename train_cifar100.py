import torch
import torchvision
from typing import Any
import time
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet200

def get_model(model_name: str, **kwargs: Any) -> torch.nn.Module:
    if model_name == 'resnet18':
        return resnet18(**kwargs)
    elif model_name == 'resnet34':
        return resnet34(**kwargs)
    elif model_name == 'resnet50':
        return resnet50(**kwargs)
    elif model_name == 'resnet101':
        return resnet101(**kwargs)
    elif model_name == 'resnet200':
        return resnet200(**kwargs)
    else:
        raise ValueError(f'Model {model_name} not supported or doesnt exist!')


def train(opt):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    MODEL_NAME = opt['model_name']
    BATCH_SIZE = opt['batch_size']
    EPOCHS = opt['epochs']
    CLASSES = opt['classes']

    cifar_train = torchvision.datasets.CIFAR100(
        root='./data/cifar100', train=True, transform=None, 
        target_transform=None, download=True)
        
    cifar_test = torchvision.datasets.CIFAR100(
        root='./data/cifar100', train=False, transform=None,
        target_transform=None, download=True)

    train_loader = torch.utils.data.DataLoader(
        cifar_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    test_loader = torch.utils.data.DataLoader(
        cifar_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    #Define model
    model = get_model(MODEL_NAME, num_classes = CLASSES).to(device)

    # Softmax loss
    criterion = torch.nn.CrossEntropyLoss()

    # AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Grad scaler for AMP
    scaler = torch.cuda.amp.GradScaler(init_scale=2**14) # Prevents early overflow

    best_val_loss = float('inf')
    print('Epoch\tTrain Loss\tTrain Acc\tVal Loss\tVal Acc\t\tLearning Rate\tTime')

    for t in range(EPOCHS):
        #Begin timer
        start_time = time.monotonic()
        #Initialize loss and accuracy
        train_loss, train_acc = 0, 0
        val_loss, val_acc = 0, 0

        #Train on train set
        model.train()
        for batchX, batchY in train_loader:
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                batchX = batchX.to(device)
                batchY = batchY.to(device)
                pred = model(batchX)
                loss = criterion(pred, batchY)

            acc = calc_acc(pred, batchY)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad(set_to_none=True)

            train_loss += loss.item()
            train_acc += acc.item()
        
        #Evaluate on test set
        model.eval()
        for batchX, batchY in test_loader:
            with torch.no_grad():
                batchX = batchX.to(device)
                batchY = batchY.to(device)
                pred = model(batchX)
                loss = criterion(pred, batchY)
                acc = calc_acc(pred, batchY)

            val_loss += loss.item()
            val_acc += acc.item()
        
        #End timer
        end_time = time.monotonic()
        elapsed_time = end_time-start_time

        #Check if validation loss is best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch=t
            #Marking the epoch as best
            print('*', end='')
            #torch.save(model.state_dict(),'output/best.pt')

        # LR scheduler
        scheduler.step()
        # Print results
        print('%d\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%f\t%2.2f' %
                (t+1, train_loss, train_acc*100, val_loss, val_acc*100,scheduler.get_last_lr()[0],elapsed_time))
        #Saving model
        #torch.save(model.state_dict(), './output/last.pt')
    return model

def calc_acc(pred, target):
    top = pred.argmax(1, keepdim = True)
    correct = top.eq(target.view_as(top)).sum()
    acc = correct.float() / target.shape[0]
    return acc        

if __name__ == "__main__":
    options = {
        'model_name': 'resnet50',
        'batch_size': 128,
        'epochs': 50,
        'classes': 100,
    }

    model = train(options)
    #Perhaps moving the saving to the train function would be better?
    #This way we can save both the best and last model during the training
    torch.save(model.state_dict(), './output/cifar100_resnet50.pt')