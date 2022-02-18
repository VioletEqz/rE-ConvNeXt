import torch
import torchvision


def train(opt):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    MODEL_NAME = opt['model_name']
    BATCH_SIZE = opt['batch_size']
    EPOCHS = opt['epochs']

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

    # TODO: Define get_model function later
    model = get_model(MODEL_NAME).to(device)

    # Softmax loss
    criterion = torch.nn.CrossEntropyLoss()

    # AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Grad scaler for AMP
    scaler = torch.cuda.amp.GradScaler(init_scale=2**14) # Prevents early overflow

    for _ in range(EPOCHS):
        for batchX, batchY in train_loader:
            with torch.cuda.amp.autocast():
                batchX = batchX.to(device)
                batchY = batchY.to(device)
                pred = model(batchX)
                loss = criterion(pred, batchY)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad(set_to_none=True)
        
        # LR scheduler
        scheduler.step()
    
    return model
        

if __name__ == "__main__":
    options = {
        'model_name': 'resnet50',
        'batch_size': 128,
        'epochs': 50,
    }

    model = train(options)

    torch.save(model.state_dict(), './models/cifar100_resnet50.pt')