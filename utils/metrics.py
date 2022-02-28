def accuracy(pred, target):
    top = pred.argmax(1)

    # If soft target
    if target.ndim == 2 and target.shape[1] > 1:
        target = target.argmax(1)

    return (top == target).float().mean()