def accuracy(pred, target):
    top = pred.argmax(1, keepdim = True)
    correct = top.eq(target.view_as(top)).sum()
    acc = correct.float() / target.shape[0]
    return acc      