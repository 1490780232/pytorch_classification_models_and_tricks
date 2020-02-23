import torch
def evaluate(model, test_loader,use_gpu):
    device=torch.device("cuda" if use_gpu else "cpu")
    model.eval()
    correct = 0
    total = len(test_loader.dataset)

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()
    print("test:ACC",correct / total)
    return