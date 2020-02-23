import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from evaluate.evaluate import evaluate
from torch.backends import cudnn
from dataset.cifar100 import get_dataset
from models.resnet import resnet50
from torch import optim
from utils.ad_lr import adjust_learning_rate
from config.config import *
from torch import nn
lr=LR
epoch=EPOCHS
use_cuda=torch.cuda.is_available()
device = torch.device('cuda' if use_cuda==True else "cpu")
model=resnet50(pretrained=True,num_classes=100)
cifar_train,cifar_test=get_dataset("./dataset/data")
train_dataloader = DataLoader(cifar_train, batch_size=32, shuffle=True, num_workers=4)
test_dataloader=DataLoader(cifar_test, batch_size=32, shuffle=False, num_workers=4)
optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.01, momentum=0.9)
criteon = nn.CrossEntropyLoss()
cudnn.benchmark = True
# model.load_state_dict(torch.load('./model/44.mdl'))
# for i,(x,y) in enumerate(tqdm(train_dataloader)):
#     print(y)
model.to(device)
for i in range(epoch):
    running_corrects=0
    running_loss=0
    for step,(x,y) in enumerate(tqdm(train_dataloader)):
        x,y=x.to(device),y.to(device)
        model.train()
        logits=model(x)
        loss=criteon(logits,y)
        running_loss+=loss.item()*x.size(0)
        pred=logits.argmax(dim=1)
        running_corrects += torch.eq(pred, y).sum().float().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch == 60:
        lr = 0.1
        adjust_learning_rate(optimizer, lr)
    if epoch == 60:
        lr = 0.02
        adjust_learning_rate(optimizer, lr)
    if epoch == 40:
        lr = 0.004
        adjust_learning_rate(optimizer, lr)
    if epoch == 40:
        lr = 0.0008
        adjust_learning_rate(optimizer, lr)
    print("corrects:",running_corrects)
    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        i, running_loss / len(train_dataloader.dataset), running_corrects / len(train_dataloader.dataset)))
    evaluate(model,test_dataloader,use_cuda)
