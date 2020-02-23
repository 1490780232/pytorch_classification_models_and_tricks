import torchvision
from PIL import Image
from   torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# from torch.utils.data.dataloader import DataLoader
def get_dataset(root_dir):
    trans=transforms.Compose([
        transforms.ToTensor()
    ])
    cifar_train = torchvision.datasets.CIFAR100(root=root_dir, train=True, download=True,transform=trans)
    # print(cifar_train[1][0].resize((64,64)).save("te.png"))

    cifar_test = torchvision.datasets.CIFAR100(root=root_dir, train=False, download=True,transform=trans)
    return cifar_train,cifar_test