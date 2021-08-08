# Learn and normalize CIFAR 10
import torch
import torchvision
import torchvision.transforms as transforms


# Compose two transformations: PIL Image to tensor, and Normalize the PIL Images to range [-1, 1] (mean, std)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Define the batch size
batch_size = 4

# Download the training set. Create dataset from the training set, and download. Apply pre-defined transform (tensor and norm)
# to the the training set
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

# Load the dataset defined above. Batch size is pre-defined above. Shuffle the data every epoch. Use two subprocesses for
# data loading.
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

# Download the test set. Same parameters as train_set but for the test set
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)

# Load the testset defined above. Same parameters as the trainloader
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

# Define the classes
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')











