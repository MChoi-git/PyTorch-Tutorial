import model
import torch
import dataset


def test():
    PATH = './cifar_net.pth'
    # Test
    net = model.Net()
    net.load_state_dict(torch.load(PATH))

    correct = 0
    total = 0

    # Disables gradient computation. Reduces mem consumption for computations that would otherwise have requires_grad=True
    with torch.no_grad():
        # For each example
        for data in dataset.testloader:
            images, labels = data

            # Get the output vector
            outputs = net(images)

            # Get the predicted class
            _, predicted = torch.max(outputs.data, 1)

            # Add the example to the total
            total += labels.size(0)

            # Check if the predicted label matches the true label
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


if __name__ == "__main__":
    test()