import torch
import torch.optim as optim
import model
import dataset
import time
import test


def main():
    net = model.Net()
    # Define the loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Define the optimizer, ie. SGD w/ momentum in this case
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train
    for epoch in range(3):

        # Track the loss
        running_loss = 0.0

        # For each example
        for i, data in enumerate(dataset.trainloader, 0):

            # Get the example and its label
            inputs, labels = data

            # Zero the gradient
            optimizer.zero_grad()

            # Send example through the net defined in model
            outputs = net(inputs)

            # Get the loss from our criterion defined above
            loss = criterion(outputs, labels)

            # Compute the backward computation
            loss.backward()

            # Update parameters
            optimizer.step()

            # Update the loss
            running_loss += loss.item()

            # Print the loss every 2000 examples
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    # Save the model
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    # Test
    test.test()


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))