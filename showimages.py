# Shows some dataset images for fun.
import matplotlib.pyplot as plt
import numpy as np
import dataset


# Function to show an image
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Function to show normalized image
def norm_imshow(img):
    img = img
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Get some random training images
dataiter = iter(dataset.trainloader)
images, labels = dataiter.next()

# Show images
imshow(dataset.torchvision.utils.make_grid(images))
norm_imshow(dataset.torchvision.utils.make_grid(images))

# Print labels
print(' '.join('%5s' % dataset.classes[labels[j]] for j in range(dataset.batch_size)))