import numpy as np
import pickle
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image

def load_cifar10_batch(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
        X = data_dict[b'data']
        Y = data_dict[b'labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)  # reshape to (10000, 32, 32, 3)
        Y = np.array(Y)
        return X, Y

def load_cifar10(data_dir):
    X_train = []
    Y_train = []
    for i in range(1, 6):
        file = os.path.join(data_dir, f'data_batch_{i}')
        X, Y = load_cifar10_batch(file)
        X_train.append(X)
        Y_train.append(Y)
    X_train = np.concatenate(X_train)
    Y_train = np.concatenate(Y_train)

    X_test, Y_test = load_cifar10_batch(os.path.join(data_dir, 'test_batch'))
    return X_train, Y_train, X_test, Y_test


class CIFAR10Dataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

class fullyConnectL(nn.Module):
    def __init__(self):
        super(fullyConnectL, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train():
    data_dir = './model/cifar-10-batches-py'  # path
    X_train, Y_train, X_test, Y_test = load_cifar10(data_dir)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = CIFAR10Dataset(X_train, Y_train, transform=transform)
    test_dataset = CIFAR10Dataset(X_test, Y_test, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    net = fullyConnectL()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    print(f"{'Loop':<10}{'Train Loss':<15}{'Train Acc %':<15}{'Test Loss':<15}{'Test Acc %':<15}")
    for epoch in range(10):
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        net.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_accuracy = 100 * correct_train / total_train

        # Evaluate on test data
        net.eval()
        correct_test = 0
        total_test = 0
        test_loss = 0.0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = net(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_accuracy = 100 * correct_test / total_test

        print(f"{epoch + 1}/10      {running_loss / len(train_loader):<15.4f}"
              f"{train_accuracy:<15.4f}{test_loss / len(test_loader):<15.4f}{test_accuracy:<15.4f}")

    # Save the trained model
    if not os.path.exists('model'):
        os.makedirs('model')
    torch.save(net.state_dict(), './model/model.ckpt')
    print("Model saved in file: ./model/model.ckpt")


def test(image_path):
    net = fullyConnectL()
    try:
        net.load_state_dict(torch.load('./model/model.ckpt'))  # Adjusted path
        net.eval()
    except FileNotFoundError:
        print("Model file not found. Please train the model first using 'python classify.py train'.")
        return

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    try:
        image = Image.open(image_path)
        image = image.resize((32, 32))
        image = transform(image).unsqueeze(0)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    output = net(image)
    _, predicted = torch.max(output, 1)
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    print('Prediction result:', classes[predicted.item()])


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Please provide an argument: 'train' or 'test <image_path>'")
    elif sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'test' and len(sys.argv) == 3:
        test(sys.argv[2])
    else:
        print("Invalid command.")
