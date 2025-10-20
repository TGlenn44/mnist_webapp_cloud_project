import torch
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from neural_networks.ConvolutionalClassifier import ConvolutionalClassifier
from neural_networks.FullyConnectedClassifier import FullyConnectedClassifier
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

# Loading Data
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(
    root="data", download=True, train=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Create an instance of the image classifier model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier1 = FullyConnectedClassifier().to(device)
classifier2 = ConvolutionalClassifier().to(device)

# Define the optimizer and loss function
optimizer1 = Adam(classifier1.parameters(), lr=0.001)
loss_fn1 = nn.CrossEntropyLoss()

optimizer2 = Adam(classifier2.parameters(), lr=0.001)
loss_fn2 = nn.CrossEntropyLoss()

# Train classifier1
for epoch in range(10):  # Train for 10 epochs
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer1.zero_grad()  # Reset gradients
        outputs = classifier1(images)  # Forward pass
        loss = loss_fn1(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer1.step()  # Update weights

    print(f"Epoch:{epoch} loss is {loss.item()}")

torch.save(classifier1.state_dict(), "models/model_state1_fc.pt")  # Save the trained model

# Train classifier2
for epoch in range(10):  # Train for 10 epochs
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer2.zero_grad()  # Reset gradients
        outputs = classifier2(images)  # Forward pass
        loss = loss_fn2(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer2.step()  # Update weights

    print(f"Epoch:{epoch} loss is {loss.item()}")

torch.save(classifier2.state_dict(), "models/model_state2_cnn.pt")  # Save the trained model
