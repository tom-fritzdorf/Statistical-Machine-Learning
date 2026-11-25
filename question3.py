# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from tqdm.auto import tqdm


from sklearn import datasets
from sklearn.model_selection import train_test_split


# !pip install torchvision
import torchvision
# !pip install torchmetrics
import torchmetrics

# %%
digits = datasets.load_digits()
X = digits.images   # images
y = digits.target   # labels

#  20% for testing.
x_train, x_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

#  60/20/20 
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    test_size=0.25,
    random_state=42,
    stratify=y_train,
)

batch_size = 32

n_train = 1077
n_val = 360

X_train = torch.tensor(x_train[0:n_train,:], dtype=torch.float32).unsqueeze(1) / 16.0#change pixel values
X_val = torch.tensor(x_val[0:n_val,:], dtype=torch.float32).unsqueeze(1) / 16.0
X_test = torch.tensor(x_test[0:n_val,:], dtype=torch.float32).unsqueeze(1) / 16.0


y_train = torch.as_tensor(y_train[0:n_train], dtype=torch.long)
y_val = torch.as_tensor(y_val[0:n_val], dtype=torch.long)
y_test = torch.as_tensor(y_test[0:n_val], dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True #only shuffle training set
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False
)

print(f"Training: {len(train_dataset)} | Validation: {len(val_dataset)} | Test: {len(test_dataset)}")


# %%

#defining the layers and forward pass
in_channels = 1
out_channels = 10

class CNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=48, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.conv2 = nn.Conv2d(in_channels=48, out_channels=32, kernel_size=3, padding=1)
        #self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)

        #self.fc1 = nn.Linear(16 * 2 * 2, 64)
        self.fc2 = nn.Linear(32 * 2 * 2, out_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        #x = F.relu(self.conv3(x))
        
        
        x = x.reshape(x.shape[0], -1) 
        #x = F.relu(self.fc1(x))
        
        x = self.fc2(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model = CNN(in_channels, out_channels).to(device)
print(model)

# %%
learning_rate = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
val_list = []
train_list = []
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    total_images = 0

    for data, targets in train_loader:
        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        scores = model(data)
        loss = criterion(scores, targets)
        loss.backward()
        optimizer.step()

        batch_size = data.size(0)
        total_loss += loss.item() * batch_size
        total_images += batch_size

    average_loss = total_loss / total_images

    model.eval()
    validation_loss = 0
    val_images = 0

    with torch.no_grad():
        for data, targets in val_loader:
            data = data.to(device)
            targets = targets.to(device)

            scores = model(data)         
            loss = criterion(scores, targets)

            validation_loss += loss.item() * targets.size(0)
            val_images += targets.size(0)

    validation_loss = validation_loss / val_images
    train_list.append(average_loss)
    val_list.append(validation_loss)
    print( f"Epoch: {epoch+1} | Train loss: {average_loss:.5f} | Validation loss: {validation_loss:.5f}")
    

# %%
model.eval()

correct = 0
total = 0
total_loss = 0.0


with torch.no_grad(): 
    for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)

        scores = model(data)         
        loss = criterion(scores, targets)

        predictions = scores.argmax(dim=1) 

        correct += (predictions == targets).sum().item()
        total += targets.size(0)
        total_loss += loss.item() * targets.size(0)

accuracy = 100 * correct / total
average_loss = total_loss / total

print(f"Test accuracy: {accuracy:.3f}%")
print(f"Test loss:     {average_loss:.7f}")

# %%
plt.figure(figsize=(9, 5.5))

epochs = range(1, len(train_list) + 1)

plt.plot(epochs, train_list, label="Training Loss", linewidth=2)
plt.plot(epochs, val_list, label="Validation Loss", linewidth=2)

plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Cross Entropy Loss", fontsize=12)
plt.title("CNN Training and Validation Loss", fontsize=14)

plt.legend(fontsize=11)
plt.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
plt.show()

# %%
model.eval()

wrong_images = []
wrong_predictions = []
wrong_labels = []

with torch.no_grad():
    for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)

        scores = model(data)
        predictions = scores.argmax(dim=1)

        wrong = predictions != targets

        wrong_images.extend(data[wrong].cpu())
        wrong_predictions.extend(predictions[wrong].cpu().tolist())
        wrong_labels.extend(targets[wrong].cpu().tolist())

        if len(wrong_images) >= 5:
            break

n = min(5, len(wrong_images))

fig, axes = plt.subplots(1, n, figsize=(15, 3))

for i, ax in enumerate(np.atleast_1d(axes)):
    ax.imshow(wrong_images[i].squeeze(), cmap="gray")
    ax.set_title(
        f"True: {wrong_labels[i]}\nPredicted: {wrong_predictions[i]}"
    )
    ax.axis("off")

plt.suptitle("Incorrectly Classified MNIST Images", fontsize=14)
plt.tight_layout()
plt.show()

# %%
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from tqdm.auto import tqdm
from torch.utils.data import random_split

# !pip install torchvision
import torchvision
# !pip install torchmetrics
import torchmetrics

# %%
#MNIST dataset 
batch_size = 32


train_dataset = datasets.MNIST(
    root="docs/",
    download=True,
    train=True,
    transform=transforms.ToTensor()
)

mean = train_dataset.data.float().mean() / 255
std = train_dataset.data.float().std() / 255

transform = transforms.Compose([
    transforms.ToTensor(),                          
    transforms.Normalize((mean,), (std,))      
])

train_dataset.transform = transform #normalize and tensor before split

train_dataset, val_dataset = random_split(
    train_dataset,
    [0.8, 0.2],
    generator=torch.Generator().manual_seed(42)
)



test_dataset = datasets.MNIST(root="docs/", download=True, train=False, transform=transform)



test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#only shuffle train dataset
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)



print("Training:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))


# %%
#This model we can utilize pooliung with bigger stride size since 28x28 isntead of 8x8
#And more layers
#same chanenls as before
in_channels = 1
out_channels = 10
class CNN2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNN2, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=64, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        
        self.bn1 = nn.BatchNorm2d(64) #add batchnormalization can help with faster acc and stabilization

        #28/4=14
        #14/2=7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, out_channels)


    def forward(self, x):
        #print("Input:", x.shape)
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.pool(x)
        #print("After pool1:", x.shape)

        x = F.relu(self.conv3(x))
        #print("After pool2:", x.shape)
        
        x = x.reshape(x.shape[0], -1) 
        #print("After flatten:", x.shape)

        x = F.relu(self.fc1(x))
        #print("After fc1:", x.shape)
        x = self.fc2(x)
        return x

images, labels = next(iter(train_loader))

print("Images:", images.shape)
print("Labels:", labels.shape)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model_2 = CNN2(in_channels, out_channels).to(device)
print(model_2)

# %%
learning_rate = 0.01
best_val_loss = float("inf")
epochs_no_imp = 0
wait = 5


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_2.parameters(), lr=learning_rate)

num_epochs = 17

count = 1
for epoch in range(num_epochs):
    model_2.train()
    total_loss = 0.0
    total_images = 0

    for data, targets in train_loader:
        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        scores = model_2(data)
        loss = criterion(scores, targets)
        loss.backward()
        optimizer.step()

        batch_size = data.size(0)
        total_loss += loss.item() * batch_size
        total_images += batch_size



    average_loss = total_loss / total_images
    model_2.eval()
    validation_loss = 0
    val_images = 0


    with torch.no_grad():
        for data, targets in val_loader:
            data = data.to(device)
            targets = targets.to(device)

            scores = model_2(data)         
            loss = criterion(scores, targets)

            validation_loss += loss.item() * targets.size(0)
            val_images += targets.size(0)


    validation_loss = validation_loss / val_images

    print( f"Epoch: {epoch+1} | Train loss: {average_loss:.5f} | Validation loss: {validation_loss:.5f}")

    if validation_loss < best_val_loss: #early stopping
        best_val_loss = validation_loss
        epochs_no_imp = 0

    else:
        epochs_no_imp += 1

    if epochs_no_imp >= wait:
        print("no validation improvment")
        break



# %%
model_2.eval()

correct = 0
total = 0
total_loss = 0.0

with torch.no_grad(): 
    for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)

        scores = model_2(data)         
        loss = criterion(scores, targets)

        predictions = scores.argmax(dim=1) 

        correct += (predictions == targets).sum().item()
        total += targets.size(0)
        total_loss += loss.item() * targets.size(0)

accuracy = 100 * correct / total
average_loss = total_loss / total


print(f"Testing accuracy: {accuracy:.3f}%")
print(f"Testing loss:     {average_loss:.7f}")

# %%
model.eval()

wrong_images = []
wrong_predictions = []
wrong_labels = []

with torch.no_grad():
    for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)

        scores = model(data)
        predictions = scores.argmax(dim=1)

        wrong = predictions != targets

        wrong_images.extend(data[wrong].cpu())
        wrong_predictions.extend(predictions[wrong].cpu().tolist())
        wrong_labels.extend(targets[wrong].cpu().tolist())

        if len(wrong_images) >= 5:
            break

n = min(5, len(wrong_images))

fig, axes = plt.subplots(1, n, figsize=(15, 3))

for i, ax in enumerate(np.atleast_1d(axes)):
    ax.imshow(wrong_images[i].squeeze(), cmap="gray")
    ax.set_title(
        f"True: {wrong_labels[i]}\nPredicted: {wrong_predictions[i]}"
    )
    ax.axis("off")

plt.suptitle("Incorrectly Classified MNIST Images", fontsize=14)
plt.tight_layout()
plt.show()

# %%
model_2.eval()

wrong_images = []
wrong_predictions = []
wrong_labels = []

with torch.no_grad():
    for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)

        scores = model_2(data)
        predictions = scores.argmax(dim=1)

        wrong = predictions != targets

        wrong_images.extend(data[wrong].cpu())
        wrong_predictions.extend(predictions[wrong].cpu().tolist())
        wrong_labels.extend(targets[wrong].cpu().tolist())

        if len(wrong_images) >= 5:
            break


n = min(5, len(wrong_images))

fig, axes = plt.subplots(1, n, figsize=(15, 3))

for i, ax in enumerate(np.atleast_1d(axes)):
    ax.imshow(wrong_images[i].squeeze(), cmap="gray")
    ax.set_title(
        f"True: {wrong_labels[i]}\nPredicted: {wrong_predictions[i]}"
    )
    ax.axis("off")

plt.suptitle("Incorrectly Classified MNIST Images", fontsize=14)
plt.tight_layout()
plt.show()


