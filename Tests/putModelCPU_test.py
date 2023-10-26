import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

# Define a simple feedforward neural network
class mixedNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(mixedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # default on GPU, but one layer on CPU
        x = self.fc1(x)

        self.fc2 = self.fc2.to('cpu')
        x = x.to('cpu')
        x = self.fc2(x)
        self.fc2 = self.fc2.to('cuda')
        x = x.to('cuda')

        x = self.fc3(x)
        return x

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # default on GPU, but one layer on CPU
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# Function to train the model
def inf_model(model, device, train_loader):
    model.eval()
    total_loss = 0.0

    for inputs, labels in train_loader:
        # t1 = time.time()
        inputs, labels = inputs.view(-1, 28 * 28).to(device), labels.to(device)
        # t2 = time.time()
        # print(f"Overhead for input moving: {t2-t1}")
        # optimizer.zero_grad()
        t3 = time.time()
        outputs = model(inputs)
        t4 = time.time()
        print(f"Time for model forward: {t4-t3}")

        # loss = criterion(outputs, labels)
        # loss.backward()
        # optimizer.step()
        # total_loss += loss.item()

    return total_loss

# # default on GPU, but one layer on CPU
# def train_model_onMixedDevice(model, device, train_loader, criterion, optimizer):
#     model.train()
#     total_loss = 0.0

#     for inputs, labels in train_loader:
#         # input to GPU
#         t1 = time.time()
#         inputs, labels = inputs.view(-1, 28 * 28).to(device), labels.to(device)
#         t2 = time.time()
#         print(f"Overhead for input moving: {t2-t1}")

#         optimizer.zero_grad()

#         # forward
#         t3 = time.time()
#         outputs = model(inputs)
#         t4 = time.time()
#         print(f"Time for model forward: {t4-t3}")
        
#         # backward
#         loss = criterion(outputs, labels)
#         loss.backward()
#         # step
#         optimizer.step()
#         total_loss += loss.item()

#     return total_loss

# Function to run the model and measure execution time
def run_model(model, device, train_loader, criterion, optimizer):
    inf_model(model, device, train_loader, criterion, optimizer)

if __name__ == "__main__":
    # Set device (GPU if available, otherwise CPU)
    device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_cpu = torch.device("cpu")

    # Define hyperparameters
    input_size = 784  # MNIST images are 28x28 pixels
    hidden_size = 128
    output_size = 10
    learning_rate = 0.001
    batch_size = 64
    epochs = 5

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model and move it to GPU
    model_gpu = SimpleNN(input_size, hidden_size, output_size).to(device_gpu)
    model_mixed = mixedNN(input_size, hidden_size, output_size).to(device_gpu)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model_gpu.parameters(), lr=learning_rate)

    print("inf on GPU mixed:")
    # Run the model on GPU and measure time
    time_gpu = inf_model(model_mixed, device_gpu, train_loader)

    print("inf on GPU:")
    # Run the model on GPU and measure time
    time_gpu = inf_model(model_gpu, device_gpu, train_loader)

    # # Initialize the same model and move it to CPU
    # model_cpu = SimpleNN(input_size, hidden_size, output_size).to(device_cpu)

    # print("Train on CPU:")
    # # Run the model on CPU and measure time
    # time_cpu = run_model(model_cpu, device_cpu, train_loader, criterion, optimizer)

# --------------------------------
# import torch
# import torch.nn as nn
# import time

# class simpleNN(nn.Module):
#     def __init__(self):
#         super(simpleNN, self).__init__()
#         self.layer1 = nn.Linear(10, 5)
#         self.layer2 = nn.Linear(5, 5)
#         self.layer3 = nn.Linear(5, 2)

#     def forward(self, x):
#         # Move input tensor to GPU before passing through layer1
#         x = self.layer1(x)
        
#         x = self.layer2(x)

#         x = self.layer3(x)

#         return x

# # Create an instance of the model
# model = simpleNN()

# # ----------- on GPU ---------------
# # Move the parameters of layer1 to CPU
# model.to('cuda')
# # Create a dummy input tensor
# input_tensor = torch.randn((3, 10)).to('cuda')  # Move input to GPU
# # Forward pass using the model
# t1 = time.time()
# output = model(input_tensor)
# t2 = time.time()
# # Print the output
# print(f"GPU time: {t2-t1}")

# # ----------- on CPU -----------------
# # Move the parameters of layer1 to CPU
# model.to('cpu')
# # Create a dummy input tensor
# input_tensor = torch.randn((3, 10)).to('cpu')  # Move input to GPU
# # Forward pass using the model
# output = model(input_tensor)
# # Print the output
# print(f"CPU time: {t2-t1}")