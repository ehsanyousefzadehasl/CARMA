import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timm import create_model
from torchsummary import summary
import argparse

# Command-line arguments
parser = argparse.ArgumentParser(description='Train EfficientNet-B0 on CIFAR-10')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs (default: 20)')
args = parser.parse_args()

# Check for device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset preparation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

# Model creation
model = create_model('efficientnet_b0', pretrained=False, num_classes=100)
model = model.to(device)

# Print model summary
print("Model Summary:")
summary(model, input_size=(3, 32, 32))

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# Training function
def train(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(inputs)}/{len(train_loader.dataset)}] "
                  f"Loss: {loss.item():.4f} | Acc: {100. * correct / total:.2f}%")

# Test function
def test():
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(f"Test Loss: {test_loss / len(test_loader):.4f} | Test Acc: {100. * correct / total:.2f}%")
    return 100. * correct / total

# Main loop
best_acc = 0.0
for epoch in range(1, args.epochs + 1):  # Train for specified epochs
    train(epoch)
    acc = test()
    scheduler.step()

    # Save the model if accuracy improves
    # if acc > best_acc:
    #     print(f"Saving model with accuracy: {acc:.2f}%")
    #     torch.save(model.state_dict(), 'efficientnet_b0_cifar10.pth')
    #     best_acc = acc
