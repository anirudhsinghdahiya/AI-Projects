import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

def get_data_loader(training=True):
    dataset = datasets.FashionMNIST(
        './data',
        train=training,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=training)
    return data_loader

def build_model():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model

def train_model(model, train_loader, criterion, T):
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    for epoch in range(T):
        total_loss = 0
        correct = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        print(f'Train Epoch: {epoch} Accuracy: {correct}/{len(train_loader.dataset)}'
              f'({100. * correct / len(train_loader.dataset):.2f}%) Loss: {total_loss / len(train_loader.dataset):.3f}')

def evaluate_model(model, test_loader, criterion, show_loss=True):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    if show_loss:
        print(f'Average loss: {test_loss:.4f}')
    print(f'Accuracy: {accuracy:.2f}%')

def predict_label(model, test_images, index):
    model.eval()
    data = test_images[index]
    data = data.unsqueeze(0)
    output = model(data)
    probabilities = F.softmax(output, dim=1)
    top_probs, top_labels = probabilities.topk(3, dim=1)
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
    for i in range(3):
        label = class_names[top_labels[0][i]]
        prob = top_probs[0][i].item()
        print(f'{label}: {100 * prob:.2f}%')

if __name__ == '__main__':
    criterion = nn.CrossEntropyLoss()
    train_loader = get_data_loader(training=True)
    test_loader = get_data_loader(training=False)
    model = build_model()
    epochs = 5
    train_model(model, train_loader, criterion, epochs)
    evaluate_model(model, test_loader, criterion, show_loss=True)
    test_images = next(iter(test_loader))[0]
    predict_label(model, test_images, 0)
