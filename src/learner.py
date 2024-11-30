import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import recall_score, precision_score, f1_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def trainer(model, data_loader, num_epochs=200):
    # Use Cross-Entropy Loss and Adam optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        
        # for inputs, labels in train_loader:
        for i in range(data_loader.num_batches):
            inputs, labels = data_loader.next(i)
            inputs, labels = inputs.to(device), labels.to(device)
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_accuracy = 100 * correct / total

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")


def evaluator(model, data_loader):
    # Evaluation Loop (Test the model)
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient computation for testing
        for i in range(data_loader.num_batches):
            inputs, labels = data_loader.next(i)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect predictions and true labels for recall calculation
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Test accuracy
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Compute Recall using sklearn
    recall_value = f1_score(all_labels, all_preds, average='macro')
    print(f"Macro-average F1: {recall_value:.4f}")

    return accuracy / 100                                 