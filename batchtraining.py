import torch
from torch.utils.data import DataLoader
from model import SentimentClassifier
from torch.utils.data import TensorDataset
from torch.nn import BCELoss
import torch.nn as nn
import numpy as np
import pandas as pd

# Load training data
train_features = np.load('./aclImdb/tokenized/train_cleaned_tokenized_features.npy')
train_df = pd.read_csv('./aclImdb//tokenized/train_cleaned_tokenized.csv')
train_labels = train_df['sentiment'].values

# Load test data
test_features = np.load('./aclImdb/tokenized/test_cleaned_tokenized_features.npy')
test_df = pd.read_csv('./aclImdb/tokenized/test_cleaned_tokenized.csv')
test_labels = test_df['sentiment'].values

# Convert to PyTorch tensors
train_features = torch.from_numpy(train_features).long()
train_labels = torch.from_numpy(train_labels).float() # Use float for BCEWithLogitsLoss
test_features = torch.from_numpy(test_features).long()
test_labels = torch.from_numpy(test_labels).float()

# Create TensorDataset for batch training
train_data = TensorDataset(train_features, train_labels)
test_data = TensorDataset(test_features, test_labels)

# Create DataLoader for batch training
batch_size = 50
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

# Init model
vocab_size = len(open('./aclImdb/vocab.txt', 'r').readlines()) + 1 # +1 for unknown token
model = SentimentClassifier(vocab_size=vocab_size, embedding_dim=400, hidden_dim=256, output_dim=1, n_layers=2)
print(model)

# Use CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr= 0.001)

# Training loop
epochs = 5
print('Starting training...')

for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.unsqueeze(1) # Reshape labels to (batch_size, 1) for BCELoss
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs) # shape: (batch_size)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.unsqueeze(1)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            # Calculate accuracy
            predictions = (outputs >= 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    accuracy = 100 * correct / total
    print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f}, Test Loss: {test_loss/len(test_loader):.4f}, Accuracy: {accuracy:.2f}%')

torch.save(model.state_dict(), './sentiment_classifier.pth')
print('Model saved to sentiment_classifier.pth')