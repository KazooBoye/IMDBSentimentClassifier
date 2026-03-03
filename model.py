import torch
import torch.nn as nn

class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers):
        super(SentimentClassifier, self).__init__()

        self.n_layers = n_layers # Number of LSTM layers
        self.hidden_dim = hidden_dim # Hidden dimension for LSTM

        # Embedding layer to convert word indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer to capture sequential information
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)

        # Fully connected layer to output sentiment prediction
        self.fc = nn.Linear(hidden_dim, output_dim)

        # Sigmoid activation for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0) # Get batch size from input

        # x shape: (batch_size, seq_length)
        embedding = self.embedding(x) # shape: (batch_size, seq_length, embedding_dim)

        # Pass through LSTM
        lstm_out, (hidden, cell) = self.lstm(embedding) # lstm_out shape: (batch_size, seq_length, hidden_dim)

        # Extract the output of the last time step
        last_output = lstm_out[:, -1, :] # shape: (batch_size, hidden_dim)

        # Pass through linear layer and sigmoid activation
        out = self.fc(last_output) # shape: (batch_size, output_dim)
        out = self.sigmoid(out) # shape: (batch_size, output_dim)

        return out