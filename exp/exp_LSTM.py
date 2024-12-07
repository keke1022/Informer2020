import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# Define a simple LSTM model
class LSTMModel(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers, pred_len, seq_len):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(feature_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, feature_size)
        self.pred = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out).permute(0, 2, 1)
        out = self.pred(out)
        return out


if __name__ == "__main__":
    data = pd.read_csv(
        "./data_cleaned/ETTh1.csv", parse_dates=["date"], index_col="date"
    )

    # Function to create sequences for forecasting
    def create_sequences(data, seq_length, pred_len):
        xs, ys = [], []
        for i in range(len(data) - seq_length - pred_len):
            x = data[i : i + seq_length, :]  # Input sequence
            y = data[
                i + seq_length : i + seq_length + pred_len, :
            ]  # Target sequence (pred_len steps ahead)
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    # Parameters
    SEQ_LENGTH = 96  # Length of the input sequence
    PRED_LEN = 24  # Number of future steps to forecast

    # Create sequences
    X, y = create_sequences(data.values, SEQ_LENGTH, PRED_LEN)
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=498
    )
    # Convert data to PyTorch tensors
    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(
        y_train, dtype=torch.float32
    )
    X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(
        y_test, dtype=torch.float32
    )
    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    feature_size = 7
    hidden_size = 100
    num_layers = 3
    model = LSTMModel(feature_size, hidden_size, num_layers, PRED_LEN, SEQ_LENGTH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training function
    def train_model(model, train_loader, criterion, optimizer, num_epochs):
        model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.permute(0, 2, 1).to(device)
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
            total_loss = total_loss / len(train_loader)
            if total_loss < 10 ^ -5:
                return
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

    train_model(model, train_loader, criterion, optimizer, num_epochs=100)

    # Define the evaluation function
    def evaluate_model(model, test_loader, criterion):
        model.eval()
        total_loss = 0.0
        with torch.no_grad():  # Disables gradient calculation for evaluation
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.permute(0, 2, 1).to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)  # Aggregate the loss

        average_loss = total_loss / len(test_loader.dataset)
        return average_loss

    # Compute the test loss
    test_loss = evaluate_model(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}")
