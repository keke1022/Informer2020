import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


# Define a simple LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, pred_len, seq_len):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
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
        "./data_cleaned/apple_stock_sentiment.csv"
    )
    data = data.drop(columns=[data.columns[0]])
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    # Function to create sequences for forecasting
    def create_train(data, seq_length, pred_len, target = None):
        xs, ys = [], []
        for i in list(range(0, len(data) - seq_length - pred_len, 32)):
            x = data[i : i + seq_length, :]  # Input sequence
            if target is None:
                y = data[
                    i + seq_length : i + seq_length + pred_len, :
                ]  # Target sequence (pred_len steps ahead)
            else:
                y = data[
                    i + seq_length : i + seq_length + pred_len, target
                ]  # Target sequence (pred_len steps ahead)
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def create_test(data, seq_length, pred_len, target=None):
        xs, ys = [], []
        for i in list(range(0, len(data) - seq_length - pred_len, 1)):
            x = data[i : i + seq_length, :]  # Input sequence
            if target is None:
                y = data[
                    i + seq_length : i + seq_length + pred_len, :
                ]  # Target sequence (pred_len steps ahead)
            else:
                y = data[
                    i + seq_length : i + seq_length + pred_len, target
                ]  # Target sequence (pred_len steps ahead)
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    # Parameters
    SEQ_LENGTH = 48  # Length of the input sequence
    PRED_LEN = 14  # Number of future steps to forecast

    # Create sequences
    X_train, y_train = create_train(data, SEQ_LENGTH, PRED_LEN, target = 0)
    X_test, y_test = create_test(data, SEQ_LENGTH, PRED_LEN, target=0)
    # Convert data to PyTorch tensors
    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(
        y_train, dtype=torch.float32
    )
    X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(
        y_test, dtype=torch.float32
    )
    y_train = y_train.reshape([y_train.shape[0], y_train.shape[1], -1])
    y_test = y_test.reshape([y_test.shape[0], y_test.shape[1], -1])
    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    input_size = data.shape[1]
    output_size = 1
    hidden_size = 32
    num_layers = 3
    model = LSTMModel(input_size, output_size, hidden_size, num_layers, PRED_LEN, SEQ_LENGTH)
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
                # print(outputs.shape)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
            total_loss = total_loss / len(train_loader)
            if total_loss < 10 ^ -5:
                return
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

    train_model(model, train_loader, criterion, optimizer, num_epochs=1000)

    # Define the evaluation function
    def evaluate_model(model, test_loader, criterion):
        model.eval()
        total_loss = 0.0
        with torch.no_grad():  # Disables gradient calculation for evaluation
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.permute(0, 2, 1).to(device)
                outputs = model(inputs)
                # print(outputs.shape)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)  # Aggregate the loss

        average_loss = total_loss / len(test_loader.dataset)
        return average_loss

    # Compute the test loss
    mse = evaluate_model(model, test_loader, criterion)
    mae = evaluate_model(model, test_loader, nn.L1Loss())
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
