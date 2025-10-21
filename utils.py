#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# %%


class ProteinDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# %%


def get_data_loaders(csv_path, batch_size, test_size=0.2):
    # Read data
    df = pd.read_csv(csv_path)

    # Separate features (X) and target (y)
    X = df.drop('RMSD', axis=1).values
    y = df['RMSD'].values

    # First, split the data into training (80%) and a temporary set (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Length of training set: {len(X_train)}")

    # Next, split the temporary set into validation (10%) and test (10%) sets
    # (test_size=0.5 means 50% of the 20% temp set, which is 10% of the original data)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    print(f"Length of validation set: {len(X_val)}")
    print(f"Length of test set: {len(X_test)}")

    # Create a scaler
    scaler = StandardScaler()

    # Fit the scaler on the training data and transform it
    X_train_scaled = scaler.fit_transform(X_train)

    # Use the same scaler to transform the validation and test data
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Create PyTorch Datasets
    train_dataset = ProteinDataset(X_train_scaled, y_train)
    val_dataset = ProteinDataset(X_val_scaled, y_val)
    test_dataset = ProteinDataset(X_test_scaled, y_test)

    # Create PyTorch DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

    num_features = X.shape[1]

    return train_loader, val_loader, test_loader, num_features


# %%


def train_model(model, train_loader, val_loader, criterion=None, optimizer=None, learning_rate=1e-3, num_epochs=30, seed=42):
    """
    Function to train the model and record loss history.
    """
    # Set default criterion
    if criterion is None:
        criterion = nn.MSELoss()
    
    # Set default optimizer
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Store loss history for plotting
    history = {'train_loss': [], 'val_loss': []}
    
    print("Starting training...")
    for epoch in range(num_epochs):
        # Training Phase
        model.train()  # Set the model to training mode
        running_train_loss = 0.0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()               # Clear previous gradients
            outputs = model(inputs)             # Forward pass (get predictions)
            loss = criterion(outputs, targets) # Calculate loss
            loss.backward()                     # Backward pass (compute gradients)
            optimizer.step()                    # Update model weights with gradients
            
            running_train_loss += loss.item() * inputs.size(0)
            
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_train_loss)
        

        # Validation Phase
        model.eval()  # Set the model to evaluation mode
        running_val_loss = 0.0
        with torch.no_grad():  # No need to track gradients during validation
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item() * inputs.size(0)
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        history['val_loss'].append(epoch_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}")
        
    print("Finished Training.")
    return history


# %%


def plot_loss(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()


# %%

def evaluate_model(model, loader):
    """
    Evaluates the model on a given dataset.
    
    This function calculates and prints the total average loss over the
    entire dataset and also shows prediction examples from the first batch.
    """
    # Calculate total Loss over the entire dataset
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    num_samples = 0

    # Set model to evaluation mode
    model.eval() 

    with torch.no_grad(): # Deactivate autograd engine for efficiency
        for inputs, targets in loader:
            # Make predictions
            predictions = model(inputs)
            
            # Calculate loss for the batch
            loss = loss_fn(predictions, targets)
            
            # Accumulate total loss (weighted by batch size)
            total_loss += loss.item() * inputs.size(0)
            num_samples += inputs.size(0)

    # Calculate the average loss over all samples
    avg_loss = total_loss / num_samples
    print(f"\n--- Model Evaluation ---")
    print(f"Average Loss (MSE) on the dataset: {avg_loss:.4f}")

    # Show prediction examples from the first batch
    print("\n--- Prediction Examples ---")
    
    # Get a single batch of data from the loader
    sample_inputs, sample_targets = next(iter(loader))

    # Get predictions for that first batch
    with torch.no_grad():
        predictions = model(sample_inputs)

    # Print the results in a nice format
    print("Predicted RMSD\t|\tActual RMSD")
    print("-" * 35)
    for pred, actual in zip(predictions, sample_targets):
        print(f"{pred.item():.4f}\t\t|\t{actual.item():.4f}")


# %%


class FullyConnectedNetwork(nn.Module):
    def __init__(self):
        # Initialize parent class (nn.Module)
        super(FullyConnectedNetwork, self).__init__()

        # First layer: Takes 9 input features, outputs 32
        self.layer1 = nn.Linear(9, 32)

        # First activation function
        self.activation1 = nn.ReLU()

        # Second layer: Takes 32 input features, outputs 128
        self.layer2 = nn.Linear(32, 128)

        # Second Activation function
        self.activation2 = nn.ReLU()

        # Output layer: Takes 128 features, outputs 1 (our predicted RMSD)
        self.layer3 = nn.Linear(128, 1)

    def forward(self, x):
        # Pass x through first layer
        x = self.layer1(x)

        # Apply first activation function
        x = self.activation1(x)

        # Pass x through second layer
        x = self.layer2(x)

        # Apply second activation function
        x = self.activation2(x)

        # Pass x through final output layer
        x = self.layer3(x)
        return x



# %%


def get_generic_model():
  # Hyperparameters
  DATA_PATH = 'data/protein.csv'
  LEARNING_RATE = 0.001
  BATCH_SIZE = 32

  # Load Data
  train_loader, val_loader, test_loader, num_features = get_data_loaders(DATA_PATH, BATCH_SIZE)

  # Define the network architecture now that we know num_features
  input_size = num_features
  output_size = 1

  model = FullyConnectedNetwork()

  criterion = nn.MSELoss()

  optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

  return model, train_loader, val_loader, test_loader, criterion, optimizer



# %%


def get_pseudo_predictions():
  return torch.tensor([[0.2], [0.8], [0.5]], dtype=torch.float32)

def get_pseudo_targets():
  return torch.tensor([[0.0], [1.0], [0.6]], dtype=torch.float32)

def plot_pseudo_data(predictions, targets, loss_value):

  # Convert to NumPy for plotting
  pred_np = predictions.numpy().flatten()
  target_np = targets.numpy().flatten()
  indices = np.arange(len(pred_np))

  # Plot predictions vs targets
  plt.figure(figsize=(7, 5))
  plt.scatter(indices, target_np, color='green', label='Targets', s=100)
  plt.scatter(indices, pred_np, color='blue', label='Predictions', s=100)

  # Draw error lines
  for i in range(len(pred_np)):
      plt.plot([indices[i], indices[i]], [target_np[i], pred_np[i]], 'r--')  # error line
      error = pred_np[i] - target_np[i]
      plt.text(indices[i] + 0.1, (target_np[i] + pred_np[i]) / 2,
              f"err²={error**2:.2f}", fontsize=9, color='red')

  plt.title(f"Mean Squared Error = {loss_value:.4f}")
  plt.xlabel("Sample Index")
  plt.ylabel("Value")
  plt.legend()
  plt.grid(True)
  plt.show()

