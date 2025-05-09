# Training loop
from models import PianoReductionCNN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


def train_model(model, train_loader, val_loader, num_epochs = 100, device='cpu') :
    criterion = nn.BCEWithLogitsLoss(pos_weight=0.4)
    optimizer = optim.Adam(model.parameters(), lr = 0.0.01)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Current device being used is .... : " + str(device))
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        for batch in train_loader:
            batch_inputs = batch['input'].to(device).permute(0, 3, 1, 2)
            batch_targets = batch['target'].to(device).permute(0, 3, 1, 2)
            # Forward pass
            optimizer.zero_grad()  # Clear previous gradients
            outputs = model(batch_inputs)  # Get predictions
            loss = criterion(outputs, batch_targets)  # Compute loss

            # Backward pass and optimize
            loss.backward()  # Compute gradients
            optimizer.step()  # Update weights

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)

        #Validation phase
        model.eval() #Set model to evaluation mode 
        running_val_loss = 0.0
        
        with torch.no_grad() : 
            for batch in val_loader:
                batch_inputs = batch['input'].to(device).permute(0, 3, 1, 2)
                batch_targets = batch['target'].to(device).permute(0, 3, 1, 2)

                outputs = model(batch_inputs)
                val_loss = criterion(outputs, batch_targets)
                running_val_loss += val_loss.item()

        epoch_val_loss = running_val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}")

    return model