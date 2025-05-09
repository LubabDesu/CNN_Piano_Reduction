
import os 
import torch
from models import PianoReductionCNN
from main import test_loader, test_dataset
import pypianoroll as ppr
import matplotlib.pyplot as plt
import numpy as np
from data import PianoReductionDataset

def show_pianoroll(roll, title=None) :
    """
    roll: 2D array or tensor shaped (128 pitches, T time-frames)
    """
    fig, ax = plt.subplots(figsize=(10,10))
    ppr.plot_pianoroll(pianoroll=roll, ax=ax)

    if title : 
        ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('MIDI Pitch')
    plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PianoReductionCNN().to(device)
model.load_state_dict(torch.load("first-train-model.pth", map_location=device))
model.eval()

sample = next(iter(test_loader))
batch_size = 64

'''# print(f"Visualizing {sample} now...")
roll_input = sample['input']
roll_target = sample['target']
pred_single = roll_input[idx]
gt_single = roll_target[idx]

inp = roll_input.permute(0,3,1,2).to(device)


with torch.no_grad():
    pred = model(inp)            
gt_roll = gt_single.squeeze()
pred_roll = pred_single.squeeze()
# # pull out the 2D arrays for plotting
# gt_roll   = roll_target.cpu().numpy().T   # maybe (P, T) after transpose
# pred_roll = pred[0,0].cpu().numpy() 

# gt_roll = np.reshape(gt_roll, (-1,88))
# pred_roll = np.reshape(pred_roll, (-1,88))

print("gt_roll shape:", gt_roll.shape, "min/max:", gt_roll.min(), gt_roll.max())
print("pred_roll shape:", pred_roll.shape, "min/max:", pred_roll.min(), pred_roll.max())

show_pianoroll(gt_roll, "groundtruth")
show_pianoroll(pred_roll,"predicted")'''

# --- Inference and Plotting Preparation ---
idx = 0 # Plot the first sample in the batch

# 1. Get Ground Truth
gt_single_tensor = sample['target'][idx] # Shape: (88, 64, 1) = (P, T, C)
gt_squeezed = gt_single_tensor.squeeze() # Shape: (88, 64) = (P, T)
gt_numpy = gt_squeezed.cpu().numpy() # Convert to NumPy -> (P, T)

# 2. Prepare Model Input and Get Prediction
inp = sample['input'].permute(0, 3, 1, 2).to(device) # -> (B, 4, 88, 64)

# Use a dummy prediction if model isn't loaded
# Replace with actual model inference
dummy_pred_output = (torch.rand(batch_size, 1, 88, 64) > 0.7).float().to(device) # B, C_out, P, T
pred = dummy_pred_output
with torch.no_grad():
    pred = model(inp) # Shape maybe (B, 1, 88, 64) = (B, C_out, P, T)

# 3. Process Prediction
pred_single_tensor = pred[idx, 0] # Select sample idx, channel 0 -> Shape: (88, 64) = (P, T)

pred_numpy = pred_single_tensor.detach().cpu().numpy() # Convert to NumPy -> (P, T)

# 4. Plot
print("--- Ground Truth ---")
show_pianoroll(gt_numpy, f"Ground Truth (Sample {idx})")

print("\n--- Prediction ---")
show_pianoroll(pred_numpy, f"Prediction (Sample {idx})")