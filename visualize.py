
import os 
import torch
from models import PianoReductionCNN
from main import test_loader, test_dataset
import pypianoroll as ppr
import matplotlib.pyplot as plt

def show_pianoroll(roll, title=None) :
    """
    roll: 2D array or tensor shaped (128 pitches, T time-frames)
    """
    fig, ax = plt.subplots(figsize=(10,4))
    ppr.plot_pianoroll(roll, ax=ax)

    if title : 
        ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('MIDI Pitch')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PianoReductionCNN().to(device)
model.load_state_dict(torch.load("first-train-model.pth", map_location=device))
model.eval()

idx = 5
sample = test_dataset[idx]
print(f"Visualizing {sample} now...")
roll_input = sample['input']
roll_target = sample['target']

inp = roll_input.to(device).permute(0, 3, 1, 2)


with torch.no_grad():
    pred = model(inp)            

# pull out the 2D arrays for plotting
gt_roll   = roll_target.cpu().numpy().T   # maybe (P, T) after transpose
pred_roll = pred[0,0].cpu().numpy() 

show_pianoroll(gt_roll)
show_pianoroll(pred_roll)