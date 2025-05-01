
import os 
import torch
from models import PianoReductionCNN
from main import test_dataset, test_loader
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

# Load model and set to evaluation mode
model = PianoReductionCNN()
model.load_state_dict(torch.load("first-train-model.pth"))
model.eval()

# move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#Eval loop
with torch.no_grad():
    for batch in test_loader :
        batch_inputs = batch['input'].to(device).permute(0, 3, 1, 2)
        batch_targets = batch['target'].to(device).permute(0, 3, 1, 2)
        batch_inputs_duration = batch['input_duration'].to(device)
        batch_targets_duration = batch['target_duration'].to(device)

        outputs = model(batch_inputs)
        show_pianoroll(batch_inputs)
        show_pianoroll(batch_targets)