import os 
import torch
from models import PianoReductionCNN
from main import test_dataset, test_loader
from data import PianoReductionDataset
from torch.utils.data import DataLoader

# Load model and set to evaluation mode
model = PianoReductionCNN()
model.load_state_dict(torch.load("first-train-model.pth"))
model.eval()

# move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# load test set (already loaded in main)

#Eval loop
with torch.no_grad :
    for batch in test_loader :
        batch_inputs = batch['input'].to(device).permute(0, 3, 1, 2)
        batch_targets = batch['target'].to(device).permute(0, 3, 1, 2)
        outputs = model(batch_inputs)
        print(outputs)
