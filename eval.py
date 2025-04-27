import os 
import torch
from models import PianoReductionCNN
from main import test_dataset, test_loader
from data import PianoReductionDataset
from torch.utils.data import DataLoader

#Helper function to binarize outputs
def binarize(output_list) :
    return (output_list > 0.5) * 1 

# Load model and set to evaluation mode
model = PianoReductionCNN()
model.load_state_dict(torch.load("first-train-model.pth"))
model.eval()

# move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#Check the total weights of the model
total = 0
for param in model.parameters():
    total += torch.sum(param.data)
print("Total weight sum:", total)



# load test set (already loaded in main)
all_generated_outputs = []
all_groundtruth_outputs = []

#Eval loop
with torch.no_grad():
    for batch in test_loader :
        batch_inputs = batch['input'].to(device).permute(0, 3, 1, 2)
        batch_targets = batch['target'].to(device).permute(0, 3, 1, 2)

        #Check after each conv
        # x = model.down_conv1(batch_inputs)
        # print("After conv1:", torch.sum(x))
        # print("sum of inputs : " + str(torch.sum(batch_inputs)))
        # outputs = model(batch_inputs)
        # print(outputs)
        # print(outputs.shape)
        # print("sum of outputs now" + str(torch.sum(outputs)))

        outputs = model(batch_inputs)

        all_generated_outputs.append(outputs.cpu())
        all_groundtruth_outputs.append(batch_targets.cpu())

all_generated_outputs = torch.cat(all_generated_outputs, dim=0)
all_groundtruth_outputs = torch.cat(all_groundtruth_outputs, dim=0)

#Binarize the outputs 
all_generated_outputs = binarize(all_generated_outputs)
all_groundtruth_outputs = binarize(all_groundtruth_outputs)



active_indices = torch.nonzero(all_generated_outputs)

print("active indices are : ", active_indices)

generated_set = set()

for idx in active_indices :
    pitch_idx = idx[2]
    onset_idx = idx[3]

    onset_time = onset_idx * hop_time_in_seconds

    generated_set.add(int(pitch_idx), onset_time)
