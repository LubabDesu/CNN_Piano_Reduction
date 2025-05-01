import os 
import torch
from models import PianoReductionCNN
from main import  test_loader
from data import PianoReductionDataset
from torch.utils.data import DataLoader
from collections import defaultdict
from visualize import show_pianoroll

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
all_input_durations = []
all_target_durations = []

#Eval loop
with torch.no_grad():
    for batch in test_loader :
        batch_inputs = batch['input'].to(device).permute(0, 3, 1, 2)
        batch_targets = batch['target'].to(device).permute(0, 3, 1, 2)
        batch_inputs_duration = batch['input_duration'].to(device)
        batch_targets_duration = batch['target_duration'].to(device)

        outputs = model(batch_inputs)

        all_generated_outputs.append(outputs.cpu())
        all_groundtruth_outputs.append(batch_targets.cpu())
        all_input_durations.append(batch_inputs_duration.cpu())
        all_target_durations.append(batch_targets_duration.cpu())

all_generated_outputs = torch.cat(all_generated_outputs, dim=0)
all_groundtruth_outputs = torch.cat(all_groundtruth_outputs, dim=0)
all_input_durations = torch.cat(all_input_durations, dim=0)
all_target_durations = torch.cat(all_target_durations, dim=0)

#Binarize the outputs 
all_generated_outputs = binarize(all_generated_outputs)
all_groundtruth_outputs = binarize(all_groundtruth_outputs)





active_indices = torch.nonzero(all_generated_outputs)

print("active indices are : ", active_indices)

input_total_duration_in_seconds = batch_inputs_duration
target_total_duration_in_seconds=  batch_targets_duration


# Process the generated piano roll, put every note in a set which contains individual 
# sets of (pitch, onset_time)
generated_set = set()
print(f"Processing {len(active_indices)} generated notes...")
for idx in active_indices :
    sample_idx = idx[0]  # Get the index of the sample this note belongs to
    pitch_idx = idx[2]
    onset_idx = idx[3]

    current_sample_duration = all_input_durations[sample_idx]

    hop_time_in_seconds = current_sample_duration / 64
    onset_time = onset_idx * hop_time_in_seconds

    generated_set.add((int(pitch_idx), float(onset_time.item())))

# Process the groundtruth piano roll, put every note in a set which contains individual 
# sets of (pitch, onset_time)
groundtruth_set = set()
active_groundtruth_indices = torch.nonzero(all_groundtruth_outputs)
print(f"Processing {len(active_groundtruth_indices)} ground truth notes...") # Add print for debugging
for idx in active_groundtruth_indices :
    sample_idx = idx[0]  # Get the index of the sample this note belongs to
    pitch_idx = idx[2]
    onset_idx = idx[3]

    current_sample_duration = all_target_durations[sample_idx]

    hop_time_in_seconds = current_sample_duration / 64
    onset_time = onset_idx * hop_time_in_seconds

    groundtruth_set.add((int(pitch_idx), float(onset_time.item())))

# Do the note by note comparison for the groundtruth and the generated, and define other constants
tolerance = 0.05
true_positives = 0 # Correctly matched notes
false_positives = 0 # In generated set, but not in groundtruth
false_negatives = 0 # in ground truth, but not in generated

# Make a dictionary of the ground truth set, sorted by pitch
groundtruth_dict = defaultdict(list)
for pitch, onset_time in groundtruth_set :
    groundtruth_dict[pitch].append(onset_time)

for pitch in groundtruth_dict : 
    groundtruth_dict[pitch].sort()

for pair in generated_set :
    gen_pitch = pair[0]
    gen_onset_time = pair[1]
    match_exists = False
    best_time_diff = 1e9
    best_match = 0

    if (gen_pitch in groundtruth_dict) :
        for true_onset_time in groundtruth_dict[gen_pitch] :
            time_diff = abs(true_onset_time - gen_onset_time)
            if (time_diff < best_time_diff) :
                best_time_diff = time_diff
                best_match = true_onset_time
                match_exists = True
                

    if match_exists and best_time_diff <= tolerance:
        true_positives += 1
        groundtruth_dict[gen_pitch].remove(best_match)
    else : 
        false_positives += 1

# Count remaining notes that arent matched in the groundtruth
for pitch in groundtruth_dict :
    length_of_list = len(groundtruth_dict[pitch])
    false_negatives += length_of_list

# Calculate each metric 
precision = 0.0
recall = 0.0
f1_score = 0.0

tp_fp = true_positives + false_positives
tp_fn = true_positives + false_negatives

if tp_fp > 0:
    precision = true_positives / tp_fp
if tp_fn > 0:
    recall = true_positives / tp_fn
if (precision + recall) > 0:
    f1_score = 2 * precision * recall / (precision + recall)

print(f"True Positives: {true_positives}")
print(f"False Positives: {false_positives}")
print(f"False Negatives: {false_negatives}")
print(f"The precision score is {precision:.4f}, and the recall score is {recall:.4f}, " +
      f"giving us an f1 score of {f1_score:.4f}")
    