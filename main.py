import os 
import torch
from models import PianoReductionCNN
from data import PianoReductionDataset
from torch.utils.data import DataLoader
from training import train_model
from sklearn.model_selection import train_test_split

# To do left : 
#  - align the rest of the files in dataset
#  - process the data such that i can loop through all the directories and datasets
#  - train the model on all the data (split into train and test sets )

input_dir = "/Users/lucasyan/Winter - Spring 25 research project/CNN_Piano_Reduction/aligned_dataset/output_aligned_files/"

# numbered = [
#     os.path.join(input_dir, str(num)) for num in range(40)
# ]

all_input_files = []
all_target_files = []
base_dir = "/Users/lucasyan/Winter - Spring 25 research project/CNN_Piano_Reduction/aligned_dataset/aligned"
found_orchestra = 0
found_piano = 0
paired_count = 0

file_pairs = {} # Dictionary to store pairs: {base_name: {'orch': path, 'piano': path}}

for subfolder in os.listdir(base_dir):
    if subfolder == ".DS_Store":
        continue
    subfolder_path = os.path.join(base_dir, subfolder)
    if not os.path.isdir(subfolder_path):
        continue

    piece_name = subfolder  # The subfolder name is the piece name
    file_pairs[piece_name] = {} # Initialize an entry for this piece

    for filename in os.listdir(subfolder_path):
        full_path = os.path.join(subfolder_path, filename)
        if filename == "orchestra.mid":
            file_pairs[piece_name]['orch'] = full_path
            found_orchestra += 1
        elif filename == "piano.mid":
            file_pairs[piece_name]['piano'] = full_path
            found_piano += 1

# Filter for complete pairs
all_input_files = []
all_target_files = []
for base_name, paths in file_pairs.items(): #base_name holds the piece/subfolder name, and paths holds the inner dictionary 
    if 'orch' in paths and 'piano' in paths:
        all_input_files.append(paths['orch'])
        all_target_files.append(paths['piano'])
        paired_count += 1
# print(file_pairs)
print(f"Found {found_orchestra} orchestra files and {found_piano} piano files.")
print(f"Successfully paired {paired_count} files.")

if paired_count == 0:
    print("Error: No paired files found. Check naming convention ('orchestra.mid', 'piano.mid') and paths.")
    exit()

# <-- Split the Data into Training, Validation and Test -->
#Define the proportions first
val_ratio = 0.15
train_ratio = 0.70
test_ratio = 0.15

#Split into training + validation AND Test
input_train_val, input_test, target_train_val, target_test = train_test_split(
    all_input_files, 
    all_target_files, 
    test_size=test_ratio, 
    random_state=42)

#Split training + validation -> training AND validation 
val_split_ratio = val_ratio/ (1 - val_ratio)
input_train, input_val, target_train, target_val = train_test_split(
    input_train_val,
    target_train_val,
    test_size=val_split_ratio,
    random_state=42
)

# <---- Create DataLoaders for each split ---->
batch_size = 64
num_workers = 0

train_dataset = PianoReductionDataset(input_files=input_train,target_files=target_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

val_dataset = PianoReductionDataset(input_files=input_val, target_files=target_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

test_dataset = PianoReductionDataset(input_files=input_test, target_files=target_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


#Initialize and train the model
model = PianoReductionCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

trained_model = train_model(model, train_loader, val_loader, num_epochs=30, device=device) # Pass device and val_loader
torch.save(model.state_dict(), "first-train-model.pth") # Save state_dict




