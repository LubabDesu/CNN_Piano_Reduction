import pretty_midi 
import os
import numpy as np
import torch
from torch.utils.data import Dataset
# print(os.path.dirname(pretty_midi.__file__))

def load_piano_midi_file(filepath,fs=16) :
    """ 
    This function computes a piano roll matrix from the piano (presumably reudced) MIDI data
    
    Parameters : 
    - filepath : path to the Orchestral MIDI file to convert it into piano roll format
    - fs : Sampling rate, default = 16
    
    Returns : 
    - A numpy array with shape (128, T), where T signifies the number of time steps taken
    """
    midi_data = pretty_midi.PrettyMIDI(filepath)
    piano_notes = []
    for instrument in midi_data.instruments :
        if not instrument.is_drum :
            piano_notes.extend(instrument.notes)

    if piano_notes :
        temp_midi = pretty_midi.PrettyMIDI(piano_notes)
        temp_instrument = pretty_midi.Instrument(program = 0)
        temp_instrument.notes = piano_notes
        temp_midi.instruments.append(temp_instrument)

         # Get piano roll for the instrument
        piano_roll = temp_midi.get_piano_roll(fs=16)
        piano_roll = (piano_roll > 0).astype(np.float64)  # Normalize to 0-1

        if piano_roll.shape[0] >= 109:
                 piano_roll = piano_roll[21:109, :] # Shape (88, T)
        elif piano_roll.shape[0] >= 21:
                # Handle cases where MIDI might not use full 128 range but includes target notes
            piano_roll = np.pad(piano_roll, ((0, 109 - piano_roll.shape[0]),(0,0)), 'constant')[21:109, :]
        else:
            # Very unlikely, but handle case where no relevant notes exist
            piano_roll = np.zeros((88, piano_roll.shape[1]))
    
        original_time_steps = piano_roll.shape[1]
            # print("Original time steps : ", original_time_steps)
        if original_time_steps > 0:
                # Bin the notes into 64 time steps
            bin_size = max(1, original_time_steps // 64)  # Ensure at least 1
            resampled_roll = np.zeros((88, 64))
            for i in range(64):
                start_idx = i * bin_size
                end_idx = min((i + 1) * bin_size, original_time_steps)
                if start_idx < original_time_steps:
                    resampled_roll[:, i] = np.any(piano_roll[:, start_idx:end_idx], axis=1)
                piano_roll.append(resampled_roll)
            else:
                piano_roll.append(np.zeros((88, 64)))  # Handle empty case
        else:
            piano_roll.append(np.zeros((88, 64)))  # Handle empty category

        # Add channel dimension: [1, 88, 64]
        return np.expand_dims(resampled_roll, axis=0)

def load_midi_file(filepath,fs=16) :
    """ 
    This function computes a piano roll matrix from the orchestral MIDI data
    
    Parameters : 
    - filepath : path to the Orchestral MIDI file to convert it into piano roll format
    - fs : Sampling rate, default = 16
    
    Returns : 
    - A numpy array with shape (128, T), where T signifies the number of time steps taken
    """
    midi_data = pretty_midi.PrettyMIDI(filepath)

    categories = {
        'brass': ['trumpet', 'horn', 'trombone', 'tuba', 'trb'],  # Brass instruments
        'woodwinds_reeds': ['clarinet', 'oboe', 'bassoon'],  # Woodwinds with reeds
        'woodwinds_no_reeds': ['flute', 'picc'],  # Woodwinds without reeds
        'strings': ['violin', 'viola', 'cello', 'celli', 'double bass', 'bass solo', 'harp']  # Strings
    }

    # Get initial tempo
    if midi_data.estimate_tempo():
        bpm = midi_data.estimate_tempo()
    else:
        bpm = 120

    # Process each category
    piano_rolls = []
    
    #Collects the instrument notes for each category of instruments
    for category_name, name_keywords in categories.items():
        has_notes = False
        # Collect notes for this category
        category_notes = []
        
        for instrument in midi_data.instruments:
            # Check if the instrument name contains any of the category keywords
            instrument_name = instrument.name.lower()
            if any(keyword in instrument_name for keyword in name_keywords) and not instrument.is_drum:
                category_notes.extend(instrument.notes)
        
        new_midi = pretty_midi.PrettyMIDI()

        if category_notes:  # Only process if there are notes
            has_notes = True
            print(f"--- Processing TARGET file: {filepath} AND IT HAS NOTES!! ---")
            new_instrument = pretty_midi.Instrument(program=0)
            new_instrument.notes = category_notes
            new_midi.instruments.append(new_instrument)
            
            # Get piano roll for the new instrument
            piano_roll = new_midi.get_piano_roll(fs=16)
            piano_roll = (piano_roll > 0).astype(np.float64)  # Normalize to 0-1
            
            # Slice to notes 21 to 108 (88 notes)
            piano_roll = piano_roll[21:109, :]  # 21 to 108 inclusive
            
            # Resample time steps to 64

            #print("Original piano roll shape:", piano_roll.shape)
            print("Sum of piano roll before resampling:", piano_roll.sum())
            original_time_steps = piano_roll.shape[1]
            # print("Original time steps : ", original_time_steps)
            if original_time_steps > 0:
                # Bin the notes into 64 time steps
                bin_size = max(1, original_time_steps // 64)  # Ensure at least 1
                resampled_roll = np.zeros((88, 64))
                for i in range(64):
                    start_idx = i * bin_size
                    end_idx = min((i + 1) * bin_size, original_time_steps)
                    if start_idx < original_time_steps:
                        resampled_roll[:, i] = np.any(piano_roll[:, start_idx:end_idx], axis=1)
                piano_rolls.append(resampled_roll)
            else:
                piano_rolls.append(np.zeros((88, 64)))  # Handle empty case
        else:
            piano_rolls.append(np.zeros((88, 64)))  # Handle empty category


    # Stack the four piano rolls to get [4, 88, 64]
    if (has_notes == False) :
        print(f"--- Processing TARGET file: {filepath} AND IT DOESNT HAVE NOTES!! ---")
    stacked_rolls = np.stack(piano_rolls, axis=0)
    print("Sum of ALL elements in the FINAL stacked array:", stacked_rolls.sum()) # Check total non-zero elements
    return stacked_rolls


class PianoReductionDataset(Dataset):
    def __init__(self, input_files, target_files):
        self.inputs = [load_midi_file(f) for f in input_files]  # List of [4, 88, 64]
        self.targets = [load_piano_midi_file(f) for f in target_files]  # List of [1, 88, 64]
        #print("Test input files:", input_files)
        #print("Test target files:", target_files)
        assert len(self.inputs) == len(self.targets), "Input and target file counts must match"

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_tensor = torch.FloatTensor(self.inputs[idx]).permute(1, 2, 0)  # [88, 64, 4]
        # print(f"Input shape: {input_tensor.shape}")
        
        target_tensor = torch.FloatTensor(self.targets[idx]).permute(1, 2, 0)  # [88, 64, 1]
        # print(f"Target shape: {target_tensor.shape}")
        return {'input': input_tensor, 'target': target_tensor} 

