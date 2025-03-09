import pretty_midi 
import os
import numpy as np
import torch
from torch.utils.data import Dataset
# print(os.path.dirname(pretty_midi.__file__))


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

    # Define program number ranges for categories
    categories = {
        'brass': [1],  # Brass instruments
        'woodwinds_reeds': [1,71],  # Woodwinds with reeds
        'woodwinds_no_reeds': [1],  # Woodwinds without reeds (flutes, etc.)
        'strings': [1]  # Strings
    }

    # Get initial tempo
    if midi_data.estimate_tempo() :
        bpm = midi_data.estimate_tempo()
    else :
        bpm = 120

    # Calculate duration of 4 measures (assuming 4/4 time)
    # Each measure has 4 beats, each beat is 60/BPM seconds, so 4 measures = 16*(60/BPM)
    measure_duration = 16 * (60 / bpm)  # Duration in seconds for 4 measures

    # Process each category
    piano_rolls = []
    for category_name, program_range in categories.items():
        # Collect notes for this category
        category_notes = []
        for instrument in midi_data.instruments:
            if instrument.program in program_range and not instrument.is_drum:
                category_notes.extend(instrument.notes)


        piano_roll = midi_data.get_piano_roll(fs=fs)
        # Normalize to 0-1 range
        piano_roll = (piano_roll > 0).astype(np.float64)
        piano_roll = piano_roll[21:109, :]  # 21 to 108 inclusive

        new_midi = pretty_midi.PrettyMIDI()
        new_instrument = pretty_midi.Instrument(program=0)
        new_instrument.notes = category_notes
        new_midi.instruments.append(new_instrument)
            
            # Get piano roll for the entire duration with fs=16
        if category_notes:  # Only process if there are notes
            piano_roll = new_midi.get_piano_roll(fs=16)
            piano_roll = (piano_roll > 0).astype(np.float64)  # Normalize to 0-1
                
                # Slice to notes 21 to 108 (88 notes)
            piano_roll = piano_roll[21:109, :]  # 21 to 108 inclusive
                
                # Resample time steps to 64
            original_time_steps = piano_roll.shape[1]
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
        return np.stack(piano_rolls, axis=0)


class PianoReductionDataset(Dataset):
    def __init__(self, input_files, target_files):
        self.inputs = [load_midi_file(f) for f in input_files]  # List of [4, 88, 64]
        self.targets = [load_midi_file(f) for f in target_files]  # List of [1, 88, 64]
        assert len(self.inputs) == len(self.targets), "Input and target file counts must match"

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_tensor = torch.FloatTensor(self.inputs[idx]).permute(1, 2, 0)  # [88, 64, 4]
        # print(f"Input shape: {input_tensor.shape}")
        target_tensor = torch.FloatTensor(self.targets[idx]).permute(1, 2, 0)  # [88, 64, 1]
        return input_tensor, target_tensor



# orchesetral_roll = load_midi_file('CNN_Piano_Reduction/aligned_dataset/output_aligned_files/0/Brahms_Symph4_iv(1-33)_ORCH+REDUC+piano_orch.mid')
# print(f"Piano roll shape : {orchesetral_roll.shape}")
# print(orchesetral_roll[:, :50, :50])  # Print first 5 pitch rows and first 5 time steps

# midi_data = pretty_midi.PrettyMIDI('CNN_Piano_Reduction/aligned_dataset/output_aligned_files/0/Brahms_Symph4_iv(1-33)_ORCH+REDUC+piano_orch.mid')
# for instrument in midi_data.instruments:
#     print(instrument.name, instrument.program)

# all_notes = []
# for instrument in midi_data.instruments:
#     all_notes.extend([note.pitch for note in instrument.notes])
# print(min(all_notes), max(all_notes))