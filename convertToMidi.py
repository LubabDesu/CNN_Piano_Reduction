import numpy as np
import pretty_midi
import os # Optional: For specifying output directory
import torch
from models import PianoReductionCNN
from main import test_loader, test_dataset
def piano_roll_to_pretty_midi(piano_roll, fs=10, bpm=120, program=0, velocity=100, pitch_offset=21):
    """
    Converts a piano roll numpy array into a PrettyMIDI object.

    Parameters:
        piano_roll (np.ndarray): The piano roll array, shape=(num_pitches, num_timesteps).
                                 Assumes binary values (0 or 1).
        fs (int): Sampling frequency (used to determine time step duration relative to BPM).
                  Higher fs means shorter time steps for a given BPM note duration.
                  Think of this as "how many time steps make up one beat?".
                  If bpm=120, one beat is 0.5s. If fs=10, each step is 0.05s.
        bpm (float): Beats Per Minute for timing calculations.
        program (int): MIDI program number (instrument). 0 is Acoustic Grand Piano.
        velocity (int): Default velocity (loudness) for notes (1-127).
        pitch_offset (int): MIDI note number corresponding to the 0-th row of the piano roll.
                            Often 21 for standard 88-key piano rolls.

    Returns:
        pretty_midi.PrettyMIDI: A PrettyMIDI object representing the piano roll.
    """
    if piano_roll.ndim != 2:
        raise ValueError("piano_roll array must be 2D (pitches, time)")

    num_pitches, num_timesteps = piano_roll.shape

    # Create a PrettyMIDI object
    midi_obj = pretty_midi.PrettyMIDI(initial_tempo=bpm)

    # Create an Instrument instance (Acoustic Grand Piano)
    piano_instrument = pretty_midi.Instrument(program=program)

    # Calculate the duration of each time step in seconds based on fs and bpm
    # seconds_per_beat = 60.0 / bpm
    # time_step_duration = seconds_per_beat / fs # This interpretation might be confusing
    # Simpler: Let's define time_step_duration directly relative to fs.
    # If fs=10, assume each step is 1/10th of a second. Adjust fs to control speed.
    time_step_duration = 1.0 / fs

    # Iterate through the piano roll array
    for time_idx in range(num_timesteps):
        for pitch_idx in range(num_pitches):
            # Check if the note is 'on' (value > 0)
            if piano_roll[pitch_idx, time_idx] > 0.235:
                # Calculate the MIDI pitch number
                midi_pitch = pitch_idx + pitch_offset

                # Ensure pitch is within valid MIDI range (0-127)
                if 0 <= midi_pitch <= 127:
                    # Calculate start and end time in seconds
                    start_time = time_idx * time_step_duration
                    end_time = (time_idx + 1) * time_step_duration

                    # Create a Note instance
                    # Note: This simple version makes each note last exactly one time step.
                    # More complex logic could track note on/off across multiple steps.
                    note = pretty_midi.Note(
                        velocity=velocity,
                        pitch=midi_pitch,
                        start=start_time,
                        end=end_time
                    )
                    # Add the note to the instrument's note list
                    piano_instrument.notes.append(note)

    # Add the instrument to the PrettyMIDI object
    midi_obj.instruments.append(piano_instrument)

    return midi_obj


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PianoReductionCNN().to(device)
model.load_state_dict(torch.load("first-train-model.pth", map_location=device))
model.eval()

sample = next(iter(test_loader))
inp = sample['input'].permute(0, 3, 1, 2).to(device) # -> (B, 4, 88, 64)
with torch.no_grad():
    pred = model(inp) # Shape maybe (B, 1, 88, 64) = (B, C_out, P, T)

idx = 0

# 3. Process Prediction
pred_single_tensor = pred[idx, 0] # Select sample idx, channel 0 -> Shape: (88, 64) = (P, T)

pred_numpy = pred_single_tensor.detach().cpu().numpy() # Convert to NumPy -> (P, T)
print(f"pred_numpy shape: {pred_numpy.shape}")
print(f"Min value: {pred_numpy.min()}")
print(f"Max value: {pred_numpy.max()}")
print(f"Mean value: {pred_numpy.mean()}")
print(f"Median value: {np.median(pred_numpy)}")
print(f"Number of values > 0.5: {np.sum(pred_numpy > 0.5)}")
print(f"Number of values > 0.1: {np.sum(pred_numpy > 0.1)}")

# --- Get the original filename for the selected sample ---
try:
    # sample['input_filename'] is a list/tuple of filenames in the batch
    original_filename_with_ext = sample['input_filename'][idx]

    # Create a new output filename based on the original
    base_name = os.path.splitext(original_filename_with_ext)[0] # Get filename without extension
    OUTPUT_FILENAME = f"{base_name}_prediction.mid" # e.g., "original_song_prediction.mid"
    print(f"Retrieved original filename: {original_filename_with_ext}")

except (KeyError, IndexError, TypeError) as e:
    print(f"Warning: Could not retrieve original filename (Error: {e}). Using default name.")
    # Fallback filename if retrieval fails
    OUTPUT_FILENAME = f"prediction_sample_{idx}.mid"


OUTPUT_FILENAME = "my_model_prediction.mid"
OUTPUT_DIRECTORY = "CNN_Piano_Reduction/generated_midi" # Optional: Save to a specific folder
FS_PARAM = 10 # Controls how fast the MIDI plays. Higher = faster/shorter notes.
BPM_PARAM = 120 # Tempo
VELOCITY_PARAM = 100 # Loudness of notes
PITCH_OFFSET_PARAM = 21 # For 88-key piano roll starting at A0

# Ensure the output directory exists (optional)
if OUTPUT_DIRECTORY and not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)
    print(f"Created directory: {OUTPUT_DIRECTORY}")

output_path = os.path.join(OUTPUT_DIRECTORY, OUTPUT_FILENAME) if OUTPUT_DIRECTORY else OUTPUT_FILENAME

print(f"Converting piano roll (shape: {pred_numpy.shape}) to MIDI...")

# Convert the piano roll to a PrettyMIDI object
try:

    pretty_midi_obj = piano_roll_to_pretty_midi(
        pred_numpy,
        fs=FS_PARAM,
        bpm=BPM_PARAM,
        velocity=VELOCITY_PARAM,
        pitch_offset=PITCH_OFFSET_PARAM
    )

    # Save the PrettyMIDI object to a MIDI file
    pretty_midi_obj.write(output_path)

    print(f"Successfully saved MIDI file to: {output_path}")
    print("You can now listen to this file using a MIDI player.")

except Exception as e:
    print(f"An error occurred: {e}")
