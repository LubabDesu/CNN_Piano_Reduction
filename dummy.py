import pretty_midi
import numpy as np
import os # Make sure os is imported

# -----------------------------------------------------------------------------
# Your load_midi_file function (with debugging enhancements)
# -----------------------------------------------------------------------------
def load_midi_file(filepath, fs=16):
    """
    This function computes a piano roll matrix from the orchestral MIDI data

    Parameters :
    - filepath : path to the Orchestral MIDI file to convert it into piano roll format
    - fs : Sampling rate, default = 16

    Returns :
    - A numpy array with shape (4, 88, 64) if notes are found, or (4, 88, 64) of zeros
    - Total duration of the file processed in seconds
    - Filename
    """
    print(f"--- Starting to load MIDI file: {filepath} ---")
    try:
        midi_data = pretty_midi.PrettyMIDI(filepath)
    except Exception as e:
        print(f"Error loading MIDI file {filepath}: {e}")
        # Return a placeholder structure that matches what the calling code might expect
        # Ensure filename is still extracted for consistent return signature
        filename = os.path.basename(filepath).split('/')[-1]
        return np.zeros((4, 88, 64)), 0, filename

    filename = os.path.basename(filepath).split('/')[-1]

    total_duration_in_seconds = midi_data.get_end_time()
    print(f"Total duration: {total_duration_in_seconds} seconds")

    categories = {
        'brass': ['trumpet', 'horn', 'trombone', 'tuba', 'trb'],
        'woodwinds_reeds': ['clarinet', 'oboe', 'bassoon'],
        'woodwinds_no_reeds': ['flute', 'picc'],
        'strings': ['violin', 'viola', 'cello', 'celli', 'double bass', 'bass solo', 'harp']
    }

    bpm = midi_data.estimate_tempo() if midi_data.estimate_tempo() else 120
    print(f"Estimated BPM: {bpm}")

    piano_rolls = []
    overall_has_notes_in_file = False

    print("\n--- Instrument Details ---")
    if not midi_data.instruments:
        print("No instruments found in this MIDI file.")
    for i, instrument in enumerate(midi_data.instruments):
        print(f"  Instrument {i}: Name='{instrument.name}', Program={instrument.program}, Is Drum={instrument.is_drum}, Notes Count={len(instrument.notes)}")

    print("\n--- Processing Categories ---")
    for category_name, name_keywords in categories.items():
        print(f"\nProcessing category: {category_name}")
        has_notes_in_category = False
        category_notes = []

        for instrument in midi_data.instruments:
            instrument_name_lower = instrument.name.lower()
            matched_keyword = None
            # Check if any keyword for this category is in the instrument name
            if any((keyword in instrument_name_lower) for keyword in name_keywords):
                matched_keyword = next((kw for kw in name_keywords if kw in instrument_name_lower), None)

            if matched_keyword and not instrument.is_drum:
                print(f"  Instrument '{instrument.name}' (matched keyword '{matched_keyword}') added to category '{category_name}'. Number of notes: {len(instrument.notes)}")
                if instrument.notes:
                    category_notes.extend(instrument.notes)
                    has_notes_in_category = True # Mark that this category has notes
                    overall_has_notes_in_file = True # Mark that the file overall has notes
                else:
                    print(f"    Instrument '{instrument.name}' matched but has no actual note objects.")
            elif not instrument.is_drum and any((keyword in instrument_name_lower) for keyword in name_keywords):
                 print(f"  Instrument '{instrument.name}' matched a keyword but was skipped (is_drum={instrument.is_drum} or other condition).")


        if category_notes: # If notes were collected for this category
            print(f"  Category '{category_name}' HAS {len(category_notes)} NOTES!!")
            new_midi = pretty_midi.PrettyMIDI() # Create a new MIDI object for this category's notes
            new_instrument = pretty_midi.Instrument(program=0) # Program 0 is Acoustic Grand Piano
            new_instrument.notes = category_notes
            new_midi.instruments.append(new_instrument)

            piano_roll = new_midi.get_piano_roll(fs=fs)
            piano_roll = (piano_roll > 0).astype(np.float64) # Normalize to 0 or 1

            # Slice to standard 88 piano keys (MIDI notes 21 to 108)
            if piano_roll.shape[0] >= 109:
                piano_roll_88 = piano_roll[21:109, :]
            else:
                # Handle cases where the original piano roll doesn't span the full 128 MIDI notes
                print(f"  WARNING: Piano roll for category '{category_name}' has fewer than 109 pitches ({piano_roll.shape[0]}). Padding/adjusting for 88 key output.")
                piano_roll_88 = np.zeros((88, piano_roll.shape[1])) # Default to zeros
                # Attempt to copy the relevant part if possible
                if piano_roll.shape[0] > 21: # If there are notes above pitch 21
                    # Copy the available pitches from original_roll[21:] into piano_roll_88
                    max_pitch_to_copy = min(109, piano_roll.shape[0]) # Don't try to read beyond what original_roll has
                    num_pitches_to_copy = max_pitch_to_copy - 21
                    if num_pitches_to_copy > 0:
                         piano_roll_88[:num_pitches_to_copy, :] = piano_roll[21:max_pitch_to_copy, :]
                # else: the piano_roll_88 remains all zeros for this category if no notes in range 21-108

            original_time_steps = piano_roll_88.shape[1]
            # print(f"  Piano roll shape for '{category_name}' (88 keys): {piano_roll_88.shape}")

            if original_time_steps > 0:
                # Resample time steps to 64
                bin_size = max(1, original_time_steps // 64)
                resampled_roll = np.zeros((88, 64))
                for i in range(64):
                    start_idx = i * bin_size
                    end_idx = min((i + 1) * bin_size, original_time_steps)
                    if start_idx < original_time_steps:
                        resampled_roll[:, i] = np.any(piano_roll_88[:, start_idx:end_idx], axis=1)
                piano_rolls.append(resampled_roll)
                # print(f"  Resampled roll sum for '{category_name}': {resampled_roll.sum()}")
            else:
                print(f"  Category '{category_name}' had notes, but 88-key piano roll has 0 time steps. Appending zeros (88, 64).")
                piano_rolls.append(np.zeros((88, 64)))
        else:
            print(f"  Category '{category_name}' has NO notes. Appending zeros (88, 64).")
            piano_rolls.append(np.zeros((88, 64)))

    if not overall_has_notes_in_file:
        print(f"--- Concluding: File: {filepath} DOESN'T APPEAR TO HAVE ANY MATCHED NOTES in any category!! ---")
    else:
        print(f"--- Concluding: File: {filepath} HAD NOTES in one or more categories. ---")

    # Ensure piano_rolls has 4 elements before stacking, even if some are empty
    while len(piano_rolls) < len(categories):
        print("Warning: Fewer piano rolls than categories. Appending empty roll(s).")
        piano_rolls.append(np.zeros((88, 64)))
    if len(piano_rolls) > len(categories): # Should ideally not happen with current logic
         print("Warning: More piano rolls than categories. Truncating to expected number.")
         piano_rolls = piano_rolls[:len(categories)]


    stacked_rolls = np.stack(piano_rolls, axis=0)
    # print(f"Sum of ALL elements in the FINAL stacked array for {filename}: {stacked_rolls.sum()}")
    return stacked_rolls, total_duration_in_seconds, filename

# -----------------------------------------------------------------------------
# Main part of the script for testing
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Define the list of MIDI files you want to test.
    #    CRITICAL: Replace these with the actual paths to YOUR MIDI files!
    midi_files_to_test = [
        "aligned_dataset/aligned/hand_picked_Spotify-10/orchestra.mid",
        "aligned_dataset/aligned/bouliane-39/orchestra.mid",
        "aligned_dataset/aligned/bouliane-37/orchestra.mid",
        "aligned_dataset/aligned/bouliane-38/orchestra.mid",
        "aligned_dataset/aligned/bouliane-30/orchestra.mid"
        
    ]

    # Create a 'test_midis' directory if you plan to put your files there
    # and want the script to create it. Adjust if your files are elsewhere.
    # os.makedirs("test_midis", exist_ok=True)

    if not midi_files_to_test:
        print("The 'midi_files_to_test' list is empty. Please add MIDI file paths to test.")
    else:
        print(f"Found {len(midi_files_to_test)} file(s) to test.")

    # 2. Loop through each file path and call your function.
    for midi_path in midi_files_to_test:
        print(f"\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(f"--- TESTING FILE: {midi_path} ---")
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        if not os.path.exists(midi_path):
            print(f"WARNING: MIDI file not found at '{midi_path}'. Skipping.")
            continue

        stacked_piano_roll, duration, filename = load_midi_file(midi_path, fs=16)

        # 3. Print the results or do further checks.
        if stacked_piano_roll is not None: # load_midi_file should always return something now
            print(f"\n--- RESULTS for '{filename}' ---")
            print(f"Processed '{filename}': Duration={duration:.2f}s, Output shape={stacked_piano_roll.shape}")
            total_notes_in_output = stacked_piano_roll.sum()
            print(f"Sum of all notes in final stacked roll: {total_notes_in_output}")
            if total_notes_in_output == 0 and duration > 0:
                 print(f"POTENTIAL ISSUE: Stacked roll for '{filename}' is all zeros, but MIDI duration is {duration:.2f}s.")
            elif total_notes_in_output > 0:
                print(f"SUCCESS: Notes found and processed for '{filename}'.")

        # The 'else' case for stacked_piano_roll being None should ideally not be hit
        # with the current error handling in load_midi_file, but kept for safety.
        else:
            print(f"CRITICAL ERROR: Failed to get a valid result from load_midi_file for '{filename}'.")
            print(f"This indicates an issue in the return values of load_midi_file itself when an error occurs.")

    print("\n--- All test files processed ---")