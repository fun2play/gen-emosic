from music21 import stream, note, midi
from midi2audio import FluidSynth
import numpy as np
import subprocess

from music21 import converter, note, chord
import json

def OLD_save_melody_as_midi(melody, filename="generated_melody.mid"):
    """
    Converts a generated melody list into a MIDI file.

    Parameters:
        melody (list of str): List of note-duration pairs, e.g., ["C4-1.0", "D4-1.0", "E4-1.0"].
        filename (str): Name of the MIDI file to save.
    """
    midi_stream = stream.Stream()

    melody_list = melody # melody.split(",") # melody[0].split(",")
    for item in melody_list:
        pitch, duration = item.rsplit('-',1)  # Example: "C4-1.0" → "C4", "1.0"
        pitch = pitch.replace('-', 'b') # TODO hack E-4 to E4
        duration = eval(duration)  # Convert duration to float

        # Create a music21 Note and set its duration
        n = note.Note(pitch)
        n.duration.quarterLength = duration

        midi_stream.append(n)

    # Write to MIDI file
    midi_stream.write('midi', fp=filename)
    print(f"MIDI file saved: {filename}")


def convert_midi_to_wav(midi_filename="generated_melody.mid", wav_filename="generated_melody.wav",
                        soundfont="default.sf2"):
    """
    Converts a MIDI file into a WAV file using FluidSynth.

    Parameters:
        midi_filename (str): Path to the MIDI file.
        wav_filename (str): Path to save the WAV file.
        soundfont (str): Path to a SoundFont file (.sf2) for rendering.
    """
    # # Use a default soundfont if not provided
    # if soundfont == "default.sf2":
    #     soundfont = "/usr/share/sounds/sf2/FluidR3_GM.sf2"  # Standard soundfont location on Linux
    #
    # # Convert MIDI to WAV using FluidSynth
    # command = ["fluidsynth", "-ni", soundfont, midi_filename, "-F", wav_filename, "-r", "44100"]
    # subprocess.run(command, check=True)

    # Initialize FluidSynth with optional sound font and sample rate parameters
    # fs = FluidSynth()  # Uses default sound font and 44100 Hz sample rate by default
    fs = FluidSynth('/usr/share/sounds/sf2/FluidR3_GM.sf2') # , sample_rate=22050) # Example with custom sound font and sample rate

    # Convert MIDI to WAV
    fs.midi_to_audio(midi_filename, wav_filename)

    print(f"Successfully converted '{midi_filename}' to '{wav_filename}'")


def OLD_midi_to_tokens(midi_file):
    """
    Converts a MIDI file to the dataset.json format.

    Parameters:
        midi_file (str): Path to the MIDI file.

    Returns:
        str: A string of note-duration pairs separated by commas.
    """
    midi = converter.parse(midi_file)
    notes = []

    for element in midi.recurse():
        if isinstance(element, note.Note):
            pitch = element.pitch.nameWithOctave
            duration = element.quarterLength
            notes.append(f"{pitch}-{duration}")
        elif isinstance(element, chord.Chord):
            # For chords, choose the root note
            pitch = element.root().nameWithOctave
            duration = element.quarterLength
            notes.append(f"{pitch}-{duration}")

    return notes # return ", ".join(notes)



# def save_as_json(midi_files, output_file="converted_dataset.json"):
#     """
#     Converts a list of MIDI files to dataset format and saves them as JSON.
#
#     Parameters:
#         midi_files (list): List of MIDI file paths.
#         output_file (str): Path to save the JSON file.
#     """
#     dataset = [midi_to_tokens(midi) for midi in midi_files]
#
#     with open(output_file, "w") as f:
#         json.dump(dataset, f, indent=4)
#
#     print(f"Dataset saved to {output_file}")

###########################################################################
#### start MIDO ###########################################################
import mido
from mido import MidiFile, MidiTrack, Message
import re

NOTE_MAPPING = {
    21: "A0", 22: "A#0", 23: "B0", 24: "C1", 25: "C#1", 26: "D1", 27: "D#1",
    28: "E1", 29: "F1", 30: "F#1", 31: "G1", 32: "G#1", 33: "A1", 34: "A#1",
    35: "B1", 36: "C2", 37: "C#2", 38: "D2", 39: "D#2", 40: "E2", 41: "F2",
    42: "F#2", 43: "G2", 44: "G#2", 45: "A2", 46: "A#2", 47: "B2", 48: "C3",
    49: "C#3", 50: "D3", 51: "D#3", 52: "E3", 53: "F3", 54: "F#3", 55: "G3",
    56: "G#3", 57: "A3", 58: "A#3", 59: "B3", 60: "C4", 61: "C#4", 62: "D4",
    63: "D#4", 64: "E4", 65: "F4", 66: "F#4", 67: "G4", 68: "G#4", 69: "A4",
    70: "A#4", 71: "B4", 72: "C5", 73: "C#5", 74: "D5", 75: "D#5", 76: "E5",
    77: "F5", 78: "F#5", 79: "G5", 80: "G#5", 81: "A5", 82: "A#5", 83: "B5",
    84: "C6", 85: "C#6", 86: "D6", 87: "D#6", 88: "E6", 89: "F6", 90: "F#6",
    91: "G6", 92: "G#6", 93: "A6", 94: "A#6", 95: "B6", 96: "C7", 97: "C#7",
    98: "D7", 99: "D#7", 100: "E7", 101: "F7", 102: "F#7", 103: "G7",
    104: "G#7", 105: "A7", 106: "A#7", 107: "B7", 108: "C8"
}
REVERSE_MAPPING = {v: k for k, v in NOTE_MAPPING.items()}

def mido_midi_to_tokens(midi_file_path):
    """Convert MIDI file to tokens."""
    mid = MidiFile(midi_file_path)
    tokens = []
    for track in mid.tracks:
        time_counter = 0
        for msg in track:
            time_counter += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                note_name = NOTE_MAPPING.get(msg.note, f"UNK{msg.note}")
                duration = msg.time / mid.ticks_per_beat  # Convert to beats
                tokens.append(f"{note_name}-{duration:.2f}")
    return tokens

def mido_tokens_to_midi(tokens, output_midi_path, ticks_per_beat=480):
    """Convert tokens back to MIDI file."""
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    for token in tokens:
        match = re.match(r'([A-G]#?\d+)-([\d.]+)', token)
        if match:
            note_name, duration = match.groups()
            note_value = REVERSE_MAPPING.get(note_name, 60)  # Default to C4 if not found
            duration_ticks = int(float(duration) * ticks_per_beat)

            # Note ON
            track.append(Message('note_on', note=note_value, velocity=64, time=0))
            # Note OFF
            track.append(Message('note_off', note=note_value, velocity=64, time=duration_ticks))

    mid.save(output_midi_path)
    print(f"Saved MIDI to {output_midi_path}")\


#################################################################################
### full  F5-1.0 format #########################################################

#####################################################################
### MidiTok #########################################################
from miditok import REMI, TokenizerConfig
from symusic import Score

def midiTok_midi_to_tokens(midi_file_path):
    # Creating a multitrack tokenizer, read the doc to explore all the parameters
    config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True)
    tokenizer = REMI(config)

    # Loads a midi, converts to tokens, and back to a MIDI
    midi = Score(midi_file_path)
    tokens = tokenizer(midi)  # calling the tokenizer will automatically detect MIDIs, paths and tokens
    return tokens

def midiTok_tokens_to_midi(tokens, output_midi_path, ticks_per_beat=480):
    # Creating a multitrack tokenizer, read the doc to explore all the parameters
    config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True)
    tokenizer = REMI(config)

    converted_back_midi = tokenizer(tokens)  # PyTorch, Tensorflow and Numpy tensors are supported

    # Save the MIDI file
    converted_back_midi.dump_midi(output_midi_path)
    converted_back_midi.dump_abc(output_midi_path + ".abc")

    print("MIDI file successfully saved as 'fr_seed_Q1_by_midiTok.mid'")

#####################################################################################################
### interface pair: #################################################################################
def midi_to_tokens(midi_file_path):
    return midiTok_midi_to_tokens(midi_file_path)
    # return OLD_midi_to_tokens(midi_file_path)

def tokens_to_midi(tokens, output_midi_path, ticks_per_beat=480):
    midiTok_tokens_to_midi(tokens, output_midi_path, ticks_per_beat)
    # OLD_save_melody_as_midi(tokens, output_midi_path)


# Example usage:
if __name__ == '__main__':
    # Replace with your actual file paths
    input_midi = "data/input/seed/seed_Q1.mid" # "data/input/seed/seed_Q4.mid"
    output_midi = "data/input/seed/test_fr_seed_Q1.mid"

    # Convert MIDI file to tokens
    tokens = midi_to_tokens(input_midi)
    print("Tokens generated from MIDI:", tokens)

    # Convert tokens back to a MIDI file
    tokens_to_midi(tokens, output_midi)
    print("MIDI file saved as:", output_midi)

    # # Example usage:
    # # Provide a list of MIDI files to convert
    # # midi_files = ["example1.mid", "example2.mid"]
    # # save_as_json(midi_files, output_file="converted_dataset.json")
    # tokens = midi_to_tokens(input_midi)
    # save_melody_as_midi(tokens, output_midi)

    pass