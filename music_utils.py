from music21 import stream, note, midi
from midi2audio import FluidSynth
import numpy as np
import subprocess


def save_melody_as_midi(melody, filename="generated_melody.mid"):
    """
    Converts a generated melody list into a MIDI file.

    Parameters:
        melody (list of str): List of note-duration pairs, e.g., ["C4-1.0", "D4-1.0", "E4-1.0"].
        filename (str): Name of the MIDI file to save.
    """
    midi_stream = stream.Stream()

    melody_list = melody.split()
    for item in melody_list:
        pitch, duration = item.split('-')  # Example: "C4-1.0" → "C4", "1.0"
        duration = float(duration)  # Convert duration to float

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
    fs = FluidSynth()  # Uses default sound font and 44100 Hz sample rate by default
    # fs = FluidSynth('path/to/soundfont.sf2', sample_rate=22050) # Example with custom sound font and sample rate

    # Convert MIDI to WAV
    fs.midi_to_audio(midi_filename, wav_filename)

    print(f"Successfully converted '{midi_filename}' to '{wav_filename}'")

