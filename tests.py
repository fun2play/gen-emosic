from miditok import REMI, midi_tokenizer
from pathlib import Path

# Path to the MIDI file
midi_file_path = Path("data/input/midi_200/Q3_c6CwY8Gbw0c_3.mid")

# Create a tokenizer instance (REMI is used as an example)
tokenizer = REMI()

# Tokenize the MIDI file
tokens = tokenizer(midi_file_path)

# Print the tokens
print(tokens.tokens)

# To save the tokens to a JSON file:
# tokens.save_tokens("path_to_save_tokens.json")