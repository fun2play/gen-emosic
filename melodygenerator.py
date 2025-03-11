"""
melody_generator.py

This script defines the MelodyGenerator class, which is responsible for generating
melodies using a trained Transformer model. The class offers functionality to produce
a sequence of musical notes, starting from a given seed sequence and extending it
to a specified maximum length.

The MelodyGenerator class leverages the trained Transformer model's ability to
predict subsequent notes in a melody based on the current sequence context. It
achieves this by iteratively appending each predicted note to the existing sequence
and feeding this extended sequence back into the model for further predictions.

This iterative process continues until the generated melody reaches the desired length
or an end-of-sequence token is predicted. The class utilizes a tokenizer to encode and
decode note sequences to and from the format expected by the Transformer model.

Key Components:
- MelodyGenerator: The primary class defined in this script, responsible for the
  generation of melodies.

Usage:
The MelodyGenerator class can be instantiated with a trained Transformer model
and an appropriate tokenizer. Once instantiated, it can generate melodies by
calling the `generate` method with a starting note sequence.

Note:
This class is intended to be used with a Transformer model that has been
specifically trained for melody generation tasks.
"""
import os
import tensorflow as tf

######################## need to "set GPU Memory growth" before other "imports" #########
# forces TensorFlow to allocate memory gradually instead of pre-allocating the entire GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# following to avoid GPU OOM
from tensorflow.keras import mixed_precision
# Enable mixed precision (saves memory)
mixed_precision.set_global_policy("mixed_float16")
# Set TensorFlow GPU Allocator
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import numpy as np
from melodypreprocessor import MelodyPreprocessor
from transformer import Transformer
from music_utils import midi_to_tokens, tokens_to_midi
from music_utils import convert_midi_to_wav
from datetime import datetime
from keras.optimizers import Adam

# Define the checkpoint directory
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, "ckpt")
# Ensure directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Global parameters
EPOCHS = 10
BATCH_SIZE = 32
DATA_PATH = "data/input/midi" # "dataset.json"
MAX_POSITIONS_IN_POSITIONAL_ENCODING = 3000 # 2048 # 1500 # 1000 # 900 # 100
optimizer = Adam()

class MelodyGenerator:
    """
    Class to generate melodies using a trained Transformer model.

    This class encapsulates the inference logic for generating melodies
    based on a starting sequence.
    """

    def __init__(self, transformer, tokenizer, max_length=100): # 50):  TODO maybe try 900 later
        """
        MelodyGenerator generates melodies using a trained Transformer model.
        
        Parameters:
            transformer (Transformer): The trained Transformer model.
            tokenizer (Tokenizer): Tokenizer used for encoding melodies.
            max_length (int): Maximum length of the generated melodies.
        """
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.max_length = max_length

    def right_pad_sequence_once1(self, tf, sequence):
        """
        Pads a sequence with a single zero at the end.

        Parameters:
            sequence (tf.Tensor): The sequence to be padded.

        Returns:
            tf.Tensor: The padded sequence.
        """
        return tf.pad(sequence, [[0, 0], [0, 1]], "CONSTANT")

    def generate_melody(self, emotion, start_tokens, temperature=1.0):
        """
        Generate a melody based on the given emotion and start sequence.
        
        Parameters:
        - emotion: An integer (1,2,3,4) representing the emotion category.
        - start_tokens: List of tokens representing the initial melody sequence.
        - temperature: Controls randomness; higher values make the output more random.
        
        Returns:
        - List of generated melody tokens.
        """
        emotion_index = np.array([[self.tokenizer.word_index[emotion]]])  # Shape (1,1)
        sequence = np.array([[self.tokenizer.word_index[start_token] for start_token in start_tokens]])
        sequence = sequence.reshape(1, -1)  # Shape (1, seq_len)
        generated_sequence = np.empty((1, 0), dtype=int)  # Ensures correct shape

        # predictions = transformer(
        #     input,
        #     target=target_input,
        #     training=True,
        #     enc_padding_mask=None,
        #     look_ahead_mask=None
        #     dec_padding_mask=None
        # )

        stop_gen = False
        # emotion_input = tf.expand_dims(emotion, axis=1)  # Add sequence dimension  # fr train.py
        emotion_tensor = tf.convert_to_tensor(emotion_index, dtype=tf.int64)
        for i in range(self.max_length - len(start_tokens)):
            # Convert input to Tensor
            input_tensor = tf.convert_to_tensor(sequence, dtype=tf.int64)
            target = tf.pad(sequence[:, 1:], [[0, 0], [0, 1]], "CONSTANT") # right_pad_sequence_once1(tf, sequence[:, 1:])
            target =  tf.convert_to_tensor(target, dtype=tf.int64) # TODO move _right_pad_sequence_once to a central place

            # Get model predictions
            predictions = self.transformer( # TODO is  following target=sequence wrong? same as input now ? added target variable above to see the result difference
                input_tensor, emotion_tensor, target=target, training=False,
                enc_padding_mask=None, look_ahead_mask=None, dec_padding_mask=None
            )

            predicted_id = self._get_prediction_from_predictions(predictions)

            # Stop if end token is reached
            if stop_gen: # or (self.tokenizer.end_token and predicted_id == self.tokenizer.end_token): # TODO
                break

            # Append predicted token to sequence
            sequence = np.append(sequence, [[predicted_id]], axis=1)
            generated_sequence = np.append(generated_sequence, [[predicted_id]], axis=1)

#            predicted_note = self._get_note_with_highest_score(predictions)
#            input_tensor = self._append_predicted_note(
#                input_tensor, predicted_note
#            )

#        generated_melody = self._decode_generated_sequence(input_tensor)

#        return generated_melody
#         return sequence[0, len(start_tokens):].tolist()  # Return generated melody tokens
        return generated_sequence[0].tolist()  # Return generated melody tokens

    def _get_prediction_from_predictions(self, predictions, temperature=1):
        """
        Gets the note with the highest score from the predictions.
        Parameters:
            predictions (tf.Tensor): The predictions from the model.
        Returns:
            predicted_note (int): The index of the predicted note.
        """
        latest_predictions = predictions[:, -1, :] # Take the last time step
        ''' Temperature < 1.0 (e.g., 0.5) → Less random, more deterministic - More predictable (good for structured melodies)
            Temperature = 1.0 → Standard sampling behavior, no modification - Balanced randomness and structure
            Temperature > 1.0 (e.g., 1.5 or 2.0) → More random, more exploration - High creativity (good for experimental or jazz-like melodies)
        '''
        # latest_predictions = latest_predictions / temperature  # Adjust randomness
        # latest_predictions /= max(temperature, 1e-6)

        randomChoice = temperature > 1
        if (randomChoice):
            return tf.random.categorical(latest_predictions, num_samples=1).numpy().item() # [0, 0]
        else:
            return tf.argmax(latest_predictions, axis=1).numpy().item() # numpy()[0]

    def _append_predicted_note(self, input_tensor, predicted_note):
        """
        Appends the predicted note to the input tensor.

        Parameters:
            input_tensor (tf.Tensor): The input tensor for the model.

        Returns:
            (tf.Tensor): The input tensor with the predicted note
        """
        return tf.concat([input_tensor, [[predicted_note]]], axis=-1)

    def _decode_generated_sequence(self, generated_sequence):
        """
        Decodes the generated sequence of notes.

        Parameters:
            generated_sequence (tf.Tensor): Tensor with note indexes generated.

        Returns:
            generated_melody (str): The decoded sequence of notes.
        """
        generated_sequence_array = generated_sequence.numpy()
        generated_melody = self.tokenizer.sequences_to_texts(
            generated_sequence_array
        )[0]
        return generated_melody

def exitNow(yes=True):
    if yes:
        exit();

if __name__ == "__main__":
    print("Start melodygenerator.py: ", datetime.now())
    # melody_preprocessor = MelodyPreprocessor(DATA_PATH, batch_size=BATCH_SIZE)
    # vocab_size = 7000 # 6321 from melody_preprocessor.number_of_tokens_with_padding
    melody_preprocessor = MelodyPreprocessor(DATA_PATH, batch_size=BATCH_SIZE)
    quitNow = False
    # following just to get/initialize word_index, index_word etc value
    train_dataset = melody_preprocessor.create_training_dataset() # TODO train_dataset is not used ?
    print("finished : melody_preprocessor.create_training_dataset() ", datetime.now())
    exitNow(quitNow)
    vocab_size = melody_preprocessor.number_of_tokens_with_padding

    # MAX_POSITIONS_IN_POSITIONAL_ENCODING = max(2048, melody_preprocessor.max_melody_length) # or set it to a number ≥ longest sequence

    transformer_model = Transformer(
        num_layers=2,
        d_model=64,
        num_heads=2,
        d_feedforward=128,
        emotion_vocab_size = vocab_size,  # 5,  # Emotion tokens are from 1 to 4
        input_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        max_num_positions_in_pe_encoder=MAX_POSITIONS_IN_POSITIONAL_ENCODING,
        max_num_positions_in_pe_decoder=MAX_POSITIONS_IN_POSITIONAL_ENCODING,
        dropout_rate=0.1, # 0.1, # TODO increase this ?
    )

    print("Generating a melody...")
    melody_generator = MelodyGenerator(
        transformer_model, melody_preprocessor.tokenizer, 3000
    )

    ########### input for generation
    emotion = "1"
    data_path = "data/input/seed/"
    seedfilepathname = data_path + "seed_Q" + emotion + ".mid"
    start_tokens = midi_to_tokens(seedfilepathname)
    # seedfilepathname = "generated_melody_Q3.mid"
    emotion_label = "EQ" + emotion
    output_filepathname = data_path + "fr_seed_Q" + emotion + ".mid"

    # start_tokens = ["C4-1.0", "D4-1.0", "E4-1.0", "C4-1.0"]
    exitNow(quitNow)

    # Create a Checkpoint Manager
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=transformer_model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_DIR, max_to_keep=5)
    # Load the latest checkpoint
    if tf.train.latest_checkpoint(CHECKPOINT_DIR):
        checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR))
        print("Checkpoint restored. Resuming training from latest checkpoint...")
    else:
        print("No checkpoint found. Starting training from scratch...")

    exitNow(quitNow)
    new_melody = melody_generator.generate_melody(emotion_label, start_tokens)
    print(f"Generated melody: {new_melody}")

    exitNow(quitNow)

    tokens_to_midi(new_melody, output_filepathname)
    # looks like fluidsynth inside the following caused seg fault
    # convert_midi_to_wav("generated_melody.mid", "generated_melody.wav")

    exitNow(quitNow)

    print("Done: ", datetime.now())

    quitNow = True
    # exitNow(quitNow)

