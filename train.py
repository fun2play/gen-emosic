"""
This file contains the training pipeline for a Transformer model specialized in
melody generation. It includes functions to calculate loss, perform training steps,
and orchestrate the training process over multiple epochs. The script also
demonstrates the use of the MelodyGenerator class to generate a melody after training.

The training process uses a custom implementation of the Transformer model,
defined in the 'transformer.py' module, and prepares data using the
MelodyPreprocessor class from 'melodypreprocessor.py'.

Global parameters such as the number of epochs, batch size, and path to the dataset
are defined. The script supports dynamic padding of sequences and employs the
Sparse Categorical Crossentropy loss function for model training.

For simplicity's sake training does not deal with masking of padded values
in the encoder and decoder. Also, look-ahead masking is not implemented.
Both of these are left as an exercise for the student.

Key Functions:
- _calculate_loss_function: Computes the loss between actual and predicted sequences.
- _train_step: Executes a single training step, including forward pass and backpropagation.
- train: Runs the training loop over the entire dataset for a given number of epochs.
- _right_pad_sequence_once: Utility function for padding sequences.

The script concludes by instantiating the Transformer model, conducting the training,
and generating a sample melody using the trained model.
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

from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from melodygenerator import MelodyGenerator
from melodypreprocessor import MelodyPreprocessor
from transformer import Transformer
from datetime import datetime
from music_utils import midi_to_tokens, tokens_to_midi

# Bill added
# from tensorflow.keras.preprocessing.text import Tokenizer

# import os
#
# total_CPU_threads_to_use = 16
# # Set TensorFlow to use all CPU threads
# os.environ["TF_NUM_INTEROP_THREADS"] = str(total_CPU_threads_to_use)  # Parallel execution of independent operations
# os.environ["TF_NUM_INTRAOP_THREADS"] = str(total_CPU_threads_to_use)  # Parallel execution within an operation
#
# # Optionally, disable GPU usage if you want pure CPU execution
# # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#
# # Log CPU settings
# tf.config.threading.set_inter_op_parallelism_threads(total_CPU_threads_to_use)  # Parallel execution of independent ops
# tf.config.threading.set_intra_op_parallelism_threads(total_CPU_threads_to_use)  # Parallel execution within an op
#
# print("Num CPUs Available:", len(tf.config.list_physical_devices('CPU')))

# Define the checkpoint directory
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, "ckpt")
# Ensure directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Global parameters
EPOCHS = 1 # 10
BATCH_SIZE = 15 # 8 # 16 or 32 cause GPU OOM
DATA_PATH = "data/input/midi" # "dataset.json"
MAX_POSITIONS_IN_POSITIONAL_ENCODING = 2048 # 3000 # 1500 # 1000 # 900 # 100  900 is critical part to make training with shape mismatch issue

# Loss function and optimizer
sparse_categorical_crossentropy = SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"
)
optimizer = Adam()

def train(train_dataset, transformer, epochs, checkpoint):
    """
    Trains the Transformer model on a given dataset for a specified number of epochs.

    Parameters:
        train_dataset (tf.data.Dataset): The training dataset.
        transformer (Transformer): The Transformer model instance.
        epochs (int): The number of epochs to train the model.
        checkpoint (tf.train.Checkpoint): The checkpoint manager for saving model states.
    """
    print("Training the model...")
    for epoch in range(epochs):
        total_loss = 0
        # Iterate over each batch in the training dataset
        for (batch, (input, emotion, target)) in enumerate(train_dataset):
            try:
                # Perform a single training step
                batch_loss = _train_step(input, emotion=emotion, target=target, transformer=transformer)
                total_loss += batch_loss
                print(datetime.now(), f"Epoch {epoch + 1} Batch {batch + 1} Loss {batch_loss.numpy()}")
            except Exception as ex:
                print(f"Exception: {ex}")
                pass

        # Save the model after every epoch
        checkpoint.save(file_prefix=CHECKPOINT_PREFIX)
        # checkpoint_callback.on_epoch_end(epoch, logs={'loss': total_loss.numpy()})
        print(datetime.now(), f"Checkpoint saved at epoch {epoch + 1}")

@tf.function
def _train_step(input, emotion, target, transformer):
    """
    Performs a single training step for the Transformer model.

    Parameters:
        input (tf.Tensor): The input sequences.
        target (tf.Tensor): The target sequences.
        transformer (Transformer): The Transformer model instance.

    Returns:
        tf.Tensor: The loss value for the training step.
    """
    # Prepare the target input and real output for the decoder
    # Pad the sequences on the right by one position
    target_input = _right_pad_sequence_once(target[:, :-1])
    target_real = _right_pad_sequence_once(target[:, 1:])
    """ From Youtube:
        input [1, 2, 3, 4]
        target [2, 3, 4, 5]
        target_input [2, 3, 4, 0]
        target_real  [3, 4, 5, 0]
    """
    emotion_input = tf.expand_dims(emotion, axis=1)  # Add sequence dimension
    # Open a GradientTape to record the operations run
    # during the forward pass, which enables auto-differentiation
    with tf.GradientTape() as tape:
        # Forward pass through the transformer model
        # TODO: Add padding mask for encoder + decoder and look-ahead mask
        # for decoder
        # predictions = transformer(input, target_input, True, None, None, None)
        # enc_padding_mask = create_padding_mask(input)
        # look_ahead_mask = create_look_ahead_mask(tf.shape(target_input)[1])
        # dec_padding_mask = create_padding_mask(target_input)
        predictions = transformer(
            input,
            emotion=emotion_input,
            target=target_input,
            training=True,
            enc_padding_mask=None,
            look_ahead_mask=None,
            dec_padding_mask=None
        )

        # Compute loss between the real output and the predictions
        loss = _calculate_loss(target_real, predictions)

    # Calculate gradients with respect to the model's trainable variables
    gradients = tape.gradient(loss, transformer.trainable_variables)

    # Apply gradients to update the model's parameters
    gradient_variable_pairs = zip(gradients, transformer.trainable_variables)
    optimizer.apply_gradients(gradient_variable_pairs)

    # Return the computed loss for this training step
    return loss

# def create_padding_mask(seq):
#     # Create a mask for padding (0 values in the sequence).
#     # Output shape: (batch_size, 1, 1, seq_len)
#     mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
#     return mask[:, tf.newaxis, tf.newaxis, :]  # Add extra dimensions for compatibility
#
# def create_look_ahead_mask(size):
#     # Create a triangular matrix with 1s in the upper-right triangle
#     # This mask ensures the model cannot look ahead in the sequence
#     mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
#     return mask  # Shape: (seq_len, seq_len)

def _calculate_loss(real, pred):
    """
    Computes the loss between the real and predicted sequences.

    Parameters:
        real (tf.Tensor): The actual target sequences.
        pred (tf.Tensor): The predicted sequences by the model.

    Returns:
        average_loss (tf.Tensor): The computed loss value.
    """

    # Compute loss using the Sparse Categorical Crossentropy
    loss_ = sparse_categorical_crossentropy(real, pred)

    # Create a mask to filter out zeros (padded values) in the real sequences
    boolean_mask = tf.math.equal(real, 0)
    mask = tf.math.logical_not(boolean_mask)

    # Convert mask to the same dtype as the loss for multiplication
    mask = tf.cast(mask, dtype=loss_.dtype)

    # Apply the mask to the loss, ignoring losses on padded positions
    loss_ *= mask

    # Calculate average loss, excluding the padded positions
    total_loss = tf.reduce_sum(loss_)
    number_of_non_padded_elements = tf.reduce_sum(mask)
    average_loss = total_loss / number_of_non_padded_elements

    return average_loss


def _right_pad_sequence_once(sequence):
    """
    Pads a sequence with a single zero at the end.

    Parameters:
        sequence (tf.Tensor): The sequence to be padded.

    Returns:
        tf.Tensor: The padded sequence.
    """
    # max_length = 1024  # Ensure padding up to max length
    # padding_size = max_length - tf.shape(sequence)[1]
    # return tf.pad(sequence, [[0, 0], [0, padding_size]], "CONSTANT")
    return tf.pad(sequence, [[0, 0], [0, 1]], "CONSTANT")


if __name__ == "__main__":
    print("Start train.py: ", datetime.now())
    melody_preprocessor = MelodyPreprocessor(DATA_PATH, batch_size=BATCH_SIZE)
    train_dataset = melody_preprocessor.create_training_dataset()
    print("finished : melody_preprocessor.create_training_dataset() ", datetime.now())
    vocab_size = melody_preprocessor.number_of_tokens_with_padding

    MAX_POSITIONS_IN_POSITIONAL_ENCODING = max(2048, melody_preprocessor.max_melody_length) # or set it to a number ≥ longest sequence

    transformer_model = Transformer(
        num_layers=2,
        d_model=64,
        num_heads=2,
        d_feedforward=128,
        emotion_vocab_size=vocab_size,
        input_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        max_num_positions_in_pe_encoder=MAX_POSITIONS_IN_POSITIONAL_ENCODING,
        max_num_positions_in_pe_decoder=MAX_POSITIONS_IN_POSITIONAL_ENCODING,
        dropout_rate=0.1,
    )

    # Create a Checkpoint Manager
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=transformer_model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_DIR, max_to_keep=5)
    # Load the latest checkpoint
    if tf.train.latest_checkpoint(CHECKPOINT_DIR):
        checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR))
        print("Checkpoint restored. Resuming training from latest checkpoint...")
    else:
        print("No checkpoint found. Starting training from scratch...")



    # # In train.py, after creating your model and before training:
    # dummy_seq_len = melody_preprocessor.max_melody_length  # should be 1024 now
    # dummy_inp = tf.random.uniform(
    #     (1, dummy_seq_len), dtype=tf.int64, minval=0, maxval=vocab_size - 1
    # )
    # dummy_tar = tf.random.uniform(
    #     (1, dummy_seq_len), dtype=tf.int64, minval=0, maxval=vocab_size - 1
    # )
    # emotion_input = tf.expand_dims(list([1]), axis=1)  # Add sequence dimension, using 1 as dummy
    # # Build the model with the correct sequence length:
    # transformer_model(
    #     dummy_inp,
    #     emotion_input,
    #     dummy_tar,
    #     training=False,
    #     enc_padding_mask=None,
    #     look_ahead_mask=None,
    #     dec_padding_mask=None,
    # )



    # Train and save after each epoch
    train(train_dataset, transformer=transformer_model, epochs=EPOCHS, checkpoint=checkpoint)

    print("Done with training: ", datetime.now())



    print("Generating a melody...")
    melody_generator = MelodyGenerator(
        transformer_model, melody_preprocessor.tokenizer, 2500
    )

    generate_music = False # True
    generate_num = 1
    while generate_music and generate_num <= 5:
        # input for generation
        seedfilepathname = "data/input/midi_200/Q3_c6CwY8Gbw0c_3.mid" # "data/input/seed/seed_Q4.mid"
        # seedfilepathname = "generated_melody_Q3.mid"
        emotion_label = "EQ3"

        start_sequence = midi_to_tokens(seedfilepathname)
        # start_sequence = ["C4-1.0", "D4-1.0", "E4-1.0", "C4-1.0"]

        # # Create a Checkpoint Manager
        # checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=transformer_model)
        # checkpoint_manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_DIR, max_to_keep=5)
        # # Load the latest checkpoint
        # if tf.train.latest_checkpoint(CHECKPOINT_DIR):
        #     checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR))
        #     print("Checkpoint restored. Resuming training from latest checkpoint...")
        # else:
        #     print("No checkpoint found. Starting training from scratch...")

        new_melody = melody_generator.generate_melody(emotion_label, start_sequence)
        print(f"Generated melody: {new_melody}")

        tokens_to_midi(new_melody, "generated_melody.mid")
        # looks like fluidsynth inside the following caused seg fault
        # convert_midi_to_wav("generated_melody.mid", "generated_melody.wav")

        generate_num += 1
        print("Done: ", datetime.now())

    pass
