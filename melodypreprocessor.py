"""
melody_preprocessor.py

This script defines the MelodyPreprocessor class, a utility for preparing melody
datasets for training in a sequence-to-sequence Transformer model. The class
focuses on processing melody data by tokenizing and encoding the melodies, and
subsequently creating TensorFlow datasets suitable for training sequence-to-sequence
models.

The MelodyPreprocessor handles the entire preprocessing pipeline including loading
melodies from a dataset file, parsing the melodies into individual notes, tokenizing
and encoding these notes, and forming input-target pairs for model training. It
also includes functionality for padding sequences to a uniform length.

Key Features:
- Tokenization and encoding of melodies.
- Dynamic calculation of maximum sequence length based on the dataset.
- Creation of input-target pairs for sequence-to-sequence training.
- Conversion of processed data into TensorFlow datasets.

Usage:
To use the MelodyPreprocessor, initialize it with the path to a dataset containing
melodies and the desired batch size. Then call `create_training_dataset` to prepare
the dataset for training a Transformer model.

Note:
This script is intended to be used with datasets containing melody sequences in a
specific format, where each melody is represented as a string of comma-separated
musical notes (pitch with octave + duration in quarter length).
"""
import json
import multiprocessing
import os
import sys

import numpy as np
import tensorflow as tf

from misc_utils import MULTIPROCESS_loop, available_memory_gb
from music_utils import midi_to_tokens

# from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer

from pympler import asizeof

class MelodyPreprocessor:
    """
    A class for preprocessing melodies for a Transformer model.

    This class takes melodies, tokenizes and encodes them, and prepares
    TensorFlow datasets for training sequence-to-sequence models.
    """

    def __init__(self, dataset_path, batch_size=32):
        """
        Initializes the MelodyPreprocessor.

        Parameters:
            dataset_path (str): Path to the dataset file.
            max_melody_length (int): Maximum length of the sequences.
            batch_size (int): Size of each batch in the dataset.
        """
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.tokenizer = Tokenizer(filters="", lower=False, split=",")
        self.max_melody_length = None
        self.number_of_tokens = None

    @property
    def number_of_tokens_with_padding(self):
        """
        Returns the number of tokens in the vocabulary including padding.

        Returns:
            int: The number of tokens in the vocabulary including padding.
        """
        return self.number_of_tokens + 1

    def create_training_dataset(self):
        """
        Preprocesses the melody dataset and creates sequence-to-sequence
        training data.

        Returns:
            tf_training_dataset: A TensorFlow dataset containing input-target
                pairs suitable for training a sequence-to-sequence model.
        """
        # ataset_with_emotions: [('EQ2', TokSequence),()]
        dataset_with_emotions = self._load_dataset_with_emotions_multiprocessing()  # Use emotion-tagged dataset
        parsed_melodies = [(emotion, melody) for emotion, melody in dataset_with_emotions]
        # "emotion" in following is token ('EQ2') instead of Id.
        # BUT "melodies" are Ids/sequences. TODO Following need to be sync-ed for word_index etc setting. so More to do to optimize loop in next line
        tokenized_melodies = [(emotion, self._tokenize_and_encode_melody(emotion, melody)) for emotion, melody in parsed_melodies] # TODO [(3,[]),(...

        self._set_max_melody_length([melody for _, melody in tokenized_melodies])
        self._set_number_of_tokens()

        # Create a tf.data.Dataset from the generator.
        output_signature = (
            tf.TensorSpec(shape=(self.max_melody_length,), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.int32),  # emotion token is a scalar
            tf.TensorSpec(shape=(self.max_melody_length,), dtype=tf.int32)
        )
        # _create_sequence_pairs_generator below will convert emotion 'EQ2' to its Id
        dataset = tf.data.Dataset.from_generator(
            lambda: self._create_sequence_pairs_generator(tokenized_melodies, stride=5),
            output_signature=output_signature
        )
        dataset = dataset.shuffle(buffer_size=1000).batch(self.batch_size)
        return dataset

        # input_sequences, emotion_sequences, target_sequences = self._create_sequence_pairs(tokenized_melodies)
        # tf_training_dataset = self._convert_to_tf_dataset(input_sequences, emotion_sequences, target_sequences)
        # return tf_training_dataset

    # def _load_dataset(self): # TODO to be removed as it's replaced by _load_dataset_with_emotions
    #     """
    #     Loads the melody dataset from a JSON file.
    #
    #     Returns:
    #         list: A list of melodies from the dataset.
    #     """
    #     with open(self.dataset_path, "r") as f:
    #         return json.load(f)

    def process_midi_file(self, data):
        """
        Process a single MIDI file: extract emotion and tokenize melody.
        """
        filename, midi_folder = data
        # example name "Q1_0vLPYiPN7qY_0.mid"
        if not (filename.endswith(".mid") or filename.endswith(".midi")):
            return None  # Skip non-MIDI files
        if len(filename) < 4:
            return None  # Skip invalid filenames

        # filename.split("_")[0]  # Set prefix EQ as EQ1, EQ2, EQ3, EQ4 to be UNIQUE emotions words and different from music tokens words
        emotion = "EQ" + filename[1]  # Unique emotion prefix (EQ1, EQ2, EQ3, EQ4)
        melody = midi_to_tokens(os.path.join(midi_folder, filename))  # Tokenize MIDI
        # Truncate melody to a fixed length to speed up training
        MAX_MELODY_LENGTH = 1024  # Adjust this based on your memory constraints
        # melody = melody[:MAX_MELODY_LENGTH] # TODO TODO

        ''' truncate to 1024 will get following error. TODO why
    File "/home/user96/Dev/ai-musicGen/Bill/gen-emosic/venv-3.8.1_dcontainer/lib/python3.8/site-packages/keras/src/layers/core/dense.py", line 244, in call
      outputs = tf.tensordot(inputs, self.kernel, [[rank - 1], [0]])
Node: 'transformer/dense_8/Tensordot'
Input to reshape is a tensor with 3225600 values, but the requested shape has 3210240
	 [[{{node transformer/dense_8/Tensordot}}]] [Op:__inference__train_step_13804]
'''
        # if (len(melody) > 1024):
        #     melody = melody[:1024]
            # return (emotion, [])  # Empty melody instead of None
            # return None

        return (emotion, melody)  # Return the processed result

    def _load_dataset_with_emotions_multiprocessing(self):
        """
        Load and process MIDI files using multiprocessing.
        """
        midi_folder = self.dataset_path
        filenames = os.listdir(midi_folder)

        # Use multiprocessing to process MIDI files in parallel
        results = MULTIPROCESS_loop(self.process_midi_file, [(filename, midi_folder) for filename in filenames])

        # Remove None values (in case of skipped files)
        dataset = [res for res in results if res is not None]

        return dataset

    def _load_dataset_with_emotions(self):
        """
        Loads melodies from dataset and adds emotion labels based on filenames.

        Returns:
            list: A list of melodies with emotion labels.
        """
        import os
        dataset = []
        midi_folder = self.dataset_path

        for filename in os.listdir(midi_folder):
            if filename.endswith(".mid") or filename.endswith(".midi"): # example name "Q1_0vLPYiPN7qY_0.mid"
                if len(filename) >= 4:  # like "Q3_asdfasfas"
                    emotion = "EQ" + filename[1]  # filename.split("_")[0]  # Set prefix EQ as EQ1, EQ2, EQ3, EQ4 to be UNIQUE emotions words and different from music tokens words
                else:
                    continue
                melody = midi_to_tokens(os.path.join(midi_folder, filename))
                dataset.append((emotion, melody))
                # dataset.append(f"{emotion}, {melody}")

        return dataset

    def _parse_melody(self, melody_str):
        """
        Parses a single melody string into a list of notes.

        Parameters:
            melody_str (str): A string representation of a melody.

        Returns:
            list: A list of notes extracted from the melody string.
        """
        return melody_str.split(", ")

    def _tokenize_and_encode_melody(self, emotion, melody):
        """
        Tokenizes and encodes a single melody.

        Parameters:
            melody (list): A melody sequence as a list of notes.

        Returns:
            list: Tokenized and encoded melody.
        """
        # unique_tokens = set(tok_seq.tokens)  # Get unique token strings
        tokens = melody

        self.tokenizer.fit_on_texts(list([emotion]))   # emotion)) # emotions))  # Train on both melodies and emotions
        self.tokenizer.fit_on_texts(list(tokens))  # Train on both melodies and emotions
        # self.tokenizer.fit_on_texts([str(token) for token in melody])  # Ensure all tokens are strings
        return self.tokenizer.texts_to_sequences([tokens])[0]  # Convert melody to tokens
        # return token_ids  # Return tokenized IDs

        # TODO more modifiecation needed to skip fit_on_texts above so to be safe, still use them above
        # index_just_constructed = not self.tokenizer.word_index
        # if index_just_constructed:
        #     self.tokenizer.word_index = {}
        # word_indices = self.tokenizer.word_index
        # new_word_index = len(word_indices) + 1
        # for word in enumerate(unique_tokens):
        #     if index_just_constructed or word not in word_indices:
        #         word_indices[word] = new_word_index
        #         self.tokenizer.index_word[new_word_index] = word
        #         new_word_index += 1
        #
        # if not self.tokenizer.word_index:
        #     # self.tokenizer.word_index = {word: i + 1 for i, word in enumerate(unique_tokens)}
        #     # self.tokenizer.index_word = {i + 1: word for i, word in enumerate(unique_tokens)}
        #
        #     # Initialize word counts and docs
        #     self.tokenizer.word_counts = {word: tok_seq.tokens.count(word) for word in unique_tokens}
        #
        #     # Count how many different melodies contain each token
        #     self.tokenizer.word_docs = {word: 0 for word in unique_tokens}
        #     self.tokenizer.index_docs = {i + 1: 0 for i, word in enumerate(unique_tokens)}
        #
        # # Update document frequency counts
        # for word in unique_tokens:
        #     try:
        #         self.tokenizer.word_docs[word] += 1  # Increment doc count for token
        #     except Exception as e:
        #         pass
        #     token_id = self.tokenizer.word_index[word]  # Get corresponding ID
        #     self.tokenizer.index_docs[token_id] += 1  # Increment index_docs
        #
        # return token_ids  # Return tokenized IDs


    def _tokenize_and_encode_melodies(self, melodies_with_emotions):
        """
        Tokenizes melodies and includes emotion labels in the tokenizer.

        Parameters:
            melodies_with_emotions (list): List of (emotion, melody) pairs.

        Returns:
            tokenized_melodies: A list of tokenized melodies.
        """
        emotions, melodies = zip(*melodies_with_emotions)  # Separate emotions and melodies
        self.tokenizer.fit_on_texts(melodies + list(emotions))  # Train on both melodies and emotions
        tokenized_melodies = [(emotion, self.tokenizer.texts_to_sequences([melody])[0]) for emotion, melody in
                              melodies_with_emotions]
        return tokenized_melodies

    def _set_max_melody_length(self, melodies):
        """
        Sets the maximum melody length based on the dataset.

        Parameters:
            melodies (list): A list of tokenized melodies.
        """
        self.max_melody_length = max([len(melody) for melody in melodies])

    def _set_number_of_tokens(self):
        """
        Sets the number of tokens based on the tokenizer.
        """
        self.number_of_tokens = len(self.tokenizer.word_index)

    def _create_sequence_pairs_generator(self, melodies_with_emotions, stride=1):
    # def _create_sequence_pairs(self, melodies_with_emotions):
        """
        Creates input-target pairs from tokenized melodies.

        Parameters:
            melodies_with_emotions (list): A list of (emotion, tokenized_melody) tuples.
sx
        Returns:
            tuple: Three numpy arrays representing input sequences, emotion sequences, and target sequences.
        """
        # input_sequences, target_sequences, emotion_sequences = [], [], []

        k = 0
        for emotion, melody in melodies_with_emotions:
            k += 1
            # tokenized_melody = self.tokenizer.texts_to_sequences([melody])[0]
            emotion_seq = self.tokenizer.texts_to_sequences([emotion]) # e.g. 'EQ2' to its id
            if emotion_seq and emotion_seq[0]:  # Ensure the token exists
                emotion_seq = emotion_seq[0][0]
            else:
                raise ValueError(f"Emotion '{emotion}' was not found in tokenizer vocabulary.")

            # available_mem_gb = available_memory_gb()
            # if available_mem_gb < 20:
            #     input_sequences_size = asizeof(target_seq)
            #     pass
            # elif available_mem_gb < 10:
            #     pass
            # elif available_mem_gb < 5:
            #     pass

            for i in range(1, len(melody), stride):
                input_seq = melody[:i]
                target_seq = melody[1:i + 1]

                padded_input_seq = self._pad_sequence(input_seq)
                padded_target_seq = self._pad_sequence(target_seq)

                # input_sequences.append(padded_input_seq)
                # target_sequences.append(padded_target_seq)
                # emotion_sequences.append(emotion_seq)
                yield padded_input_seq, emotion_seq, padded_target_seq

        # return np.array(input_sequences), np.array(emotion_sequences), np.array(target_sequences)

    def _pad_sequence(self, sequence):
        """
        Pads a sequence to the maximum sequence length.

        Parameters:
            sequence (list): The sequence to be padded.

        Returns:
            list: The padded sequence.
        """
        return sequence + [0] * (self.max_melody_length - len(sequence))

    def _convert_to_tf_dataset(self, input_sequences, emotion_sequences, target_sequences):
        """
        Converts input, emotion, and target sequences to a TensorFlow Dataset.
        Parameters:
            input_sequences (list): Input sequences for the model.
            emotion_sequences (list): Emotion sequences for the model.
            target_sequences (list): Target sequences for the model.

        Returns:
            batched_dataset (tf.data.Dataset): A batched and shuffled TensorFlow Dataset.
        """
        dataset = tf.data.Dataset.from_tensor_slices(
            (input_sequences, emotion_sequences, target_sequences)
        )
        shuffled_dataset = dataset.shuffle(buffer_size=1000)
        batched_dataset = shuffled_dataset.batch(self.batch_size)
        return batched_dataset

if __name__ == "__main__":
    # Usage example
    preprocessor = MelodyPreprocessor("dataset.json", batch_size=32)
    training_dataset = preprocessor.create_training_dataset()
