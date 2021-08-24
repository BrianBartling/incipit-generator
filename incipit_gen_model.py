import tensorflow as tf
import argparse
import os
import random
from time import time

SEMANTIC_CLEFS = ['clef-C1', 'clef-C2', 'clef-C3', 'clef-C4', 'clef-C5', 'clef-F3',
                   'clef-F4', 'clef-F5', 'clef-G1', 'clef-G2']

def generate_mask(word2int):
    skip_ids = word2int(['[UNK]'])[:, None]
    sparse_mask = tf.SparseTensor(
        values=[-float('inf')]*len(skip_ids),
        indices=skip_ids,
        dense_shape=[len(word2int.get_vocabulary())])
    return tf.sparse.to_dense(sparse_mask)


class InitModel(tf.keras.Model):
    def __init__(self, corpus, set, vocabulary, semantic=True, checkpoint_dir='./trained_model',
                epochs=30, val_split=0.1, batch_size=64, buffer_size=10000):
        super().__init__()

        # Parametrization
        self.val_split = val_split
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.semantic = semantic
        self.corpus_dirpath = corpus
        self.vocabulary = vocabulary

        self.checkpoint_dir = checkpoint_dir

        self.epochs = epochs

        self.word2int = tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=self.vocabulary, num_oov_indices=0, mask_token=None
        )
        self.int2word = tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=self.word2int.get_vocabulary(), mask_token=None, invert=True
        )

        # Corpus
        corpus_file = open(set,'r')
        self.corpus_list = corpus_file.read().splitlines()
        corpus_file.close()
        random.shuffle(self.corpus_list)

        self.dataset = None
        self.model = None

        self.init_model()

    def init_model(self):
        print("Initializing model...")
        self.dataset = tf.data.Dataset.from_tensor_slices(self.corpus_list)
        self.dataset = (
            self.dataset.map(
                self.populate_data, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
        )
        self.dataset = self.dataset.map(self.split_input_target)

        self.dataset = (
            self.dataset
            .shuffle(self.buffer_size)
            .padded_batch(self.batch_size)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        # The embedding dimension
        embedding_dim = 256

        # Number of RNN units
        rnn_units = 1024

        dropout_rate = 0.5

        self.model = LSTMModel(
            vocab_size=len(self.word2int.get_vocabulary()),
            embedding_dim=embedding_dim,
            rnn_units=rnn_units,
            dropout_rate=dropout_rate)

    def train_model(self, loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam'):
        self.model.compile(optimizer=optimizer, loss=loss)

        # Name of the checkpoint files
        checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt_{epoch}")

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True)

        history = self.model.fit(self.dataset, epochs=self.epochs, callbacks=[checkpoint_callback])

        return history

    def populate_data(self, sample_filepath):
        sample_fullpath = self.corpus_dirpath + os.sep + sample_filepath + os.sep + sample_filepath

        label_file = sample_fullpath

        if self.semantic:
            label_file = label_file + '.semantic'
        else:
            label_file = label_file + '.agnostic'

        sample_gt_file = tf.io.read_file(label_file)
        stripped = tf.strings.strip(sample_gt_file)
        sample_gt_plain = tf.strings.split(stripped)

        target = tf.cast(self.word2int(sample_gt_plain), tf.int32)

        return target

    @staticmethod
    def split_input_target(sequence):
        input_text = sequence[:-1]
        target_text = sequence[1:]
        return input_text, target_text


class LSTMModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units, dropout_rate):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        forward_layer = tf.keras.layers.LSTM(rnn_units, return_sequences=True, return_state=True,dropout=dropout_rate)
        backward_layer = tf.keras.layers.LSTM(rnn_units, activation='relu', return_sequences=True, return_state=True,
                                                    go_backwards=True, dropout=dropout_rate)
        self.rnn_outputs = tf.keras.layers.Bidirectional(forward_layer, backward_layer = backward_layer)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.rnn_outputs.get_initial_state(x)
        x, states = self.rnn_outputs(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x

class OneStep(tf.keras.Model):
    def __init__(self, model, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.int2word = model.int2word
        self.word2int = model.word2int

        self.prediction_mask = generate_mask(self.word2int)

    @tf.function
    def generate_one_step(self, inputs, states=None):
        input_words = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ints = self.word2int(input_words).to_tensor()

        predicted_logits, states = self.model(inputs=input_ints, states=states, return_state=True)
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits/self.temperature
        predicted_logits = predicted_logits + self.prediction_mask

        predicted_ints = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ints = tf.squeeze(predicted_ints, axis=-1)

        predicted_words = self.int2word(predicted_ints)

        return predicted_words, states


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument('-corpus', dest='corpus', type=str, required=True, help='Path to the corpus.')
    parser.add_argument('-set',  dest='set', type=str, required=True, help='Path to the set file.')
    parser.add_argument('-save_model', dest='save_model', type=str, required=False, help='Path to save the model.')
    parser.add_argument('-vocabulary', dest='voc', type=str, required=True, help='Path to the vocabulary file.')
    parser.add_argument('-semantic', dest='semantic', action="store_true", default=False)
    parser.add_argument('-use_model', dest='use_model', type=str, required=False, default=None, help='Load a model from an external file, continue training')
    args = parser.parse_args()

    oneStep = InitModel(args.corpus, args.set, args.voc, args.semantic)

    ## Test out model
    for input_example_batch, target_example_batch in oneStep.dataset.take(1):
        example_batch_predictions = oneStep.model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

    print(oneStep.model.summary())

    sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

    print(sampled_indices)

    print("Input:\n", oneStep.int2word(input_example_batch[0]-1).numpy())
    print()
    print("Next Char Predictions:\n", oneStep.int2word(sampled_indices-1).numpy())