import numpy as np
import os
import random

import tensorflow as tf

from . import common

def run():
  print(f"Using tensorflow {tf.__version__}")
  print(f"Reading data from {common.DATA_PATH}...")

  text = open(common.DATA_PATH, "rb").read().decode(encoding="utf-8")
  vocab = sorted(set(text))

  print(f"Identified {len(vocab)} unique characters.")

  char2idx = {u: i for i, u in enumerate(vocab)}
  idx2char = np.array(vocab)

  text_as_int = np.array([char2idx[c] for c in text])

  # Create training examples / targets
  char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

  sequences = char_dataset.batch(common.SEQ_LEN + 1, drop_remainder=True)

  def split_input_target(chunk):
      input_text = chunk[:-1]
      target_text = chunk[1:]
      return input_text, target_text

  dataset = sequences.map(split_input_target)
  dataset = dataset.shuffle(common.BUFFER_SIZE).batch(common.BATCH_SIZE, drop_remainder=True)

  # Length of the vocabulary in chars
  vocab_size = len(vocab)

  # The embedding dimension
  embedding_dim = 256

  # Number of RNN units
  rnn_units = 1024

  model = common.build_model(
    vocab_size = len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=common.BATCH_SIZE)

  def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

  # example_batch_loss  = loss(target_example_batch, example_batch_predictions)
  model.compile(optimizer="adam", loss=loss)

  checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
      filepath=common.CHECKPOINT_PREFIX,
      save_weights_only=True
  )

  history = model.fit(dataset, epochs=common.EPOCHS, callbacks=[checkpoint_callback])