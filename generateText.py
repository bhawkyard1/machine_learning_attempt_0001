import numpy as np
import os
import random

import tensorflow as tf

from . import common

def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 1000

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 0.2

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the character returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted character as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (start_string + "".join(text_generated))

def get_start_string(text):
  sentence = random.choice(text.split("."))
  words = sentence.split(" ")
  return " ".join(words[:min(len(words) - 1, 3)])

def run():
  print(f"Using tensorflow {tf.__version__}")

  text = open(common.DATA_PATH, "rb").read().decode(encoding="utf-8")
  vocab = sorted(set(text))

  print(f"Identified {len(vocab)} unique characters.")

  char2idx = {u: i for i, u in enumerate(vocab)}
  idx2char = np.array(vocab)

  # The embedding dimension
  embedding_dim = 256
  # Number of RNN units
  rnn_units = 1024
  # Length of the vocabulary in chars
  vocab_size = len(vocab)

  model = common.build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
  model.load_weights(tf.train.latest_checkpoint(common.CHECKPOINTS_DIR))
  model.build(tf.TensorShape([1, None]))

  start_string = get_start_string(text)
  print("START:",start_string,"\n")
  print(
    generate_text(
      model, 
      start_string=start_string
    )
  )


  