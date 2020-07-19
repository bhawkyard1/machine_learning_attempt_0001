import os
import tensorflow as tf

BATCH_SIZE = 64
BUFFER_SIZE = 10000
CHECKPOINTS_DIR = os.path.join(os.getcwd(), "checkpoints")
CHECKPOINTS_PREFIX = os.path.join(CHECKPOINTS_DIR, "checkpoint_{epoch}")
DATA_PATH = "all.txt"
EPOCHS = 15
SEQ_LEN = 100

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer="glorot_uniform"),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model