import pickle
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import LSTMCell, LSTMStateTuple
import helpers

tf.reset_default_graph()
sess = tf.InteractiveSession()
eng = open("eng_seq.pkl", "rb")
cz = open("Cze_seq.pkl", "rb")

eng_seq = pickle.load(eng)
Cze_Seq = pickle.load(cz)

in_voc_size = 8406
out_voc_size = 13839

in_embed_size = 150
out_embed_size = 150

en_hid_unit = 100
deco_hid_unit = 200

EOS = 1
PAD = 0

en_input = tf.placeholder(shape=(None, None), dtype=tf.int32, name="Encoder_input")
en_in_len = tf.placeholder(shape=(None,), dtype=tf.int32, name="Encoder_input_length")

out_tar = tf.placeholder(shape=(None, None), dtype=tf.int32, name="OutPut_target")
out_tar_len = tf.placeholder(shape=(None,), dtype=tf.int32, name="OutPut_target_length")

en_embeddings = tf.Variable(tf.random_uniform([in_voc_size, in_embed_size], -1.0, 1.0), dtype=tf.float32)
en_input_embedded = tf.nn.embedding_lookup(en_embeddings, en_input)

out_embeddings = tf.Variable(tf.random_uniform([out_voc_size, out_embed_size], -1.0, 1.0), dtype=tf.float32)
out_in_embedded = tf.nn.embedding_lookup(out_embeddings, out_tar)

encoder_cell = LSTMCell(en_hid_unit)
((en_fw_output, en_bw_output), (en_fw_final_state, en_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(
    cell_fw=encoder_cell,
    cell_bw=encoder_cell,
    inputs=en_input_embedded,
    sequence_length=en_in_len,
    dtype=tf.float32,
    time_major=True
)

encoder_outputs = tf.concat((en_fw_output, en_bw_output), 2)
encoder_final_state_c = tf.concat(
    (en_fw_final_state.c, en_bw_final_state.c), 1)
encoder_final_state_h = tf.concat(
    (en_fw_final_state.h, en_bw_final_state.h), 1)

encoder_final_state = LSTMStateTuple(
    c=encoder_final_state_c,
    h=encoder_final_state_h
)

decoder_cell = LSTMCell(deco_hid_unit)
encoder_max_time, batch_size = tf.unstack(tf.shape(en_input))
decoder_lengths = en_in_len + 10

W = tf.Variable(tf.random_uniform([deco_hid_unit, out_voc_size], -1, 1), dtype=tf.float32)
b = tf.Variable(tf.zeros([out_voc_size]), dtype=tf.float32)

assert EOS == 1 and PAD == 0

eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='EOS')
pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='PAD')

# retrieves rows of the params tensor. The behavior is similar to using indexing with arrays in numpy
eos_step_embedded = tf.nn.embedding_lookup(out_embeddings, eos_time_slice)
pad_step_embedded = tf.nn.embedding_lookup(out_embeddings, pad_time_slice)


def loop_fn_initial():
    initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
    # end of sentence
    initial_input = eos_step_embedded
    # last time steps cell state
    initial_cell_state = encoder_final_state
    # none
    initial_cell_output = None
    # none
    initial_loop_state = None  # we don't need to pass any additional information
    return (initial_elements_finished,
            initial_input,
            initial_cell_state,
            initial_cell_output,
            initial_loop_state)


def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
    def get_next_input():
        output_logits = tf.add(tf.matmul(previous_output, W), b)
        prediction = tf.argmax(output_logits, axis=1)
        next_input = tf.nn.embedding_lookup(out_embeddings, prediction)
        return next_input

    elements_finished = (time >= decoder_lengths)

    finished = tf.reduce_all(elements_finished)  # -> boolean scalar
    input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)

    state = previous_state
    output = previous_output
    previous_loop_state = None

    return (elements_finished,
            input,
            state,
            output,
            previous_loop_state)


def loop_fn(time, previous_output, previous_state, previous_loop_state):
    if previous_state is None:  # time == 0
        assert previous_output is None and previous_state is None
        return loop_fn_initial()
    else:
        return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)


decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
decoder_outputs = decoder_outputs_ta.stack()
decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, out_voc_size))
decoder_prediction = tf.argmax(decoder_logits, 2)

stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(out_tar, depth=out_voc_size, dtype=tf.float32),
    logits=decoder_logits,
)

loss = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer().minimize(loss)

loss_track = []

"""
def input_len(u, l):
    length = -1
    for l in eng_seq[l:u]:
        length = max(len(l), length)
    return length


def padded_target(lo, u):
    return [np.asarray(k) for k in Cze_Seq[lo:u]]


def encoder_input(lo, up):
    return [ for k in eng_seq[lo:up]]

"""

sess.run(tf.global_variables_initializer())


def next_input(l, u):
    en_inputs, en_input_len = helpers.batch(eng_seq[l:u], 84)
    # print(en_input_len)
    decoder_target, _ = helpers.batch(
        [sequence + [EOS] for sequence in Cze_Seq[l:u]], 94)
    return {
        en_input: en_inputs,
        en_in_len: en_input_len,
        out_tar: decoder_target,
    }

# every time show the predicted line for each training step

try:

    for e in range(50):
        for i in range(0, 2000, 50):
            fd = next_input(i, i + 50)
            _, l = sess.run([train_op, loss], fd)
            loss_track.append(l)
            if i % 50 == 0:
                print(l)
        test = next_input(2001, 2020)
        predict_ = sess.run(decoder_prediction, test)
        for i, (inp, pred) in enumerate(zip(test[en_input].T, predict_.T)):
            print('  sample {}:'.format(i + 1))
            print('  input     > {}'.format(inp))
            print('  predicted > {}'.format(pred))



except KeyboardInterrupt:
    print("training interrupted")
