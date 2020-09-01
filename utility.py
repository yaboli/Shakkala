import os
from keras.utils.vis_utils import plot_model
from keras import backend as K
import tensorflow as tf
import tensorflow_datasets as tfds

from Shakkala import Shakkala


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


def invert_map(original_map):
    return {v: k for k, v in original_map.items()}


def process_data():
    pass


class Utility(object):
    def __init__(self, version=3):
        self.shakkala = Shakkala('./', version=version)
        self.max_length = self.shakkala.max_sentence
        self.model_location = self.shakkala.model_location
        self.dictionary = self.shakkala.dictionary
        self.model, self.graph = self.shakkala.get_model()
        self.output_vocab_to_int = invert_map(self.dictionary['output_int_to_vocab'])

    def print_model_summary(self):
        self.model.summary()

    def plot_model(self, output_path='./images/model_plot.png'):
        plot_model(self.model, to_file=output_path, show_shapes=True, show_layer_names=True)

    def freeze_model(self, output_dir='./model', file_name='model_frozen', as_text=False):
        frozen_graph = freeze_session(K.get_session(),
                                      output_names=[out.op.name for out in self.model.outputs])
        if as_text:
            tf.train.write_graph(frozen_graph, output_dir, file_name + '.pbtxt', as_text=True)
        else:
            tf.train.write_graph(frozen_graph, output_dir, file_name + '.pb', as_text=False)

    def do_transfer_learning(self):
        pass

    def fine_tune(self, output_path=''):
        pass


if __name__ == '__main__':
    # utility = Utility()
    # utility.plot_model()

    pass
