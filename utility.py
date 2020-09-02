import os
from keras import backend as K
from keras.utils.vis_utils import plot_model
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model.signature_def_utils import predict_signature_def
from tensorflow.python.saved_model import tag_constants

from Shakkala import Shakkala
from data_utils import Dataset


class Utility(object):
    def __init__(self, version=3):
        self.shakkala = Shakkala('./', version=version)
        self.max_length = self.shakkala.max_sentence
        self.model_location = self.shakkala.model_location
        self.dictionary = self.shakkala.dictionary
        self.model, self.graph = self.shakkala.get_model()
        self.ds = Dataset(version=version)

    def print_model_summary(self):
        self.model.summary()

    def plot_model(self, output_path='./images/model_plot.png'):
        plot_model(self.model, to_file=output_path, show_shapes=True, show_layer_names=True)

    def export_h5_to_pb(self, export_path):
        # Set the learning phase to Test since the model is already trained.
        K.set_learning_phase(0)
        # Build the Protocol Buffer SavedModel at 'export_path'
        builder = saved_model_builder.SavedModelBuilder(export_path)
        # Create prediction signature to be used by TensorFlow Serving Predict API
        signature = predict_signature_def(inputs={"tokens": self.model.input},
                                          outputs={"tags": self.model.output})
        with K.get_session() as sess:
            # Save the meta graph and the variables
            builder.add_meta_graph_and_variables(sess=sess, tags=[tag_constants.SERVING],
                                                 signature_def_map={"predict": signature})
        builder.save()

    # Load model and continue training with new data
    def train(self,
              lr=1e-5,
              epochs=5,
              batch_size=32,
              filename='test_small.txt',
              new_model='new_model.h5'):
        x_train, x_val, y_train, y_val = self.ds.build_dataset(filename)
        K.set_value(self.model.optimizer.lr, lr)
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val))
        new_model_path = os.path.join('./model', new_model)
        self.model.save(new_model_path)

    def do_transfer_learning(self):
        pass

    def fine_tune(self):
        pass


if __name__ == '__main__':
    utility = Utility()
    # utility.plot_model()
    # utility.print_model_summary()
    utility.train()
