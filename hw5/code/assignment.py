from custom_snr import custom_send_and_recv
from tensorflow.keras.layers import Dense, Layer
import tensorflow as tf
import dgl
import preprocess
import numpy as np
import os
os.environ['DGLBACKEND'] = "tensorflow"


class Model(tf.keras.Model):
    """Model class representing your MPNN."""

    def __init__(self):
        """
        Instantiate a lifting layer, an optimizer, some number of MPLayers 
        (we recommend 3), and a readout layer.
        """
        super(Model, self).__init__()

        # TODO: Initialize hyperparameters
        self.raw_features = None
        self.num_classes = None
        self.learning_rate = 1e-4
        self.hidden_size = None
        self.batch_size = None

        # TODO: Initialize trainable parameters

    def call(self, g, is_testing=False):
        """
        Computes the forward pass of your network.
        1) Lift the features of the batched graph passed in. Don't apply an activation function.
        2) After the node features for the graph have been lifted, run them 
           through the MPLayers.
        3) Feed the output of the final MPLayer through the readout function
           to get the logits.
        :param g: The DGL graph you wish to run inference on.
        :return: logits tensor of size (batch_size, 2)
        """
        # TODO: Fill this in!
        return None

    def readout(self, g, node_feats):
        """
        Reduces the dimensionality of the graph to
        num_classes, and sums the node features in order to return logits.
        :param g: The batched DGL graph
        :param node_feats: The features at each node in the graph. Tensor of shape
                                   (num_atoms_in_batched_graph,
                                    size_of_node_vectors_from_prev_message_passing)
        :return: logits tensor of size (batch_size, 2)
        """
        # TODO: Set the node features to be the output of your readout layer on
        # node_feats, then use dgl.sum_nodes to return logits.
        return None

    def accuracy_function(self, logits, labels):
        """
        Computes the accuracy across a batch of logits and labels.
        :param logits: a 2-D np array of size (batch_size, 2)
        :param labels: a 1-D np array of size (batch_size)
            (1 for if the molecule is active against cancer, else 0).
        :return: mean accuracy over batch.
        """
        # TODO: Fill this in!
        return None


class MPLayer(Layer):
    """
    A TensorFlow Layer designed to represent a single round of message passing.
    This should be instantiated in your Model class several times.
    """

    def __init__(self, in_feats, out_feats):
        """
        Make a message computation layer which will compute the messages sent
        by each node to its neighbors and an output layer which will be
        applied to all nodes as a final transformation after message passing
        from size in_feats to out_feats.
        :param in_feats: The size of vectors at each node of your graph when you begin
        message passing for this round.
        :param out_feats: The size of vectors that you'd like to have at each of your
        nodes when you end message passing for this round.
        """
        super(MPLayer, self).__init__()
        # TODO: Fill this in!

    def call(self, g, is_testing=False):
        """
        Computes the forward pass of your MPNN layer
        1) Call the either DGL's send and receive function or your own,
            depending on the is_testing flag
        2) Calculate the output features by passing the graph's node features
            through the output layer
        3) Set the graph's node features to be the output features

        To assign/retrieve the node data, you can use `g.ndata["node_feats"]`

        The send and receive functions to be used are the following:
            g.send_and_recv                 # DGL send_and_recv
            custom_send_and_recv            # custom send_and_recv
        We assign the "messager" function and the "reducer" function to be
            passed in to the send and receive function for you

        :param g: The batched DGL graph you wish to run inference on.
        :param is_testing: True if using custom send_and_recv, false if using DGL
        :return: None
        """
        # The message function for testing
        messager = lambda x: self.message(x, True) if is_testing else self.message(x)
        # The reduce function for testing
        reducer = lambda x: self.reduce(x, True) if is_testing else self.reduce(x)
        # TODO: Fill this in!

    def message(self, edges, is_testing=False):
        """
        This function, when called on a group of edges, should compute a message 
        for each edge to be sent from the edge's src node to its dst node. 

        The message is computed by passing the src node's features into a linear layer
        with ReLU activation. This message will be sent to all dst nodes at once
        by sending a dictionary with key 'msg' to a shared mailbox.

        The source nodes' features can all be accessed like:
            edges.src['node_feats']    # DGL send_and_recv
            edges                      # custom send_and_recv

        :param edges: All the DGL edges in the batched DGL graph.
        :param is_testing: True if using custom send_and_recv, false if using DGL
        :return: A dictionary from some 'msg' to all the messages
        computed for each edge.
        """
        pass

    def reduce(self, nodes, is_testing=False):
        """
        This function, when called on a group of nodes, should aggregate (i.e. sum)
        the messages in the mailboxes of each node. Each node will only have messages
        from its neighbors.

        We will then save these new features in each node under the attribute 'node_feats'.
        The messages of all nodes can be accessed like:
            nodes.mailbox['msg']    # DGL send_and_recv
            nodes['msg']            # custom send_and_recv

        :param nodes: All the DGL nodes in the batched DGL Graph.
        :param is_testing: True if using custom send_and_recv, false if using DGL
        :return: A dictionary from 'node_feats' to the summed messages for each node.
        """
        pass


def build_graph(molecule):
    """
    Constructs a DGL graph out of a molecule from the train/test data.
    :param molecule: a Molecule object (see molecule.py for more info)
    :return: A DGL Graph with the same number of nodes as atoms in the molecule, edges connecting them,
             and node features applied.
    """
    # TODO: Initialize a DGL Graph
    # TODO: Call the graph's add_nodes method with the number of nodes in the molecule.
    # TODO: Turn molecule's nodes into a tensor, and set it to be the data of this graph.
    #       To set the data "data" for graph "graph", call `graph.ndata["node_feats"] = data`
    # TODO: Construct a tuple of src and dst nodes from the list of edges in molecules.
    #       e.g if the edges of the molecule looked like [(1,2), (3,4), (5,6)] return
    #       (1,3,5) and (2,4,6).
    # TODO: Call the graph's add_edges method to add edges from src to dst and dst to src.
    #       Edges are directed in DGL, but undirected in a molecule, so you have
    #       to add them both ways.

    return None


def train(model, train_data):
    """
    Trains your model given the training data.
    For each batch of molecules in train data...
        1) Make dgl graphs for each of the molecules in your batch; collect them in a list.
        2) Call dgl.batch to turn your list of graphs into a batched graph.
        3) Turn the labels of each of the molecules in your batch into a 1-D tensor of size
            batch_size
        4) Pass this graph to the Model's forward pass. Run the resulting logits
                        and the labels of the molecule batch through SparseCategoricalCrossentropy.
        5) Compute the gradients of the Model's trainable variables.
        6) Take a step with the optimizer.
    :param model: Model class representing your MPNN.
    :param train_data: A 1-D list of molecule objects, representing all the molecules
    in the training set from get_data
    :return: nothing.
    """
    # This is the loss function, usage: loss(labels, logits)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # TODO: Implement train with the docstring instructions


def test(model, test_data):
    """
    Testing function for our model.
    Batch the molecules in test_data, feed them into your model as described in train.
    After you have the logits: turn them back into numpy arrays, compare the accuracy to the labels,
    and keep a running sum.
    :param model: Model class representing your MPNN.
    :param test_data: A 1-D list of molecule objects, representing all the molecules in your
    testing set from get_data.
    :return: total accuracy over the test set (between 0 and 1)
    """
    # TODO: Fill this in!
    return None


def main():
    # TODO: Return the training and testing data from get_data
    # TODO: Instantiate model
    # TODO: Train and test for up to 15 epochs.
    pass


if __name__ == '__main__':
    main()
