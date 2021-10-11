import os
os.environ['DGLBACKEND'] = "tensorflow"
import dgl
import tensorflow as tf
from helper import perform_message_passing, simple_message, simple_reduce, gen_node_feats_tests, get_test_edges, gen_random_edges, run_visualize_message_passing
from collections import defaultdict as ddict
# ---------- DO NOT EDIT ABOVE THIS LINE ----------


def custom_send_and_recv(g, message_func, reduce_func):
    """
    A function that implements message passing on a DGL graph. When this function is called, it should
    loop through all nodes, generate a message based on the node's features, and store that message in order to
    communicate it to each node it is connected to.

    It should then generate a new feature for each node based on the messages it received,
    and finally update g.ndata with the updated node features.

    Note that you are still allowed to use most of DGL in this function (just not send_and_recv!), so be
    sure to check out the documentation.

    Essentially, for each node, compute its message using message_func and send its message to its neighbors;
    then, for each node, reduce its incoming messages with reduce_func. Recall that you need to create a mailbox
    which will hold mappings of (node -> message_title -> messages_of_that_title). Here, message_title must always
    be "msg" as that is what is used in simple_reduce and simple_message.

    :param g: The DGL graph on which message passing is being performed.
    :param message_func: A function that, given the dst node of an edge, will generate a message to be sent to
    each edge's src node.
    :param reduce_func: A function that, given a node's messages, will compute the node's updated features
    based on its messages.
    :return: None
    """
    # TODO: Create a mailbox
    # TODO: Extract the current node features
    # TODO: For each node, send its message
    # TODO: Set the new node features
    
    d = lambda: ddict(d)
    mailbox = d()
    features = g.ndata['node_feats']
    
    for node in g.nodes():
        key = node.numpy()
        message = message_func(features[key])['msg']
        
        neighbors = g.successors(node.numpy()).numpy()
        for n in neighbors:
            if n in mailbox:
                mailbox[n]['msg'].append(message)
            else:
                mailbox[n]['msg'] = [message]
    
    new_features = []
    for node in g.nodes():
        key = node.numpy()
        mailbox[key]['msg'] = tf.convert_to_tensor(mailbox[key]['msg'], dtype=tf.float32)
        mailbox[key]['msg'] = tf.expand_dims(mailbox[key]['msg'], 0) #reshape to (1, num_messages, message_size)
        new_features.append(reduce_func(mailbox[key])['node_feats'])
    new_features = tf.convert_to_tensor(new_features, dtype=tf.float32)
    g.ndata["node_feats"] = tf.squeeze(new_features)
    

def run_tests():
    """
    Tests the custom_send_and_recv against DGL's send_and_recv given a graph and some initial node features
    DO NOT EDIT THIS FUNCTION
    """

    def test_node_feats(g, node_feats, test_num):
        tf.debugging.assert_equal(
            perform_message_passing(g, node_feats, custom_send_and_recv, True),
            perform_message_passing(g, node_feats, g.send_and_recv),
            "SNR implementation failed for test " + str(test_num))

    # test on a batch of graphs (size= batch_size)
    batch_size = 3
    batched_graphs = []
    cur_batch = []

    test_dict = get_test_edges()

    for key in test_dict:
        u, v = test_dict[key]
        g = dgl.DGLGraph((u, v))
        g = dgl.to_bidirected(g)

        if len(cur_batch) == batch_size:
            batched_graphs.append(cur_batch)
            cur_batch = []
        cur_batch.append(g)

        node_feats_tests = gen_node_feats_tests(g)
        for i in range(len(node_feats_tests)):
            test_node_feats(g, node_feats_tests[i], i+1)
            print(f"{key} Test #{i+1} passed")

    for k in range(len(batched_graphs)):
        g = dgl.batch(batched_graphs[k])
        node_feats_tests = gen_node_feats_tests(g)
        for i in range(len(node_feats_tests)):
            test_node_feats(g, node_feats_tests[i], i+1)
            print("Batched Graph Test " + str(k+1) + "." + str(i+1) + " passed")

    print("All tests passed! 🎉🎉🎉")

def main():
    # TODO: Call run_visualize_message_passing

    run_tests()
    # run_visualize_message_passing()

if __name__ == '__main__':
    main()
