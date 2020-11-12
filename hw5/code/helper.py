import tensorflow as tf
import numpy as np
import os

os.environ['DGLBACKEND'] = "tensorflow"
import dgl
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from random import sample

def get_test_edges():

    return {
        "EMPTY":([],[]),
        "ALL_TO_ONE":([0, 1, 2, 3, 4, 5, 6], [0, 0, 0, 0, 0, 0, 0]),
        "RANDOM0":gen_random_edges(),
        "RANDOM1":gen_random_edges(num_edges=1000),
        "FULLY_CONNECTED":np.array(np.meshgrid(np.arange(10), np.arange(10))).T.reshape(-1,2).T
    }

def gen_node_feats_tests(g):
    node_feats_tests = [
            # test with all 0s
            tf.fill((g.num_nodes(), g.num_nodes()), 0.0),
            # test with all 1s
            tf.fill((g.num_nodes(), g.num_nodes()), 1.0),
            # test with random
            tf.convert_to_tensor([tf.random.normal((g.num_nodes(),)) for x in range(g.num_nodes())])
        ]
    return node_feats_tests

def perform_message_passing(g, node_feats, send_and_recv_fn, custom = False):
    """
    Responsible for computing message passing on an arbitrary graph.
    1) You should reassign g's ndata field to be the node features passed to this function.
    2) Trigger message passing and aggregation on g using custom_send_and_recv.
        You'll want to conditionally call either your or DGL's send_and_recv implementation depending
        on the value of use_custom_snr. Remember that each function has different parameters.
    3) Pop the node features and return.
    You can assign/retrieve node data by accessing the graph's ndata field with some attribute
    that you'd like to save the features under in the graph (e.g g.ndata["h"] = node_feats)
    :param g: The DGL graph you wish to run inference on.
    :param node_feats: Beginning node features for your graph. Can be a tensor of any shape.
    :return: updated node features with the same shape as the node_feats param
    """
    g.ndata['node_feats'] = node_feats

    if custom:
        messager = lambda x: simple_message(x, True)
        reducer = lambda x: simple_reduce(x, True)
        send_and_recv_fn(g,
            message_func=messager,
            reduce_func=reducer)
    else:
        send_and_recv_fn(g.edges(),
            message_func=simple_message,
            reduce_func=simple_reduce)

    node_feats = g.ndata.pop('node_feats')
    return node_feats

def simple_message(edges, custom=False):
    """
    A function to be passed to custom_send_and_recv. This function should compute a message
    given either a group of edges (DGL) or the node features of a particular node (custom_send_and_recv).

    The source nodes' features can all be accessed like:
        edges.src['node_feats']    # DGL send_and_recv
        edges                      # custom send_and_recv

    Note that when using DGL's send_and_recv implementation, the nodes param will be an EdgeBatch with many edges.
    Your implementation, however, will pass this function the data of a single node.

    :param edges: An EdgeBatch of edges in the DGL graph, or the node feature of the source node of one edge
    for custom_send_and_recv.
    :param custom: Whether we are using custom send and receive or not
    :return: A map from "msg" to all the messages computed for the edges. These messages can then be
    retrieved at each destination node's mailbox (e.g mailbox[node]["msg"]) once custom_send_and_recv
    distributes them.
    """
    node_feats = edges if custom else edges.src['node_feats']
    return {"msg": node_feats * 2}

def simple_reduce(nodes, custom=False):
    """
    A function to be passed to custom_send_and_recv. This function should aggregate (i.e. sum)
    all the messages received for a group of nodes (DGL) or the messages received for a
    particular node (custom_send_and_recv) from message passing.

    Your custom_send_and_recv function will then save this sum in an updated node_feats tensor
    for the DGL graph (g.send_and_recv handles this in DGL's implementation).

    Note that when using DGL's send_and_recv implementation, the nodes param will be a NodeBatch with many nodes.
    Your implementation, however, will pass this function the messages for a single node.

    The messages of all nodes can be accessed like:
        nodes.mailbox['msg']    # DGL send_and_recv
        nodes['msg']            # custom send_and_recv

    :param nodes: A NodeBatch of nodes in the DGL graph, or the messages for one node in custom_send_and_recv.
    :param custom: Whether we are using custom send and receive or not
    :return: A map from "node_feats" to the summed messages for each node. The node data for this node will be
    updated to the summed messages by your implementation.
    """
    messages = nodes['msg'] if custom else nodes.mailbox['msg']
    return {"node_feats": tf.math.reduce_sum(messages, axis=1)}

def gen_random_edges(num_edges=10, max_edges_per_node=2):
    """
    Generates num_edges edges connecting nodes with ids [0,...,num_edges).
    :param num_edges: The number of edges to generate.
    :param max_edges_per_node: The maximum number of edges a node can have.
    :return: A tuple of 2 lists of the same length representing src and dst nodes of
    each edge (i.e. edge_i == (u[i], v[i])).
    """
    if (num_edges < max_edges_per_node):
        return ([],[])
    sample_edges = lambda k: sample(range(num_edges), k)

    srcs = [sample_edges(num_edges // max_edges_per_node) for i in range(max_edges_per_node)]
    u = [];
    for s in srcs:
        u.extend(s)

    v = sample_edges(num_edges)

    return (u, v)

def interpolate(x, max_x, min_x, max_y, min_y):
    """
    Performs the interpolation calculation to translate a number from one range
    to another.
    :param x: The number to be interpolated.
    :param max_x: The maximum of the range the number was originally within.
    :param min_x: The minimum of the range the number was originally within.
    :param max_y: The maximum of the range the number will be interpolated to.
    :param min_y: The minimum of the range the number will be interpolated to.
    :return: x translated to the new range.
    """
    if (max_x == min_x):
        return x * min_y
    return ((max_y - min_y) / (max_x - min_x)) * x + min_y

def calc_arr_rep_of_interp(x, max_x, min_x, max_y, min_y, num_vals=1):
    """
    Produces an array of num_vals values in the range [min_y, max_y] representing x
    interpolated to the range [min_y, max_y * num_vals].
    For example if you want an array of values from 0 to 1 (representing rgb values) given some
    value and the range that value is situated in:
        max_y = 1
        min_y = 0
        max_x, min_x = max(range_x), min(range_x)
    :param x: The number to be interpolated.
    :param max_x: The maximum value that x could be.
    :param min_x: The minimum value that x could be.
    :param max_y: The maximum value each entry of the output array could be.
    :param min_y: The minimum value each entry of the output array could be.
    """
    max_value = num_vals * max_y
    y = interpolate(x, max_x, min_x, max_value, min_y)

    vals = []
    if (y < max_y): # avoid unnecessary calculation
        vals = [y, *([min_y] * (num_vals - 1))]
    elif (y >= max_value):
        vals = [max_y] * num_vals
    else:
        for i in range(0, num_vals):
            if (y > max_y * i):
                if (int(y) > max_y * i):
                    vals.append(max_y)
                else:
                    vals.append(y - (max_y * i))
            else:
                vals.append(min_y)

    return vals

def prerender_graph_data(g, message_passing_func, rounds, init_node_feats = None):
    """
    Prerenders node data in graph for efficiency and smooth animation.
    :param g: The graph that will be animated.
    :param message_passing_func: The function that performs message passing.
    :param rounds: The number of rounds of message passing.
    :param init_node_feats: Optional initial node features passed to the first message passing round.
    :return: A dict containing the graph and the rendered node features, labels, sizes, and colors for each frame.
    """
    node_feats = init_node_feats if init_node_feats != None else tf.convert_to_tensor([tf.random.normal((g.num_nodes(),)) for x in range(g.num_nodes())])
    prerendered_node_feats = []
    prerendered_node_labels = []
    prerendered_node_sizes = []
    prerendered_node_colors = []
    for i in range(rounds):
        node_feats = message_passing_func(g, node_feats, g.send_and_recv)
        readable_node_feats = [int(np.sum(feats)) for feats in node_feats.numpy()]

        prerendered_node_feats.append(readable_node_feats)

        prerendered_node_labels.append(dict(zip(range(len(readable_node_feats)), readable_node_feats)))

        max_feat, min_feat = max(readable_node_feats), min(readable_node_feats)
        min_node_size = 300
        max_node_size = min(min_node_size * 5, min_node_size * (1 + len(str(round(max_feat)))))
        prerendered_node_sizes.append([interpolate(x, max_feat, min_feat, max_node_size, min_node_size) for x in readable_node_feats])

        node_feats_offset = abs(min_feat)
        # prerendered_node_colors.append([calc_arr_rep_of_interp(x + node_feats_offset, max_feat + node_feats_offset, min_feat + node_feats_offset, 1, 0, num_vals=3) for x in readable_node_feats])
        prerendered_node_colors.append([[0, 0, interpolate(x + node_feats_offset, max_feat + node_feats_offset, min_feat + node_feats_offset, 1, 0)] for x in readable_node_feats])

    return {
        "graph": g,
        "node_feats": prerendered_node_feats,
        "node_labels": prerendered_node_labels,
        "node_sizes": prerendered_node_sizes,
        "node_colors": prerendered_node_colors,
    }

def visualize_message_passing(g, message_passing_func, init_node_feats=None, rounds=10, interval=300, font_color="lightgray", bypass_node_limit=False):
    """
    Responsible for prerendering data and initiating the message passing animation. This is what you should
    be calling when you want to visualize the message passing process.
    :param g: The graph on which message passing will be performed.
    :param message_passing_func: The function which performs message passing on a graph.
    :param init_node_feats: Optional initial node features passed to the first message passing round.
    :param rounds: The number of rounds of message passing.
    :param interval: The delay between frames in the animation.
    :param bypass_node_limit: Disables check for whether there are too many nodes to reasonable render.
    :return: None
    """
    if (not bypass_node_limit and g.num_nodes() > 10000): # takes too long to render
        return None

    graph_data = prerender_graph_data(g, message_passing_func, rounds, init_node_feats=init_node_feats)

    fig, ax = plt.subplots()
    fig.set_size_inches(16, 8)

    def update(frame, fargs):
        plt.clf()
        draw_graph(
            fargs["graph"],
            fargs["node_feats"][frame],
            labels=fargs["node_labels"][frame],
            sizes=fargs["node_sizes"][frame],
            colors=fargs["node_colors"][frame],
            font_color=font_color,
        )
        fig.suptitle("Round " + str(frame + 1))

    anim = FuncAnimation(
        fig,
        update,
        fargs=[graph_data],
        frames=range(rounds),
        interval=interval
    )
    plt.show()

def draw_graph(g, node_feats, labels=None, sizes=None, colors=None, font_color="lightgrag", edgecolors="black"):
    """
    Handles drawing a graph with nodes with specific labels, sizes, and colors.
    :param g: The graph to be drawn.
    :param node_feats: The node features of the graph
    :param labels: Optional labels for nodes.
    :param sizes: Optional sizes for nodes.
    :param colors: Optional colors for nodes.
    :param font_colors: Optional font colors for nodes.
    :param edgecolors: The border color of nodes.
    """
    nx_G = g.to_networkx().to_undirected()
    pos = nx.kamada_kawai_layout(nx_G)
    nx.draw_networkx(
        nx_G,
        pos,
        labels=labels,
        node_color=colors,
        font_color=font_color,
        node_size=sizes,
        edgecolors=edgecolors,
    )

def run_visualize_message_passing():
    """
    Constructs a graph and some initial node features, then calls visualize_message_passing
    to begin an animation of the message passing process.
    """
    # TODO: Construct a graph and a tensor of node features
    # TODO: Visualize message passing on the graph by filling in the arguments of visualize_message_passing
    # FOR STENCIL:
    # g = None
    # visualize_message_passing(g, perform_message_passing, init_node_feats=None)
    test_g_num = 1

    test_g_data = [
        (list(range(10)), [*range(1, 10), 0]),
        gen_random_edges(num_edges=100),
        gen_random_edges(num_edges=1000),
    ]

    u, v = test_g_data[test_g_num]
    g = dgl.DGLGraph((u, v))

    test_node_feats = [
        [1, *[0] * (g.num_nodes() - 1)],
        tf.convert_to_tensor([tf.random.normal((g.num_nodes(),)) for x in range(g.num_nodes())]),
    ]
    init_node_feats = test_node_feats[test_g_num]

    visualize_message_passing(g, perform_message_passing, init_node_feats=tf.convert_to_tensor(init_node_feats))