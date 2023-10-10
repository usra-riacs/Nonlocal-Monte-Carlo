import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.sparse import csr_matrix
import seaborn as sns


def generate_adjacency(n, levels):
    """
    Generate an adjacency matrix for the graph with a Wishart backbone and trees branching from it.

    Parameters:
    - n (int): Number of nodes in the backbone.
    - levels (int): Number of levels of trees branching from each node of the backbone.

    Returns:
    - np.array: Adjacency matrix for the generated graph.
    """

    # Calculate the total number of nodes in the graph, considering the backbone and branching trees
    total_nodes = n * (2 ** (levels + 1) - 1)
    adj_matrix = np.zeros((total_nodes, total_nodes))

    # Create connections between all nodes in the backbone to form a fully connected graph
    adj_matrix[:n, :n] = np.ones((n, n)) - np.eye(n)

    # Create tree branches branching out from each node in the backbone
    curr_node = n
    for i in range(n):
        queue = [i]
        for level in range(1, levels + 1):
            next_queue = []
            for parent_node in queue:
                # Establish connections between the parent node and its children
                adj_matrix[parent_node, curr_node] = 1
                adj_matrix[parent_node, curr_node + 1] = 1
                adj_matrix[curr_node, parent_node] = 1
                adj_matrix[curr_node + 1, parent_node] = 1

                # Queue up the children nodes for the next level
                next_queue.extend([curr_node, curr_node + 1])
                curr_node += 2
            queue = next_queue
    return adj_matrix


def assign_random_weights(adj_matrix, backbone_nodes, min_outside_weight, max_outside_weight, min_backbone_weight,
                          max_backbone_weight):
    """
    Assign random weights to the edges of the graph.

    Parameters:
    - adj_matrix (np.array): Initial adjacency matrix.
    - backbone_nodes (int): Number of nodes in the backbone.
    - min_outside_weight (float): Minimum possible weight for connections outside the backbone.
    - max_outside_weight (float): Maximum possible weight for connections outside the backbone.
    - min_backbone_weight (float): Minimum possible weight for connections inside the backbone.
    - max_backbone_weight (float): Maximum possible weight for connections inside the backbone.

    Returns:
    - np.array: Weighted adjacency matrix.
    """

    # Assign random weights to the edges within the backbone
    for i in range(backbone_nodes):
        for j in range(i + 1, backbone_nodes):
            # Generate a random weight for the connection
            weight = min_backbone_weight + (max_backbone_weight - min_backbone_weight) * np.random.rand()
            # Apply certain conditions to make the weight negative or positive
            weight = -abs(weight) if (i + j) % 2 == 0 else abs(weight)
            adj_matrix[i, j] = weight
            adj_matrix[j, i] = weight

    # Assign random weights to edges connecting backbone nodes to tree nodes
    rand_weights = (min_outside_weight + (max_outside_weight - min_outside_weight) * np.random.rand(backbone_nodes,
                                                                                                    len(adj_matrix) - backbone_nodes)) * adj_matrix[
                                                                                                                                         0:backbone_nodes,
                                                                                                                                         backbone_nodes:]
    adj_matrix[0:backbone_nodes, backbone_nodes:] = rand_weights
    adj_matrix[backbone_nodes:, 0:backbone_nodes] = rand_weights.T

    # Assign random weights to edges within the trees
    outside_weights = (min_outside_weight + (max_outside_weight - min_outside_weight) * np.random.rand(
        len(adj_matrix) - backbone_nodes, len(adj_matrix) - backbone_nodes)) * adj_matrix[backbone_nodes:,
                                                                               backbone_nodes:]
    adj_matrix[backbone_nodes:, backbone_nodes:] = outside_weights

    # Ensure the adjacency matrix remains symmetric after weight assignment
    adj_matrix = np.maximum(adj_matrix, adj_matrix.T)
    return adj_matrix


def add_cross_connections(adj_matrix, backbone_nodes, num_cross_connections, min_cross_connection_weight,
                          max_cross_connection_weight):
    """
    Add random cross connections between nodes outside the backbone.

    Parameters:
    - adj_matrix (np.array): Weighted adjacency matrix.
    - backbone_nodes (int): Number of nodes in the backbone.
    - num_cross_connections (int): Number of cross connections to be added.
    - min_cross_connection_weight (float): Minimum possible weight for cross connections.
    - max_cross_connection_weight (float): Maximum possible weight for cross connections.

    Returns:
    - np.array: Adjacency matrix with added cross connections.
    """

    total_nodes = len(adj_matrix)
    cross_connections = set()

    # Keep adding random connections until the desired number is reached
    while len(cross_connections) < num_cross_connections:
        # Randomly select two nodes outside the backbone
        node1 = np.random.randint(backbone_nodes, total_nodes)
        node2 = np.random.randint(backbone_nodes, total_nodes)

        # Ensure the nodes are distinct and no previous connection exists between them
        if node1 != node2 and (node1, node2) not in cross_connections and (node2, node1) not in cross_connections:
            # Generate a random weight for the connection
            weight = min_cross_connection_weight + (
                        max_cross_connection_weight - min_cross_connection_weight) * np.random.rand()

            # Update the adjacency matrix with the new connection
            adj_matrix[node1, node2] = weight
            adj_matrix[node2, node1] = weight
            cross_connections.add((node1, node2))
    return adj_matrix


def remove_random_backbone_edges(adj_matrix, backbone_nodes, num_edges):
    """
    Remove random connections from within the backbone.

    Parameters:
    - adj_matrix (np.array): Weighted adjacency matrix.
    - backbone_nodes (int): Number of nodes in the backbone.
    - num_edges (int): Number of edges to be removed from the backbone.

    Returns:
    - np.array: Adjacency matrix with selected edges removed.
    """

    removed_edges = set()

    # Keep removing random edges until the desired number is reached
    while len(removed_edges) < num_edges:
        # Randomly select two nodes from the backbone
        node1 = np.random.randint(0, backbone_nodes)
        node2 = np.random.randint(0, backbone_nodes)

        # Ensure the nodes are distinct, a connection exists between them, and they haven't been chosen before
        if node1 != node2 and adj_matrix[node1, node2] != 0 and (node1, node2) not in removed_edges and (
        node2, node1) not in removed_edges:
            # Remove the connection between the nodes
            adj_matrix[node1, node2] = 0
            adj_matrix[node2, node1] = 0
            removed_edges.add((node1, node2))
    return adj_matrix


def txt_to_A_wishart(txtfile):
    """
    Convert a txt file into matrices J and h.

    The txt file should contain data in the format:
    spin1 spin2 value

    If spin1 not equals spin2, the value is assigned to J (interaction).
    There is no h bias for wishart instances, so h is assigned to 0.

    Parameters:
    - txtfile (str): Path to the txt file.

    Returns:
    - tuple: Tuple containing sparse matrix J and h matrices.
    """

    W = {}

    # Parse the txt file and populate the dictionary with interaction values
    with open(txtfile, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            x = list(map(float, line.split()))
            if int(x[0]) == int(x[1]):
                continue
            W[(int(x[0]), int(x[1]))] = x[2]
            W[(int(x[1]), int(x[0]))] = x[2]

    N = max(max(W.keys())) + 1
    h = np.zeros((N, 1))
    W_matrix = np.zeros((N, N))

    # Convert the dictionary into a 2D matrix
    for (i, j), value in W.items():
        W_matrix[i, j] = value

    # Convert the matrix to a sparse format for efficient storage and computations
    W_sparse = csr_matrix(W_matrix)

    return W_sparse, h


def save_to_txt(J, h, filename):
    """
    Save the matrices J and h to a txt file in the desired format.

    Parameters:
    - J (np.array): Interaction matrix.
    - h (np.array): Bias vector.
    - filename (str): Name of the txt file to save the data.
    """

    # Open the file in write mode
    with open(filename, 'w') as f:

        # Iterate over the upper triangle of J to avoid duplicates
        for i in range(J.shape[0]):
            for j in range(i, J.shape[1]):
                if J[i, j] != 0:  # Only write non-zero interactions
                    f.write(f"{i} {j} {-J[i, j]}\n") # saving -J for positive hamiltonian convention

        # If h is provided, write the biases to the file
        if h is not None:
            for i in range(len(h)):
                if h[i] != 0:  # Only write non-zero biases
                    f.write(f"{i} {i} {-h[i]}\n") # saving -h for positive hamiltonian convention


def main(instances=1, base_seed=1345):
    """
    Main function to generate, modify, and visualize the adjacency matrix for multiple instances.

    Parameters:
    - instances (int): Number of instances to generate. Default is 1.
    - base_seed (int): Base seed for random number generation. Default is 1345.
    """

    # Generate the adjacency matrix for the graph
    backbone_nodes = 50 # size of wishart
    outside_tree_levels = 2
    alpha = 0.20

    # Create the main directory if it doesn't exist
    main_dir_name = "wishart_contrived_trees"
    if not os.path.exists(main_dir_name):
        os.makedirs(main_dir_name)

    subfolder_name = f"wishart_planting_N_{backbone_nodes}_alpha_{alpha:.2f}_contrived_tree"
    subfolder_path = os.path.join(main_dir_name, subfolder_name)

    # Create the subfolder if it doesn't exist
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

    for inst in range(1, instances + 1):
        np.random.seed(base_seed + inst)  # Adjust the seed based on the base seed and instance number
        adj_matrix = generate_adjacency(backbone_nodes, outside_tree_levels)

        # Assign random weights to the edges
        max_h = 0.2
        max_outside_weight = 1
        max_backbone_weight = 10
        min_outside_weight = -max_outside_weight
        min_backbone_weight = -max_backbone_weight
        adj_matrix = assign_random_weights(adj_matrix, backbone_nodes, min_outside_weight, max_outside_weight,
                                           min_backbone_weight, max_backbone_weight)

        min_cross_connection_weight = -1
        max_cross_connection_weight = 1
        num_cross_connections = 50
        adj_matrix = add_cross_connections(adj_matrix, backbone_nodes, num_cross_connections, min_cross_connection_weight,
                                           max_cross_connection_weight)

        # Remove random backbone edges
        num_remove_edges = 0
        adj_matrix = remove_random_backbone_edges(adj_matrix, backbone_nodes, num_remove_edges)

        J = adj_matrix

        path = f"wishart_planting_N_{backbone_nodes}_alpha_{alpha:.2f}"
        file_name = f"wishart_planting_N_{backbone_nodes}_alpha_{alpha:.2f}_inst_{inst}.txt"
        file_path = f"{path}/{file_name}"
        J_wishart, _ = txt_to_A_wishart(file_path)

        J_wishart = -np.array(J_wishart)  # negative J for wishart
        J_wishart = J_wishart.toarray()

        J[0:backbone_nodes, 0:backbone_nodes] = max_backbone_weight * J_wishart / np.max(np.abs(J_wishart))
        h = (np.random.rand(len(J)) - 0.5) * 2 * max_h * max_backbone_weight

        # Generate a unique file name for this instance
        file_name = f"wishart_planting_N_{backbone_nodes}_alpha_{alpha:.2f}_inst_{inst}_contrived_tree.txt"
        output_path = os.path.join(subfolder_path, file_name)

        # Save the matrix to a .txt file
        save_to_txt(J, h, output_path)

        # # Plot the adjacency matrix
        # J_plot = J.copy()
        # J_plot[J == 0] = np.nan
        #
        # # Create a colormap and set NaNs to be black
        # cmap = cm.gray.copy()
        # cmap.set_bad(color='black')
        #
        # plt.figure(figsize=(10, 10))
        # sns.heatmap(J_plot, cmap=cmap, xticklabels=10, yticklabels=10)
        # plt.title(f'Adjacency Matrix - Instance {inst}')
        # plt.xlabel('Index of Node')
        # plt.ylabel('Index of Node')
        #
        # # Modify tick locations and labels for every 10th entry
        # ticks = np.arange(0, len(J), 10)
        # plt.xticks(ticks, ticks)
        # plt.yticks(ticks, ticks)
        #
        # plt.show()

if __name__ == "__main__":
    main(instances=50, base_seed=1345)