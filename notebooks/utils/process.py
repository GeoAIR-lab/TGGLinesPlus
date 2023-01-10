import csv
import itertools

import numpy as np

from skimage.filters import threshold_mean
from skimage.morphology import skeletonize
from skimage import graph as skgraph

import networkx as nx

import rasterio


def read_in_mnist(filename):
    """
    Returns the original MNIST dataset with lists for dataset images and corresponding labels.
        
    Parameters:
        filename: the location of the MNIST CSV file
    
    Returns:
        images_list: list of a numpy array for each image in the MNIST dataset

        labels_list: a list of digit labels (integers), corresponding to each image in images_list
    """
    with open(filename, 'r') as csv_file:
        images_list = []
        labels_list = []

        fileObject =  csv.reader(csv_file)
                
        for data in fileObject:
            # the first column is the label
            label = data[0]
            labels_list.append(int(label))
            
            # the rest of columns are pixels
            digit_img = data[1:]

            # make those columns into a array of 8-bits pixels
            # this array will be of 1D with length 784
            # the pixel intensity values are integers from 0 to 255
            digit_img = np.array(digit_img, dtype='uint8')

            # reshape the array into 28 x 28 array (2-dimensional array)
            digit_img = digit_img.reshape((28, 28))
            images_list.append(digit_img)

    return images_list, labels_list


def read_in_chinese_mnist(filename):
    """
    Returns the compact CSV version of the Chinese MNIST dataset with lists for dataset images and corresponding labels.
        
    Parameters:
        filename: the location of the Chinese MNIST CSV file
    
    Returns:
        images_list: list of a numpy array for each image in the MNIST dataset

        labels_list: a list of digit labels (integers), corresponding to each image in images_list
    """
    with open(filename, 'r') as csv_file:
            images_list = []
            labels_list = []
            digit_labels_list = []

            fileObject =  csv.reader(csv_file)

            # skip the first line, the header
            next(fileObject)

            for data in fileObject:
                # the last two columns are the labels
                # the last element is the chinese character
                # the second to last element is the digit label assigned to the character
                label = data[-1]
                labels_list.append(label)

                digit_label = data[-2]
                digit_labels_list.append(digit_label)

                # the rest of columns are pixels
                digit_img = data[0:-2]

                # make those columns into a array of 8-bits pixels
                # this array will be of 1D with length 4096
                # the pixel intensity values are integers from 0 to 255
                digit_img = np.array(digit_img, dtype='uint8')

                # reshape the array into 64 x 64 array (2-dimensional array)
                digit_img = digit_img.reshape((64, 64))
                images_list.append(digit_img)
                
    return images_list, labels_list, digit_labels_list


def open_tiff(path):
    """
    Open a
    
    Parameters:
        path: the full name / location of the TIFF file
    
    Returns:
        array: the array extracted from the input TIFF file
    """
    with rasterio.open(path) as dataset:
        
        array = dataset.read(dataset.indexes[0])
    
    return array


def create_binary(image):
    """
    Given an input image, binarize it and return the result.

    Parameters:
        image: the input image as an array
    
    Returns:
        binary: an array representing an image binary
    """
    thresh = threshold_mean(image)
    binary = image > thresh
    
    return binary


def create_binary_reverse(image):
    """
    Given an input image, binarize it and return the reverse of the result.

    Parameters:
        image: the input image as an array
    
    Returns:
        binary: an array representing an image binary
    """
    thresh = threshold_mean(image)
    binary = image < thresh
    
    return binary


def pad_image(image):
    """
    Returns an image with a 1px border along each edge.
    
    Example: a (28, 28) input image will become a (30, 30) image
    
    Parameters:
        image: the input image
    
    Returns:
        the input image with a 1px border all the way around
    
    """
    return np.pad(image, 1)


def create_skeleton(binary):
    """
    Given an input image binary, skeletonize it, pad it, and return the result.

    Parameters:
        binary: an array representing an image bianry
    
    Returns:
        skeleton: an array representing an image skeleton
    """
    skeleton = skeletonize(binary)
    skeleton = pad_image(skeleton)
    
    return skeleton


def create_skeleton_graph(skeleton, connectivity=1):
    """
    Return a list of (x, y) coordinates from a True/False or 0/1 skeleton grid.
    
    Parameters:
        skeleton: a numpy array retrieved from a call to skimage.morphology.skeletonize
    
    Returns
        skeleton_graph: a scipy sparse matrix
    
        skeleton_coords: a list of lists containing [x, y] coordinate pairs for each True/1 pixel in input skeleton

    """
    img_shape = skeleton.shape
    skeleton_graph, ravel_positions = skgraph.pixel_graph(skeleton, connectivity=connectivity)
    
    x_pos, y_pos = np.unravel_index(ravel_positions, img_shape)
    skeleton_coords = [[x_pos[i], y_pos[i]] for i in range(len(x_pos))]
    
    return skeleton_graph, skeleton_coords


def reverse_coordinates(coordinates_list):
    """
    For a list of lists containing [x, y] pairs, return a list of list
    reversing the coordinates.
    
    Parameters:  
        coordinates_list: a list of lists containing [x, y] pairs
    
    Returns:
        coordinates_reversed: a list of lists containing [y, x] pairs (the reverse of coordinates_list)
    
    """
    coordinates_reversed = [[y, x] for [x, y] in coordinates_list]
    
    return coordinates_reversed


def get_node_locations(coordinates_list):
    """
    Return two dictionaries containing either nodes as keys and node locations as values or
    node locations as keys and nodes as values. This is useful because we can look to see if a 
    given pixel in an image is a node in our network, but also determine where nodes are located
    in an image by their position in a network.
    
    Parameters:
        coordinates_list: a list containing (x, y) coordinate pairs of the location for each node
                          in a skeleton graph
    
    Returns:
        search_by_node: a dictionary with nodes as keys and coordinates as values
    
        search_by_location: a reverse lookup dictionary with coordinates as keys and nodes as values
    
    """
    search_by_node = {}

    # create dict where node is key, location is value
    for i, node_location in enumerate(coordinates_list):
        #node_locations[str(node_location)] = i
        search_by_node[i] = node_location
        
    # create reverse lookup dict where location is key, node is value
    dict_items_list = list(search_by_node.items())
    key_values_flipped = [(str(v), k) for (k, v) in dict_items_list]
    search_by_location = dict(key_values_flipped)
    
    return search_by_node, search_by_location


def find_neighbors(pixel):
    """
    Return a list of 8 (x, y) coordinates surrounding a pixel in a grid. This list should
    have the same length as input list pixels_list, but instead of 1 (x, y) pair for each
    entry, there should be 8.
    
    This method works by going clockwise around the center pixel.
    Example for pixel [6, 7]:
    [[5, 7], [5, 8], [6, 8], [7, 8], [7, 7], [7, 6], [6, 6], [5, 6]]
    
    NOTE: this method is an idealized scenario that does not check image boundaries, i.e.,
    all pixels of interest are at least 1 pixel within all image bounds.
    
    Parameters:
        pixel: an (x, y) pixel to get neighbors for
    
    Returns:
        neighbors_list: a nested list of 8 (x, y) neighboring pixels

    """    
    x, y = pixel
    neighbors_list = [[x-1, y], [x-1, y+1], [x, y+1], [x+1, y+1], [x+1, y], [x+1, y-1], [x, y-1], [x-1, y-1]]
        
    return neighbors_list


def get_pixel_values(pixel_list, image):
    """
    Extract the pixel values at a given location in a 2D input image.
    
    Parameters:
        pixel_list: a list of (x, y) pixels from an image
        
        image: a binary input image, either with binary True/False or 0/1 values
    
    Returns:
        pixel_values: the values of each pixel in pixel_list
    
    """
    pixel_values = []
    image = image + 0 # -> from True/False to 0/1, else no change
    
    for pixel in pixel_list:
        x, y = pixel
        pixel_value = image[x, y]
        pixel_values.append(pixel_value)
    
    return pixel_values


def get_neighbor_values(neighbors_list, image):
    """
    For each "neighbor" pixel in neighbor_list, extract the pixel value
    in the input image using the get_pixel_values() method.
    
    Parameters:
        neighbors_list: a list of (x, y) pixels from an image
        
        image: a binary input image, either with binary True/False or 0/1 values
    
    Returns:
        neighbor_values_list: the values for the center node pixel (1 by definition), 
                              followed by the values of its 8 neighboring pixels

    """
    neighbor_values_list = []
    for neighbor in neighbors_list:
        neighbor_values = get_pixel_values(neighbor, image)
        # we want to append [1] to the front of every list
        # because this is the center pixel, which by definition has value 1
        neighbor_values_list.append([1] + neighbor_values)

    return neighbor_values_list


def get_node_degree(pixel_values_list):
    """
    This method returns the degree a given node should have. This means that it returns a degree >=
    the current node degree and is used to find nodes that are missing connections with neighboring nodes.
    
    Parameters:
        pixels_values_list: a list of 9 pixel values: 1 for node, 8 for neighbors
    
    Returns:
        degree of node (how many connections it *should* have, not what it *does* have)
    """
    # we do not want to include the center pixel (node) in our sum
    return [np.sum(sublist[1:]) for sublist in pixel_values_list]


def node_in_neighbors(neighbors_list, node_coordinates):
    """
    This method returns which pixels from neighbors_list are in node_coordinates,
    essentially identifying neighbors are nodes from pixel coordinates.
    
    Parameters:
        neighbors_list: a list of 9 pixel values, 1 for each set of neighbors for a given node
    
        node_coordinates: the (x,y) coordinates of each node in the skeleton graph
    
    Returns:
        node_neighbors: a list of coordinates for which neighbors are nodes
    
    """
    node_neighbors = []
    for neighbors in neighbors_list:
        node_neighbors.append([pixel for pixel in node_coordinates if pixel in neighbors])
    
    return node_neighbors


def degree_to_node_type(degree):
    """
    For an integer degree, return node type: a value of either be J, T, or E
    
    J: junction node, degree >= 3
    T: turning node, degree = 2
    E: end node, degree = 1
    I: isolated node, degree = 0
    
    Parameters:
        degree: an integer representing the number of degrees a node in a graph has
    
    Returns:
        node_type: a mapped value from degree to node type

    """
    node_type = ""
    
    if degree < 0:
        raise ValueError("Degree < 0 not allowed")
    elif degree == 0:
        node_type = "I"
    elif degree == 1:
        node_type = "E"
    elif degree == 2:
        node_type = "T"
    else:
        node_type = "J"
        
    return node_type


def get_unique_cliques(graph, junction_locations):
        """
        Get cliques from a NetworkX subgraph built with the junction nodes in junctions_list
        
        Parameters:
            graph: a NetworkX graph

            junction_locations: a list of nodes in input graph that have 3+ connections

        Returns:
            cliques: a list of cliques resulting from calling nx.find_cliques(graph, nodes=[...]) on every node in junction_locations

            unique_cliques: a unique set of cliques where duplicates have been removed from cliques
        """
        # find cliques and flatten list of lists returned by NetworkX
        cliques = [list(nx.find_cliques(graph, nodes=[junction])) for junction in junction_locations]
        cliques = flatten_list(cliques)
        cliques = [sorted(sublist) for sublist in cliques]
        
        # now find unique cliques (there are many repeats due to clique-triangle permuations, i.e., [3, 4, 5], [3, 5, 4], [4, 3, 5], etc.)
        cliques_set = list(set([tuple(clique) for clique in cliques]))
        unique_cliques = sorted([list(clique_tuple) for clique_tuple in cliques_set])

        return cliques, unique_cliques


def get_node_combinations(clique):
    """
    Given a clique containing 3 nodes, create all combinations between them.
   
    NOTE: we are assuming that cliques have been filtered to find those that are length 3 for this method.
    
    Example: given the clique [1, 2, 3], this method will return
    [[1, 2], [1, 3], [2, 3]]

    Parameters:
        clique: a clique resulting from calling nx.find_cliques(graph, nodes=[node]) on a single node
                
    Returns:
        node_combinations: a list of lists, where each list is a combination of 2 nodes in a clique
    """
    node_combinations = sorted(list(set(itertools.combinations(clique, 2))))
    node_combinations = [list(combo) for combo in node_combinations]
    
    return node_combinations


def get_path_weights(node_combinations, search_by_node):
    """
    ...
    
    Parameters:
        node_combinations: a list of lists, where each list is a combination of 2 nodes in a clique
        
        search_by_node: a dictionary with nodes as keys and coordinates as values
        
    Returns:
        path_weights: a list of path weights for a combination of nodes in a clique of length 3
    
    Example: 
    After finding the following node locations for an individual clique:
    [
      [[9, 14], [9, 15]], 
      [[9, 14], [10, 15]], 
      [[9, 15], [10, 15]]
    ]
    
    The method will calculate the (absolute) distance between each pair:
    [
      # np.absolute([[9-9, 14-15], [9-10, 14-15], [9-10, 15-15]])
      [0, 1], 
      [1, 1], 
      [1, 0]
    ]
    
    We can simply add these values together and take the square root of them, yielding:
    [
      # np.sqrt([0+1, 1+1, 1+0])
      [1, 1.414, 1],
    ]
    
    So we see only nodes that are horizontal or vertical from each other form a right triangle.
    """ 
    path_weights = [np.sqrt(np.sum(np.absolute(np.array(search_by_node[node_pair[0]]) - np.array(search_by_node[node_pair[1]])))) for node_pair in node_combinations]
    
    return path_weights


def find_primary_junctions(clique, search_by_node):
    """
    ..
    
    Parameters:
        clique: a clique resulting from calling nx.find_cliques(graph, nodes=[node]) on a single node

        search_by_node: a dictionary with nodes as keys and coordinates as values
    
    Returns:
        primary_junctions: junction nodes that form the base of a right triangle in a clique of length 3 (that is part of both a horiztonal and vertial edge)
    
    """
    # create a combination of 2 nodes, for the 3 node clique
    node_combinations = get_node_combinations(clique)
    
    # get the path weight: 1 for horizontal/vertical, 1.414 for diagonal
    path_weights = get_path_weights(node_combinations, search_by_node)
    
    # find which paths are horizontal/vertical
    hv_indices = list(np.where(np.array(path_weights) == 1)[0])
    horizontal_vertical_edges = [set(node_combinations[idx]) for idx in hv_indices]

    # the primary junction is the junction for which a vertical and horizontal edge meet
    primary_junctions = list(set.intersection(*horizontal_vertical_edges))[0]

    return primary_junctions


def find_removable_edges(clique, search_by_node):
    """
    This method finds diagonal edges in cliques and returns them to be removed from a graph.
    This method is important in being able to find which junction among adjacent junctions 
    (two junctions connected that are side-by-side) to keep as primary junctions whe we segment
    the paths in a graph.
    
    After removing edges and filtering adjacent junctions, we are able to segment a graph properly. Without doing this,
    traversing these edges will cause problems with the algorithm (specifically: non-unique paths, paths that double back on themselves,
    paths that skip edges, etc.).
    
    Parameters:
        clique: a clique resulting from calling nx.find_cliques(graph, nodes=[node]) on a single node

        search_by_node: a dictionary with nodes as keys and coordinates as values
    
    Returns:
        edges_to_remove: a list of edges to remove in a NetworkX graph
    
    """
    # create a combination of 2 nodes, for the 3 node clique
    node_combinations = get_node_combinations(clique)
    
    # get the path weight: 1 for horizontal/vertical, 1.414 for diagonal
    path_weights = get_path_weights(node_combinations, search_by_node)

    # find which paths are diagonal, we will want to remove these
    # if the path weight is not 1, the only other option is sqrt(2) = 1.414
    diagonal_indices = list(np.where(np.array(path_weights) != 1)[0])

    # these are lists of tuples, because that is what the NetworkX method remove_edges_from() wants
    # https://networkx.org/documentation/stable/reference/classes/generated/networkx.Graph.remove_edges_from.html
    edges_to_remove = [tuple(node_combinations[idx]) for idx in diagonal_indices][0]

    return edges_to_remove


def flatten_list(input_list):
    """
    Return a list of lists with a flattened structure. 
    
    NOTE: if you input a list of lists of lists, you will get back a list of lists.
    However, if you input a list of lists, this will return a list (which can cause problems for other methods).

    Parameters:
        graph: a NetworkX graph
        
        endpoints_list: a list of endpoints (junctions + terminals) that we want to find paths for

    Returns:
        a list with a flattened structure
        
    
    """
    return [val for sublist in input_list for val in sublist]


def get_initial_paths(graph, endpoints_list):
    """
    Return a list of lists, where each list is a NetworkX path between a starting node and an end node.
    
    Parameters:
        graph: a NetworkX graph
        
        endpoints_list: a list of endpoints (junctions + terminals) that we want to find paths for

    Returns:
        initial_paths: a list containing paths from a starting to ending node pair in endpoints_list
    """

    initial_paths = []

    for i, node in enumerate(endpoints_list[:-1]):
        next_node = endpoints_list[i+1]
        paths_list = list(nx.all_simple_paths(graph, node, next_node))
        initial_paths.append(paths_list)

    # flatten list of lists of lists to list of lists
    initial_paths = flatten_list(initial_paths)

    return initial_paths


def split_path(initial_paths, endpoints_list):
    """
    Given a set of initial paths in a graph, we want to extract all "subpaths" from them where
    paths are split so that they start and end with a path segmentation endpoint.
    
    Parameters:
        initial_paths: the initial graphs found in a NetworkX graph, from calling nx.all_simple_paths(graph, node1, node2)
                       for all nodes in endpoints_list
        
        endpoints_list: a list of endpoints (junctions + terminals) that we want to find paths for

    Returns:
        split_paths: a unique set of lists in initial_paths

    """
    split_paths = []

    for path in initial_paths:
        found_endpoints = [node for node in path if node in endpoints_list]
        
        endpoints_idx = [path.index(endpoint) for endpoint in found_endpoints]
        
        # https://stackoverflow.com/questions/21752610/iterate-every-2-elements-from-list-at-a-time
        endpoint_combos = [[start, end] for start, end in zip(endpoints_idx[:-1], endpoints_idx[1:])]
        
        subpaths = [path[start: end+1] for start, end in endpoint_combos]
        
        split_paths.append(subpaths)

    split_paths = [tuple(sublist) for sublist in flatten_list(split_paths)]
    split_paths = [list(sublist) for sublist in list(set(split_paths))]

    return split_paths


def shorten_path(path, endpoints_list):
    """
    This method stops adding elements from path when it finds an element in junctions_list. In this way,
    each path will start and end with a primary junction node and have no primary junctions in between them.
    
    NOTE: this method assumes that the first element is part of junctions_list, which is why we
    iterate through path[1:].
    
    Example:
    Let's say we have junctions_list = [3, 18, 23, 27, 34, 41, 44] and path = [3, 4, 5, 6, 7, 10, 13, 16, 19, 21, 24, 23, 22, 20, 18]
    The first element of path is in junctions_list ([3]), so we iterate from the element at index 1 until we find another element in junctions_list.
    [18], the last element in path is a junction, but so is [23] which comes before it. So we expect that this method will return
    [3, 4, 5, 6, 7, 10, 13, 16, 19, 21, 24, 23] so that the path only contains nodes between 1 starting and 1 ending node. 

    Parameters:
        path: a list of nodes in a NetworkX graph
        
        endpoints_list: a list of endpoints (junctions + terminals) that we want to find paths for

    Returns:
        short_path: a (potentially) shortened list starting and ending with a primary node (and with no primary nodes between them)
    """
    # https://stackoverflow.com/questions/9572833/using-break-in-a-list-comprehension
    path_start = [path[0]]
    path_middle = list(itertools.takewhile(lambda x: x not in endpoints_list, path[1:]))

    # the list stops when it encounters a primary junction node
    # so we want to add that node to the end of the list
    # there is a possibility that path[1] is also an endpoint, so if path_middle is [], we need to add path[1] as the end of the list
    # (i.e., the path is 2 nodes long)
    if (len(path_middle) == 0):
        path_end = [path[1]]
    else:
        #path_end_idx = np.where(np.array(path) == path_middle[-1])[0][0] + 1
        path_end_idx = path.index(path_middle[-1]) + 1
        path_end = [path[path_end_idx]]

    short_path = path_start + path_middle + path_end

    return short_path


def remove_reversed_list(paths_list):
    """
    In our case, there may be lists that are simply the reverse of each other.
    
    For instance, if the lists contain the *all* of the same elements, but go from start-end and end-start,
    then we only need to keep one of those lists.
    
    Ex:
    [18, 15, 12, 9, 3] is just a reversal of [3, 9, 12, 15, 18]
    This method will find which list it finds first.

    Parameters:        
        paths_list: a list of lists, with each sublist containing a path of nodes in a NetworkX graph

    Returns:
        paths_list: a version of the input paths_list with any paths removed that are mirrors (reversals) of another path in paths_list
    """    
    for i, path in enumerate(paths_list):
        reversed_path = list(reversed(path))
        if(reversed_path in paths_list):
            paths_list.remove(reversed_path)

    paths_list = sorted(paths_list)

    return paths_list


def segment_paths(graph, endpoints_list):
    """
    Segment a graph from given list of endpoints. Here, we define endpoints to be:
    - terminal nodes
    - junction nodes that we have found and filtered via different methods, primarily those that are
        solo (form a T), that branch (form a Y), and that form the base of right triangles

    Since NetworkX defines nodes in a graph from top to bottom, left to right, we take a sorted list (endpoints_list)
    and create paths between pairs of nodes starting from least (top-left most) and ending with the highest (bottom-right most).
    The goal is to get unique paths that span the graph, with each path starting and ending with a node in endpoints_list (continaing no
    nodes in endpoints_list between them).

    Parameters:
        graph: a NetworkX graph
        
        endpoints_list: a list of endpoints (junctions + terminals) that we want to find paths for

    Returns:
        final_paths_list: a list of lists containing unique paths in input graph
    """
    # initial_paths_list: a nested list of lists of paths between a start and end node
    initial_paths_list = get_initial_paths(graph, endpoints_list)

    # a set of unique paths in a graph, all of which start and end with a path segmentation endpoint
    split_paths_list = split_path(initial_paths_list, endpoints_list)

    # prevent path reversals from being included, e.g. [1, 2, 3] and not also [3, 2, 1]
    final_paths_list = remove_reversed_list(split_paths_list)

    return final_paths_list


def add_loops(graph):
    """
    ...

    Parameters:
        graph: a NetworkX graph
        
    Returns:
        loop_path:
    """
    # In most cases, there will be at least one end node or junction node. 
    # But in a special case where the graph is perfectly circular (not in a geometric case, but in the sense each node leads to the next in a circular manner), 
    # then the path of the graph should be the graph itself.
    # find which other nodes the first node is connected to the top, left-most node in NetworkX
    top_left_node = sorted(list(graph.nodes))[0]
    left_right_connection_nodes = [node for node in flatten_list(list(graph.edges(top_left_node))) if node != top_left_node]

    potential_paths = [list(nx.all_simple_paths(graph, top_left_node, node)) for node in left_right_connection_nodes]
    # since the nodes in left_right_connection_nodes are adjacent to node the top-left node, 
    # two of the paths will be [top_left_node, x], [top_left_node, y]; but we want the full paths that complete the full graph circle
    potential_paths = [sublist for sublist in flatten_list(potential_paths) if len(sublist) > 2]
    
    if(len(potential_paths) == 0):
        loop_path = []
        print("Problem image: {}".format(idx))
    else:
        loop_path = sorted(potential_paths)[0]

    return loop_path


def add_cycles(graph, endpoints_list):
    """
    ...

    Parameters:
        graph: a NetworkX graph
        
        endpoints_list: a list of endpoints (junctions + terminals) that we want to find paths for

    Returns:
        cycle_paths
    """
    
    # we need to check whether our paths span the graph
    # if not, the reason is because there are cycles, so we will need to add them
    cycle_paths = nx.cycle_basis(graph)

    # shift paths so that the path starts with the endpoint
    for idx, cycle in enumerate(cycle_paths):
        if(cycle[0] not in endpoints_list or cycle[-1] not in endpoints_list):
            endpoint = [node for node in cycle if node in endpoints_list][0]
            endpoint_idx = cycle.index(endpoint)
            new_cycle = cycle[endpoint_idx:] + cycle[0:endpoint_idx]
            
            # NetworX's cycle_basis() method does not include the starting point of the path twice
            # so we want to make sure that our endpoint is the start and end of the graph
            if(new_cycle[0] in endpoints_list):
                new_cycle.append(endpoint)
            else:
                new_cycle.insert(0, endpoint)

            cycle_paths[idx] = new_cycle

    # cycles can include multiple segmentation endpoints, so we need to shorten these paths
    cycle_paths = sorted(list(set([tuple(shorten_path(path, endpoints_list)) for path in cycle_paths])))
    cycle_paths = [list(path) for path in cycle_paths]
    
    return cycle_paths


def TGGLinesPlus(skeleton, progress=False):
    """
    This method is currently designed for one image skeleton, though it also works for lists of skeletons. 
    For instance, you can use a list comprehension on a list of input images like so: 
        image_binaries_list = [create_binaries(image) for image in input_images_list]
        image_skeletons_list = [create_seletons(binary) for binary in image_binaries_list]
        results_dict_list = [TGGLinesPlus(skeleton) for skeleton in image_skeletons_list]

    Parameters:
        skeleton: an array representing an image skeleton (image --> binary --> skeleton)

    Returns:
        a dictionary of important values and objects generated during the method

    """
    # set default values for variables that depend on the value of connectivity
    all_paths_list = []
    path_seg_graphs_list = []
    path_seg_endpoints_list = []

    ### CREATE GRAPHS ####
    if(progress):
        print("Creating image skeleton and graph...")
    # convert skeleton to scipy sparse array, then create graph from scipy sparse array
    skeleton_array, skeleton_coordinates = create_skeleton_graph(skeleton, connectivity=2)
    skeleton_graph = nx.from_scipy_sparse_array(skeleton_array)
    search_by_node, search_by_location = get_node_locations(skeleton_coordinates)

    # get subgraphs from main graph
    subgraph_nodes = [list(subgraph) for subgraph in list(nx.connected_components(skeleton_graph))]
    skeleton_subgraphs = [skeleton_graph.subgraph(c).copy() for c in subgraph_nodes]
    
    # remove subgraphs with less than 3 nodes, this might just be noise or "skeckle" in the image
    subgraph_nodes = [node_list for node_list in subgraph_nodes if len(node_list) >= 3]
    skeleton_subgraphs = [subgraph for subgraph in skeleton_subgraphs if len(subgraph.nodes()) >= 3]
    num_subgraphs = len(skeleton_subgraphs)

    ### PATH SEGMENTATION ####
    if(progress):
        print("Starting path segmentation...")
    try:
        for idx, subgraph in enumerate(skeleton_subgraphs):
            if(progress):
                print("       Segmenting subgraph {} of {}".format(idx+1, num_subgraphs))
            ### GRAPH PATH SIMPLIFICATION ####
            nodes = list(subgraph.nodes)
            
            # calculate final node degrees and node types from updated graph
            degrees = [val for (node, val) in subgraph.degree()]
            node_types = list(map(degree_to_node_type, degrees))

            # create NetworkX subgraph from junction nodes to find cliques
            junction_locations = list(np.where(np.array(node_types)=="J")[0])
            junction_nodes = [nodes[idx] for idx in junction_locations]
            junction_subgraph = nx.subgraph(subgraph, nbunch=junction_nodes)

            # find cliques and primary junction nodes
            cliques, unique_cliques = get_unique_cliques(junction_subgraph, junction_nodes)
            edges_to_remove = [find_removable_edges(clique, search_by_node) for clique in unique_cliques if len(clique) == 3]

            path_seg_graph = subgraph.copy()
            path_seg_graph.remove_edges_from(edges_to_remove)
            path_seg_graphs_list.append(path_seg_graph)

            ### FIND ENDPOINTS, SEGMENT PATHS ####
            # AFTER we find which edges to remove, we can update our junctions list
            # after we remove edges, some nodes lose that edge and are no longer junctions (3+ connections)
            degrees_ = [val for (node, val) in path_seg_graph.degree()]
            node_types_ = list(map(degree_to_node_type, degrees_))
            junction_locations_ = list(np.where(np.array(node_types_)=="J")[0])
            junctions_updated = [nodes[idx] for idx in junction_locations_]

            # for path segmentation, we also want to include "terminal" end nodes
            end_node_locations = list(np.where(np.array(node_types) == "E")[0])    
            end_nodes = [nodes[idx] for idx in end_node_locations]

            # segment paths
            path_seg_endpoints = sorted(junctions_updated + end_nodes)
            path_seg_endpoints_list.append(path_seg_endpoints)
            paths_list = segment_paths(path_seg_graph, path_seg_endpoints)
            
            # if paths_list is an empty list at this point, then there are no path segmentation endpoints
            # this only happens when a subgraph is a perfect loop, so we can add those here
            if(len(paths_list) == 0):
                loop_path = add_loops(path_seg_graph)
                paths_list.append(loop_path)
            
            # lastly, we need to check for whether the paths span the graph
            # if they don't, then we know there are cycles within it and need to add them
            paths_set = tuple(flatten_list(paths_list))
            nodes_set = tuple(path_seg_graph.nodes())
            uncovered_nodes = list(set(nodes_set)-set(paths_set))
            
            if(len(uncovered_nodes) > 0): # paths do not span the graph
                cycle_paths = add_cycles(path_seg_graph, path_seg_endpoints)
                paths_list += cycle_paths

            all_paths_list.append(paths_list)
        
        if(progress):
            print("Done")
        # return the updated graph object and important info as dict
        return {
        #     # "cliques": cliques,
        #     # "cliques_unique": unique_cliques,
        #     # "cliques_1_single_junction": cliques_1_single_junction,
        #     # "cliques_2_adjacent_junctions": cliques_2_adjacent_junctions,
        #     # "cliques_3_right_triangles": cliques_3_right_triangles,
        #     # "junction_locations": junction_locations,
        #     # #"junction_subgraph": junction_subgraph,
        #     # "junctions_primary": primary_junctions,
        #     # #"neighbor_locations": neighbor_locations,
        #     # #"neighbor_values": neighbor_values,
        #     # "nodes_terminal": end_nodes,
        #     # "removed_edges": edges_to_remove,
        #     # "search_by_location": search_by_location,
            "all_paths_list": all_paths_list, 
            "node_degrees": degrees,
            "node_types": node_types,
            "path_seg_graphs_list": path_seg_graphs_list, 
            "path_seg_endpoints_list": path_seg_endpoints_list, 
            "search_by_node": search_by_node,
            "skeleton": skeleton,
            "skeleton_coordinates": skeleton_coordinates,
            "skeleton_graph": skeleton_graph,
            "skeleton_subgraphs": skeleton_subgraphs,
        }
    except Exception as error:
        print("{}".format(error))
