import csv
import itertools

import numpy as np

from skimage.filters import threshold_mean
from skimage.morphology import skeletonize
from skimage import graph as skgraph

import networkx as nx


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
    return [np.sum(sub_list[1:]) for sub_list in pixel_values_list]


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


def add_missing_connections(problem_nodes, problem_node_neighbor_idx, node_locations, input_graph):
    """
    This method returns which pixels from neighbors_list are in node_coordinates,
    essentially identifying neighbors are nodes from pixel coordinates.
    
    Parameters:
        problem_nodes: a list integers of nodes that are missing potential connections with neighboring nodes
    
        problem_node_neighbor_idx: a list of (x, y) pairs for neighboring pixels for each node in problem_nodes
    
        node_locations: a dict with strings of (x, y) coordinates as keys and node numbers as values
                        ex: {'[x, y]': node_num}
                    
        input_graph: a NetworkX Graph object from which problem_nodes, problem_node_neighbor_idx, and node_locations are derived
    
    Returns:
        updated_graph: a copy of input_graph with any missing connections (edges) added
    
    """
    updated_graph = input_graph.copy()
    
    for i, element in enumerate(problem_node_neighbor_idx):
        # obtain a list of nodes that should be connected to problem_node
        problem_node = problem_nodes[i]
        potential_connections = [node_locations[str(loc)] for loc in element]
        check_connection = [nx.is_path(updated_graph, [problem_node, node]) for node in potential_connections]
        not_connected_idx = list(np.where(np.array(check_connection) == False)[0])
        not_connected = [potential_connections[idx] for idx in not_connected_idx]
        
        # add connections here
        connections_to_add = [(problem_node, missing_connection) for missing_connection in not_connected]
        updated_graph.add_edges_from(connections_to_add)
        
    return updated_graph


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


def get_node_combinations(clique):
    """
    We are assuming that cliques are of length 3.
    
    Example: given the clique [1, 2, 3], this method will return
    [[1, 2], [1, 3], [2, 3]]
    """
    node_combinations = sorted(list(set(itertools.combinations(clique, 2))))
    node_combinations = [list(combo) for combo in node_combinations]
    return node_combinations


def get_path_weights(node_combinations, search_by_node):
    """
    ...
    
    Parameters:
        node_combinations:
        
        search_by_node:
        
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
    
    Returns:
    
    
    """
    # create a combination of 2 nodes, for the 3 node clique
    node_combinations = get_node_combinations(clique)
    
    # get the path weight: 1 for horizontal/vertical, 1.414 for slanted
    path_weights = get_path_weights(node_combinations, search_by_node)
    
    # find which paths are horizontal/vertical
    hv_indices = list(np.where(np.array(path_weights) == 1)[0])
    horizontal_vertical_edges = [set(node_combinations[idx]) for idx in hv_indices]

    # the primary junction is the junction for which a vertical and horizontal edge meet
    primary_junctions = list(set.intersection(*horizontal_vertical_edges))[0]
    return primary_junctions


def find_removable_edges(clique, search_by_node):
    """
    ..
    
    Parameters:
    
    Returns:
    
    
    """
    # create a combination of 2 nodes, for the 3 node clique
    node_combinations = get_node_combinations(clique)
    
    # get the path weight: 1 for horizontal/vertical, 1.414 for slanted
    path_weights = get_path_weights(node_combinations, search_by_node)

    # find which paths are slanted, we will want to remove these
    # if the path weight is not 1, the only other option is sqrt(2) = 1.414
    s_indices = list(np.where(np.array(path_weights) != 1)[0])

    # these are lists of tuples, because that is what the NetworkX method remove_edges_from() wants
    # https://networkx.org/documentation/stable/reference/classes/generated/networkx.Graph.remove_edges_from.html
    edges_to_remove = [tuple(node_combinations[idx]) for idx in s_indices][0]

    return edges_to_remove


def flatten_list(input_list):
    """
    Return a list of lists with a flattened structure. Each element of
    input_list is now a list, not itself a nested list.
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


def shorten_path(path, endpoints_list):
    """
    This method stops adding elements from path when it finds an element in junctions_list. In this way,
    each path will start and end with a primary junction node and have no primary junctions in between them.
    
    Note, this method assumes that the first element is part of junctions_list, which is why we
    iterate through path[1:].
    
    Example:
    Let's say we have junctions_list = [3, 18, 23, 27, 34, 41, 44] and path = [3, 4, 5, 6, 7, 10, 13, 16, 19, 21, 24, 23, 22, 20, 18]
    The first element of path is in junctions_list ([3]), so we iterate from the element at index 1 until we find another element in junctions_list.
    [18], the last element in path is a junction, but so is [23] which comes before it. So we expect that this method will return
    [3, 4, 5, 6, 7, 10, 13, 16, 19, 21, 24, 23] so that the path only contains nodes between 1 starting and 1 ending node. 
    """
    # https://stackoverflow.com/questions/9572833/using-break-in-a-list-comprehension
    path_start = [path[0]]
    path_middle = list(itertools.takewhile(lambda x: x not in endpoints_list, path[1:]))

    # the list stops when it encounters a primary junction node
    # so we want to add that to the end of the list
    path_end_idx = np.where(np.array(path) == path_middle[-1])[0][0] + 1
    path_end = [path[path_end_idx]]

    short_path = path_start + path_middle + path_end
    return short_path


def find_reversed_list(paths_list):
    """
    In our case, there may be lists that are simply the reverse of each other.
    
    For instance, if the lists contain the *all* of the same elements, but go from start-end and end-start,
    then we only need to keep one of those lists.
    
    Ex:
    [18, 15, 12, 9, 3] is just a reversal of [3, 9, 12, 15, 18]
    This method will find which list it finds first.
    """    
    for i, path in enumerate(paths_list):
        reversed_path = list(reversed(path))
        if(reversed_path in paths_list):
            paths_list.remove(reversed_path)

    paths_list = sorted(paths_list)
    return paths_list
    

def TGGLinesPlus(image, connectivity=2):
    """
    This method is currently designed for one image, though we could also design
    it to work for lists of images. Alternatively, we can keep it how it is and let
    users define a list comprehension on input images like: 
        output_images = [padded_adjacency(image) for image in input_images]

    Parameters:
        image: the input image

        connectivity: an int value
            1 represents vertical and horizontal edges in a graph
            2 represents horizontal, vertical, and slanted edges (full triangle: two lengs and the hypotenuse)

    Returns:

    """    
    # create binary image
    thresh = threshold_mean(image)
    binary = image > thresh
    
    # create skeleton, pad image
    skeleton = skeletonize(binary)
    
    # pad skeleton image AFTER thresholding, otherwise it can affect the resulting skeleton
    skeleton = pad_image(skeleton)

    # then convert to scipy sparse array
    skeleton_array, skeleton_coords = create_skeleton_graph(skeleton, connectivity=connectivity)

    # create graph from scipy sparse array, get node locations and save as dict
    skeleton_graph = nx.from_scipy_sparse_array(skeleton_array)
    search_by_node, search_by_location = get_node_locations(skeleton_coords)
    
    # find neighboring pixels and their values
    neighbor_locations = [find_neighbors(pixel) for pixel in skeleton_coords]
    neighbor_values = get_neighbor_values(neighbor_locations, skeleton)

    # identify degree mismatch for each node in graph
    # if degree mismatch exists, compile a list of each node where this is true in nx_graph
    potential_degrees = get_node_degree(neighbor_values)
    current_degrees = [val for (node, val) in skeleton_graph.degree()]
    problem_nodes = list(np.where(np.array(potential_degrees) - np.array(current_degrees) != 0)[0])

    # find problem nodes
    node_neighbors = node_in_neighbors(neighbor_locations, skeleton_coords)
    problem_node_neighbor_idx = [node_neighbors[idx] for idx in problem_nodes]

    # update any missing connections between neighboring nodes in nx_graph
    skeleton_graph_updated = add_missing_connections(problem_nodes, problem_node_neighbor_idx, search_by_location, skeleton_graph)

    # calculate final node degrees and node types from updated graph
    degrees = [val for (node, val) in skeleton_graph_updated.degree()]
    node_types = list(map(degree_to_node_type, degrees))
    
    # convert graph object back to scipy sparse array object for plotting
    skeleton_array_updated = nx.to_scipy_sparse_array(skeleton_graph_updated)

    # find cliques and primary junction nodes
    junction_locations = list(np.where(np.array(node_types)=="J")[0])
    junction_subgraph = nx.subgraph(skeleton_graph_updated, nbunch=junction_locations)
    
    # find cliques and flatten list of lists returned by NetworkX
    cliques = [list(nx.find_cliques(junction_subgraph, nodes=[junction])) for junction in junction_locations]
    cliques = flatten_list(cliques)
    cliques = [sorted(sublist) for sublist in cliques]
    
    # now find unique cliques (there are many repeats due to clique-triangle permuations, i.e., [3, 4, 5], [3, 5, 4], [4, 3, 5], etc.)
    cliques_set = list(set([tuple(clique) for clique in cliques]))
    unique_cliques = sorted([list(clique_tuple) for clique_tuple in cliques_set])

    # from here, we want to find primary junctions, which are either:
    #   - solo junctions: a junction (by definition with 3+ connections) not connected to any other junctions
    #   - triangle cliques: junctions that form right triangles with themselves
    cliques_1_single_junction = [clique for clique in unique_cliques if len(clique) == 1]
    cliques_1_single_junction = flatten_list(cliques_1_single_junction)

    cliques_3_right_triangles = set([find_primary_junctions(clique, search_by_node) for clique in unique_cliques if len(clique) == 3])
    cliques_3_right_triangles = sorted(list(cliques_3_right_triangles))

    edges_to_remove = [find_removable_edges(clique, search_by_node) for clique in unique_cliques if len(clique) == 3]
    path_seg_graph = skeleton_graph_updated.copy()
    path_seg_graph.remove_edges_from(edges_to_remove)

     # AFTER we find cliques_3_right_triangles, we can find cliques_2_adjacent_nodes
    # after we remove edges, some nodes lose that edge and are no longer junctions (3+ connections)
    # we can ignore these nodes and only focus on "branching" nodes, or nodes that look like a Y (i.e., they become cliques_1_single_junction nodes)
    adjacent_junctions_list = [clique for clique in unique_cliques if len(clique) == 2]
    adjacent_junctions_list = sorted(list(set(flatten_list(adjacent_junctions_list))))
    cliques_2_adjacent_junctions = [node for node in adjacent_junctions_list if len(path_seg_graph.edges(node)) >= 3]

        # for path segmentation, we also want to include "terminal" end nodes
    # the location in node_types is the same node number in graph
    end_nodes = list(np.where(np.array(node_types) == "E")[0])

    primary_junctions = sorted(cliques_1_single_junction + cliques_2_adjacent_junctions + cliques_3_right_triangles)
    path_seg_endpoints = sorted(cliques_1_single_junction + cliques_2_adjacent_junctions + cliques_3_right_triangles + end_nodes)

    # return the updated graph object and important info as dict
    return {
        "cliques": cliques,
        "cliques_unique": unique_cliques,
        "cliques_1_single_junction": cliques_1_single_junction,
        "cliques_2_adjacent_junctions": cliques_2_adjacent_junctions,
        "cliques_3_right_triangles": cliques_3_right_triangles,
        "endpoints_path_seg": path_seg_endpoints,
        "junction_locations": junction_locations,
        "junction_subgraph": junction_subgraph,
        "junctions_primary": primary_junctions,
        "neighbor_locations": neighbor_locations,
        "neighbor_values": neighbor_values,
        "node_degrees": degrees,
        "node_degrees_before_update": current_degrees,
        "node_types": node_types,
        "nodes_terminal": end_nodes,
        #"paths_list": paths_list,
        "removed_edges": edges_to_remove,
        "search_by_location": search_by_location,
        "search_by_node": search_by_node,
        "skeleton": skeleton,
        "skeleton_array": skeleton_array_updated,
        "skeleton_array_old": skeleton_array,
        "skeleton_coordinates": skeleton_coords,
        "skeleton_graph": skeleton_graph_updated,
        "skeleton_graph_original": skeleton_graph,
        "skeleton_graph_path_seg": path_seg_graph,
    }


def total_added_connections(result_dict_list):
    """
    Returns the total number of connections added prior and post to running the processing method padded_adjacency()

    Parameters:
        result_dict_list: a list of result objects from calling padded_adjacency()

    Returns:
        total_added: an int (0+) representing the number of connections added to a graph or list of graph objects
    """
    degrees_before = np.sum([np.sum(result_dict["node_degrees_before_update"]) for result_dict in result_dict_list])
    degrees_after = np.sum([np.sum(result_dict["node_degrees"]) for result_dict in result_dict_list])

    total_added = degrees_after - degrees_before
    print(f"Total connections added: {total_added}")
    return total_added


def merge_lists(lists, results=None):
    """
    Merge lists containing one or more common elements.
    Source: # https://stackoverflow.com/questions/55348640/how-to-merge-lists-with-common-elements-in-a-list-of-lists

    Parameters:
        lists: the lists that you want to try to merge

        results: the list you want to save the results to (optional)

    Returns:
        the input lists after merge

        Note: this will result in <= len(lists), where an unsuccessful merge will return lists

    """
    if results is None:
        results = []

    if not lists:
        return results

    first = lists[0]
    merged = []
    output = []

    for li in lists[1:]:
        for i in first:
            if i in li:
                merged = merged + li
                break
        else:
            output.append(li)

    merged = merged + first
    results.append(list(set(merged)))

    return merge_lists(output, results)


######### TEMP: POSSIBLY DELETE ########
def find_unique_cliques(result_dict): 
    """
    ...

    Parameters:

    Returns:
    
    """
    graph = result_dict["skeleton_graph"]
    node_types = result_dict["node_types"]
    
    junction_locations = list(np.where(np.array(node_types)=="J")[0])
    junctions_subgraph = nx.subgraph(graph, nbunch=junction_locations)
    cliques = [nx.cliques_containing_node(junctions_subgraph, nodes=junction) for junction in junction_locations]
    unique_cliques = sorted([sorted(val) for sublist in cliques for val in sublist])

#     cliques = [sorted(list(nx.find_cliques(junctions_subgraph, nodes=[junction]))[0]) for junction in junction_locations]
#     cliques_set = list(set([tuple(clique) for clique in cliques]))
#     unique_cliques = sorted([sorted(list(clique_tuple)) for clique_tuple in cliques_set])

    keep_merging = True
    cliques_length = len(unique_cliques)

    # there are many non-unique lists; here we will merge them into shapes in a recursive search
    while(keep_merging is True):
        unique_cliques = merge_lists(unique_cliques)

        if(len(unique_cliques) == cliques_length):
            keep_merging = False
        else:
            cliques_length = len(unique_cliques)

    unique_cliques = [sorted(merged_cliques) for merged_cliques in unique_cliques]
    return unique_cliques
