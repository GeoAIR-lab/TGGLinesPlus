# ---------------------------------------------------------------------------
#       TGGLinesPlus Algorithm Python Implementation
#       Website: https://geoair.lipingyang.org/
#       Copyright (C) 2022-2025 GeoAIR Lab
#--------------------------------------------------
#  License
#     This file is part of TGGLinesPlus.

#     TGGLinesPlus python implementation is free software: 
#     you can redistribute it and/or modify it
#     under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     TGGLinesPlus is distributed in the hope that it will be useful, but WITHOUT
#     ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
#     FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
#     for more details.

#     You should have received a copy of the GNU General Public License
#     along with TGGLinesPlus GitHub repository.  If not, see <http://www.gnu.org/licenses/>.

#---------------------------------------------------------------------------


from collections import Counter
import csv
import itertools
import timeit

import numpy as np

from skimage.filters import threshold_mean
from skimage.morphology import skeletonize
from skimage import graph as skgraph

import networkx as nx
# this type alias is for type checking
nxGraph = nx.classes.graph.Graph

import rasterio


def read_in_mnist(filename: str):
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


def read_in_chinese_mnist(filename: str):
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


def open_tiff(path: str) -> np.ndarray:
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


def create_binary(image: np.ndarray) -> np.ndarray:
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


def create_binary_reverse(image: np.ndarray) -> np.ndarray:
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


def pad_image(image: np.ndarray) -> np.ndarray:
    """
    Returns an image with a 1px border along each edge.
    
    Example: a (28, 28) input image will become a (30, 30) image
    
    Parameters:
        image: the input image
    
    Returns:
        the input image with a 1px border all the way around
    
    """
    return np.pad(image, 1)


def create_skeleton(binary: np.ndarray) -> np.ndarray:
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


def create_skeleton_graph(skeleton: np.ndarray, connectivity: int = 1):
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


def reverse_coordinates(coordinates_list: list) -> list:
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


def get_node_locations(coordinates_list: list):
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


def find_junctions(graph: nxGraph, node_list: list):
        """
        Find degrees for each node in a graph and convert that to node type.

        Parameters:
            graph: a NetworkX graph

            node_list: a list of nodes in input graph

         Returns:
            node_types: a list of mapped values from node degree to node type (see method degree_to_node_type() for more info)

            junction_nodes: a list of junction nodes in input graph
        """
        degrees = [val for (node, val) in graph.degree()]
        node_types = list(map(degree_to_node_type, degrees))
        junction_locations = list(np.where(np.array(node_types)=="J")[0])
        junction_nodes = [node_list[idx] for idx in junction_locations]

        return node_types, junction_nodes


def find_neighbors(pixel: tuple) -> list:
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


def get_pixel_values(pixel_list: list, image: np.ndarray) -> list:
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


def get_neighbor_values(neighbors_list: list, image: np.ndarray) ->list:
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


def get_node_degree(pixel_values_list: list) -> list:
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


def node_in_neighbors(neighbors_list: list, node_coordinates: list) -> list:
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


def degree_to_node_type(degree: int) -> str:
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


def get_unique_cliques(graph: nxGraph, junction_locations: list):
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


def get_node_combinations(clique: list) -> list:
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


def get_path_weights(node_combinations: list, search_by_node: dict) -> list:
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


def find_primary_junctions(clique: list, search_by_node: dict) -> list:
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


def find_removable_edges(clique: list, search_by_node: dict) -> list:
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


def flatten_list(input_list: list) -> list:
    """
    Return a list of lists with a flattened structure. 
    
    NOTE: if you input a list of lists of lists, you will get back a list of lists.
    However, if you input a list of lists, this will return a list (which can cause problems for other methods).

    Parameters:        
        input_list: a list of lists

    Returns:
        a list with a flattened structure
        
    
    """
    return [val for sublist in input_list for val in sublist]


def is_subset(list_1: list, list_2: list) -> set:
    """
    Calculate the multiset difference between two lists, i.e., is one set a subset of a larger set
    """
    # https://stackoverflow.com/questions/15147751/how-to-check-if-all-items-in-a-list-are-there-in-another-list
    set_1, set_2 = Counter(list_1), Counter(list_2)
    
    return not set_1 - set_2


def get_initial_paths(graph: nxGraph, pathseg_points_list: list):
    """
    Return a list of lists, where each list is a NetworkX path between a starting node and an end node.
    
    Parameters:
        graph: a NetworkX graph
        
        pathseg_points_list: a list of points (junctions + terminals) that we want to find paths for

    Returns:
        initial_paths: a list containing paths from a starting to ending node pair in pathseg_points_list
    """

    initial_paths = []

    # for each node in pathseg_points_list check for a path between the next point
    for i, node in enumerate(pathseg_points_list[:-1]):
        pathseg_points_sublist = pathseg_points_list[i+1:]

        for next_node in pathseg_points_sublist:
            # if the length of this list is 0, then it means the only neighbors that 
            # the current node might have are path segmentation points
            neighbors_not_points_list = [node for node in list(graph.neighbors(node)) if node not in pathseg_points_list]
            # we can check to see if there are neighbors that are path segmentation points
            pathseg_neighbors = [node for node in list(graph.neighbors(node)) if node in pathseg_points_list]
            # if both conditions are met below, it means that all non-path segmentation point paths have been checked
            # if(len(neighbors_not_points_list) == 0 and len(pathseg_neighbors) == 0):
            if(len(neighbors_not_points_list) == 0 and len(pathseg_neighbors) == 0):
                break

            # checking to make sure there is a path saves time
            if(nx.has_path(graph, node, next_node)):
                # we need to iterate over this, because a node can connect to another node along multiple paths
                # ex: say we are interested in the path between nodes 3 and 18
                # this could produce the following paths:
                # [3, 9, 12, 15, 18], [3, 2, 1, 0, 8, 11, 14, 17, 18], [3, 4, 5, 6, 7, 10, 13, 16, 19, 21, 24, 23, 22, 20, 18]
                # all of which we want to include in our final paths list
                while(nx.has_path(graph, node, next_node)):
                    shortest_path = list(nx.shortest_path(graph, node, next_node))
                    initial_paths.append(shortest_path)
                    #if the path consists solely of path segmentation points, we don't need to check this path again
                    if(is_subset(shortest_path, pathseg_points_list)):
                        break

                    nodes_to_remove = [node for node in shortest_path if node not in pathseg_points_list]
                    graph.remove_nodes_from(nodes_to_remove)

    return initial_paths, graph


def format_list(input_list: list) -> list:
    """
    This method will reverse a list if the last element in the list is less
    than the first element of the list.

    NOTE: This method is cosmetic, but also helps detect duplicate paths in a NetworkX graph. Additionally, it helps 
    trace paths in a NetworkX graph since it will always orient a path to start with the top, left-most node out of the 
    path segmentation points at input_list[0] and input_list[-1].
    
    Parameters:        
        input_list: a list of nodes in a NetworkX graph
    
    Returns: either the orginal input_list, or a reversed list so that the first element is always less than the last element
        
    """
    if(input_list[0] > input_list[-1]):
        return list(reversed(input_list))
    else:
        return input_list


def add_cycles(graph: nxGraph, current_paths_list: list, pathseg_points_list: list) -> list:
    """
    ...

    Parameters:
        graph: a NetworkX graph
        
        current_paths_list: the current list of paths to add cycle_paths to

        pathseg_points_list: a list of points (junctions + terminals) that we want to find paths for

    Returns:
        full_paths_list: current_paths_list + cycles
    """
    
    # this could return a list of lists containing cycle paths, or [] if no cycles exist
    # networkx.shortest_path(g, node, node) will return [node], which is not what we want
    updated_cycles_list = []

    cycle_paths = nx.cycle_basis(graph)
    
    for cycle in cycle_paths:
        found_pathseg_point = [node for node in cycle if node in pathseg_points_list]
        # if this is an empty list, then there are no path segmentation points (perfect loop case)
        if(len(found_pathseg_point) == 0):
            updated_cycles_list.append(cycle)
        # else, there could be 1+ path segmentation points in the cycle
        # normally only 1, where the loop starts and ends at a given node
        # but it can also be the case that multiple junctions form a "chain" and make up part of a loop
        else:
            pathseg_point_idx = [cycle.index(pathseg_point) for pathseg_point in found_pathseg_point][0]
            updated_cycle = cycle[pathseg_point_idx:] + cycle[:pathseg_point_idx]
            updated_cycles_list.append(updated_cycle)
    
    # add starting node to the end of the list so that the cycle starts and ends at the same node
    updated_cycles_list = [cycle + [cycle[0]] for cycle in updated_cycles_list]
    paths_plus_cycles = current_paths_list + updated_cycles_list
    
    return paths_plus_cycles


# https://stackoverflow.com/questions/176918/finding-the-index-of-an-item-in-a-list, https://datagy.io/python-list-index/
def find_all_indices(search_list: list, search_item) -> list:
    """
    Given an input list search_list, iterate through each iteam and return the indices
    for each match of search_item.
    
    Parameters:
        search_list: the list to search through
        
        search_item: the item to match on

    Returns:
        a list of all (if any) indices mathcing the locations of search_item in search_list
    """
    return [index for (index, item) in enumerate(search_list) if item == search_item]


def split_path(current_paths_list: list, pathseg_points_list: list) -> list:
    """
    Given a set of initial paths in a graph, we want to extract all "subpaths" from them where
    paths are split so that they start and end with a path segmentation point.
    
    Parameters:
        current_paths_list: the initial graphs found in a NetworkX graph, from calling nx.all_simple_paths(graph, node1, node2)
                       for all nodes in pathseg_points_list
        
        pathseg_points_list: a list of points (junctions + terminals) that we want to find paths for

    Returns:
        split_paths: a unique set of lists in initial_paths
    """
    split_paths = []

    for path in current_paths_list:
        found_pathseg_points = [node for node in path if node in pathseg_points_list]
        if(len(found_pathseg_points) == 0):
            split_paths.append([path])
        else:
            found_pathseg_points = list(set(tuple(found_pathseg_points)))
            pathseg_points_idx = sorted(flatten_list([find_all_indices(path, pathseg_point) for pathseg_point in found_pathseg_points]))
            
            # https://stackoverflow.com/questions/21752610/iterate-every-2-elements-from-list-at-a-time
            pathseg_point_combos = [[start, end] for start, end in zip(pathseg_points_idx[:-1], pathseg_points_idx[1:])]
            subpaths = [path[start: end+1] for start, end in pathseg_point_combos]
            split_paths.append(subpaths)

    split_paths = [tuple(format_list(sublist)) for sublist in flatten_list(split_paths)]
    split_paths = sorted([list(sublist) for sublist in list(set(split_paths))])

    return split_paths


def segment_paths(graph: nxGraph, pathseg_points_list: list) -> list:
    """
    Segment a graph from given list of path segmentation points. Here, we define path segmentation points to be:
    - terminal nodes
    - junction nodes that we have found and filtered via different methods, primarily those that are
        solo (form a T), that branch (form a Y), and that form the base of right triangles

    Since NetworkX defines nodes in a graph from top to bottom, left to right, we take a sorted list (pathseg_points_list)
    and create paths between pairs of nodes starting from least (top-left most) and ending with the highest (bottom-right most).
    The goal is to get unique paths that span the graph, with each path starting and ending with a node in pathseg_points_list (continaing no
    nodes in pathseg_points_list between them).

    Parameters:
        graph: a NetworkX graph
        
        pathseg_points_list: a list of points (junctions + terminals) that we want to find paths for

    Returns:
        final_paths_list: a list of lists containing unique paths in input graph
    """
    # get initial paths list
    # these paths need to be split and there may be cycles to add
    initial_paths_list, graph = get_initial_paths(graph, pathseg_points_list)

    # add cycles to paths list, if they exist
    paths_plus_cycles = add_cycles(graph, initial_paths_list, pathseg_points_list)
    
    # the shortest_path() algorithm produces paths that may contain other sub-paths
    # so we want to split each path so that it only contains a pathseg_points_list at the start and end of each path
    split_paths_list = split_path(paths_plus_cycles, pathseg_points_list)

    # final_paths_list = [format_list(sublist) for sublist in paths_plus_cycles]
    final_paths_list = sorted(split_paths_list)

    return final_paths_list


def TGGLinesPlus(skeleton: np.ndarray) -> dict:
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
    start = timeit.default_timer()
    
    # this list will be used to keep track of sublists
    subgraphs_list = []
    
    ### CREATE GRAPH ####
    # convert skeleton to scipy sparse array, then create graph from scipy sparse array
    skeleton_array, skeleton_coordinates = create_skeleton_graph(skeleton, connectivity=2)
    skeleton_graph = nx.from_scipy_sparse_array(skeleton_array)
    search_by_node, search_by_location = get_node_locations(skeleton_coordinates)

    # get subgraphs from main graph
    # we need to get the nodes for each connected component in subgraph because nx.subgraph() expects a list of nodes called 'nbunch'
    subgraph_nodes = [list(subgraph) for subgraph in list(nx.connected_components(skeleton_graph))]
    subgraphs = [skeleton_graph.subgraph(node_list).copy() for node_list in subgraph_nodes]

    # we do want to keep a list of potentially "noisy" nodes, so we make sure we have full path coverage of the graph
    speckle_nodes = [node_list for node_list in subgraph_nodes if len(node_list) < 3]
    
    # otherwise, remove subgraphs with less than 3 nodes, this might just be noise or "speckle" in the image
    subgraph_nodes = [node_list for node_list in subgraph_nodes if len(node_list) >= 3]
    subgraphs = [subgraph for subgraph in subgraphs if len(subgraph.nodes()) >= 3]

    for idx, subgraph in enumerate(subgraphs):
        ### GRAPH PATH SIMPLIFICATION ####
        # calculate node degrees and node types from graph
        nodes = list(subgraph.nodes)
        node_types, junction_nodes = find_junctions(subgraph, nodes)

        # create NetworkX subgraph from junction nodes to find cliques
        junction_subgraph = nx.subgraph(subgraph, nbunch=junction_nodes)

        # find cliques and primary junction nodes
        cliques, unique_cliques = get_unique_cliques(junction_subgraph, junction_nodes)
        edges_to_remove = [find_removable_edges(clique, search_by_node) for clique in unique_cliques if len(clique) == 3]

        simple_subgraph = subgraph.copy()
        simple_subgraph.remove_edges_from(edges_to_remove)
        pathseg_graph = simple_subgraph.copy()

        # we need to re-calcualte degrees and node types as path simplification may have removed some junctions
        node_types_updated, junction_nodes_updated = find_junctions(pathseg_graph, nodes)

        # for path segmentation, we also want to include "terminal" end nodes
        end_node_locations = list(np.where(np.array(node_types_updated) == "E")[0])    
        end_nodes = [nodes[idx] for idx in end_node_locations]
        
        ### PATH SEGMENTATION #### 
        # collect junctions and end nodes
        pathseg_points = sorted(junction_nodes_updated + end_nodes)
        paths_list = segment_paths(pathseg_graph, pathseg_points)

        # there is some repetition in returned values here
        # if we did not re-include things like search_by_node, skeleton, etc., then the same plotting methods
        # for the main graph and paths list would not work for subgraphs and their individual path lists
        # NOTE: this does not include the 'runtime' parameter like the whole TGGLinesPlus method does
        subgraph_dict = {
            "cliques": unique_cliques,
            "end_nodes": end_nodes,
            "junction_nodes": junction_nodes_updated,
            "node_types": node_types_updated,
            "paths_list": paths_list, 
            "pathseg_points": pathseg_points, 
            "removed_edges": edges_to_remove,
            "search_by_location": search_by_location,
            "search_by_node": search_by_node,
            "simple_graph": simple_subgraph,
            "skeleton": skeleton,
            "skeleton_coordinates": skeleton_coordinates,
            "skeleton_graph": subgraph,
            "speckle_nodes": speckle_nodes,
        }

        subgraphs_list.append(subgraph_dict)

    # now combine subgraph lists into flattened lists for reporting and plotting
    cliques = sorted(flatten_list([subgraph_dict["cliques"] for subgraph_dict in subgraphs_list]))
    end_nodes = sorted(flatten_list([subgraph_dict["end_nodes"] for subgraph_dict in subgraphs_list]))
    junction_nodes = sorted(flatten_list([subgraph_dict["junction_nodes"] for subgraph_dict in subgraphs_list]))
    node_types = sorted(flatten_list([subgraph_dict["node_types"] for subgraph_dict in subgraphs_list]))
    paths_list = sorted(flatten_list([subgraph_dict["paths_list"] for subgraph_dict in subgraphs_list]))
    pathseg_points = sorted(flatten_list([subgraph_dict["pathseg_points"] for subgraph_dict in subgraphs_list]))
    removed_edges = sorted(flatten_list([subgraph_dict["removed_edges"] for subgraph_dict in subgraphs_list]))

    simple_graph = skeleton_graph.copy()
    simple_graph.remove_edges_from(removed_edges)

    # lastly, we need to check for whether the paths span the graph
    # if they don't, then we know there are cycles within it and need to add them
    nodes_set = set(tuple(skeleton_graph.nodes()))
    paths_set = set(tuple(flatten_list(paths_list)))
    speckle_set = set(tuple(flatten_list(speckle_nodes)))
    paths_plus_noise = paths_set.union(speckle_set)
    uncovered_nodes = nodes_set - paths_plus_noise

    # check to see if paths (minus noise in the image) span the graph
    if(len(uncovered_nodes) > 0):
        print("Not every node in the graph is covered by a path.")
        print("Uncovered nodes: ", uncovered_nodes)
        print()
        raise Exception("Not every node in the graph is covered by a path.")

    stop = timeit.default_timer()
    runtime = stop - start

    # return the updated graph object and important info as dict
    return {
        "cliques": cliques,
        "end_nodes": end_nodes,
        "junction_nodes": junction_nodes,
        "node_types": node_types,
        "paths_list": paths_list, 
        "pathseg_points": pathseg_points, 
        "removed_edges": removed_edges,
        "runtime": runtime,
        "search_by_location": search_by_location,
        "search_by_node": search_by_node,
        "simple_graph": simple_graph,
        "skeleton": skeleton,
        "skeleton_coordinates": skeleton_coordinates,
        "skeleton_graph": skeleton_graph,
        "subgraphs_list": subgraphs_list,
    }


def print_stats(result_dict: dict) -> dict:
    """
    Print useful statistics about the image skeleton and graph after the TGGLinesPlus() method has completed.

    Parameters:
        result_dict: a NetworkX graph
        
    Returns:
        stats_dict: a dictionary of the computed statistics that are printed out to the user
    """
    num_junctions = len(result_dict["junction_nodes"])
    num_terminals = len(result_dict["end_nodes"])
    num_pathseg_points = len(result_dict["pathseg_points"])
    num_graph_nodes = len(result_dict["skeleton_graph"].nodes())
    percent_pathseg_points = num_pathseg_points / num_graph_nodes

    num_skeleton_pixels = len(np.where(result_dict["skeleton"] == 1)[0])
    num_image_pixels = result_dict["skeleton"].flatten().shape[0]
    percent_skeleton_pixels = num_skeleton_pixels / num_image_pixels

    num_subgraphs = len(result_dict["subgraphs_list"])
    
    runtime = result_dict["runtime"]
    
    # create metrics dictionary below
    stats_dict = {}
    stats_dict["runtime"] = runtime
    stats_dict["num_junctions"] = num_junctions
    stats_dict["num_terminals"] = num_terminals
    stats_dict["num_pathseg_points"] = num_pathseg_points
    stats_dict["num_graph_nodes"] = num_graph_nodes
    stats_dict["percent_pathseg_points"] = percent_pathseg_points
    stats_dict["num_image_pixels"] = num_image_pixels
    stats_dict["percent_skeleton_pixels"] = percent_skeleton_pixels

    print("Number of junctions:                      ", num_junctions)
    print("Number of terminal nodes:                 ", num_terminals)
    print("Number of path segmentation points:       ", num_pathseg_points)
    print("Number of nodes in graph:                 ", num_graph_nodes)
    print("Path seg points as total node percent:    ", np.round(percent_pathseg_points, 3))
    print("------------------------------------------")
    print("Number of subgraphs in main graph:        ", num_subgraphs)
    print("------------------------------------------")
    print("Number of pixels in image:                ", num_image_pixels)
    print("Skeleton pixels as total image percent:   ", np.round(percent_skeleton_pixels, 3))
    print("------------------------------------------")
    print(f"Time to run:                               {(runtime):.5f}s")
    print()
    
    return stats_dict
