import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib import colors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap

import numpy as np
import networkx as nx

from utils.process import reverse_coordinates


def plot_networkx_graph(skeleton, graph, coordinates, save_fig=False, save_dir="./", **kwargs) -> None:
    """
    Draw a NetworkX Graph object, with the option of overlaying it onto an image using Matplotlib's imshow() method.
    
    Parameters:
        skeleton:

        graph: 
    
        coordinates: where the nodes should be placed when plotted (i.e., the coordinates of the nodes from the input image)
        
    Returns:
        None
    
    """
    plot_coords = reverse_coordinates(coordinates)

    plot_options = {
        "node_size": 50,
        "font_size": 8,
    }
    

    fig, ax = plt.subplots(figsize=(7, 7))
    
    ax.imshow(skeleton, cmap="gray")
    # https://networkx.org/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html
    nx.draw_networkx(graph, pos=plot_coords, ax=ax, **plot_options)
    
    if(save_fig is True):
        if not(os.path.isdir(save_dir)):
            os.mkdir(save_dir)
        #plt.savefig(save_dir+figtitle+".pdf", bbox_inches='tight', format='pdf')
        plt.savefig("skeleton_graph.pdf", bbox_inches='tight', format='pdf')     
        # close figure to save on memory if saving many figures at once
        plt.close(fig)
    else:
        plt.show()


def display_updated_graphs(result_dict, save_fig=False, save_dir="./") -> None:
    """
    Display original_graph side by side with updated_graph, which contains updated node connections (if any
    were detected). Note: if padded_adjacency() is called with connectivity=2, the before and after images will
    be the same. Whereas with padded_adjacnecy(connectivity=1), slanted edges will be detected, thus the before and
    after images will be different.

    Parameters:
        result_dict: a dictionary of processed attributes from a call to padded_adjacency()

    Returns:
        None
    """
    skeleton = result_dict["skeleton"]
    original_graph = result_dict["skeleton_graph_original"]
    graph = result_dict["skeleton_graph"]
    coordinates = result_dict["skeleton_coordinates"]
    plot_coords = reverse_coordinates(coordinates)

    plot_options = {
        "node_size": 50,
        "font_size": 8,
    }
    
    fig, ax = plt.subplots(1, 2, figsize=(9, 7))
    
    # draw skeletons
    ax[0].imshow(skeleton, cmap="gray")
    ax[1].imshow(skeleton, cmap="gray")
    
    # overlay graphs
    nx.draw_networkx(graph, pos=plot_coords, ax=ax[0], **plot_options)
    nx.draw_networkx(graph, pos=plot_coords, ax=ax[1], **plot_options)

    ax[0].axis('off')
    ax[1].axis('off')
    plt.tight_layout()
    
    if(save_fig is True):
        if not(os.path.isdir(save_dir)):
            os.mkdir(save_dir)
        #plt.savefig(save_dir+figtitle+".pdf", bbox_inches='tight', format='pdf')
        plt.savefig("graph_comparison.pdf", bbox_inches='tight', format='pdf')     
        # close figure to save on memory if saving many figures at once
        plt.close(fig)
    else:
        plt.show()


def get_junction_color(length: int) -> str:
    """
    Return a color based on an integer from 1-5+, where 5+ defaults to red.
    
    Note that this method does not check for length <= 0 because that is done
    prior to running this method.

    Parameters:
        length: an int value 1-5+

    Returns:
        a string representing a color
    """
    return {
        1: "dodgerblue",
        2: "green",
        3: "orange",
        4: "purple",
    }.get(length, "red")


def plot_junctions(result_dict, label, idx, save_fig=False, save_dir="./") -> None:
    """
    Overlay a NetworkX graph onto a skeletonized image. Then, color all junction nodes (nodes with 3+ connections)
    a separate color so that they can be easily identified.

    Parameters:
        result_dict: a dictionary of processed attributes from a call to padded_adjacency()

        label: what digit, character, shape, etc. does result_dict represent

        idx: the index of where the digit, character, shape, etc. is found in a dataset

    Returns:
        None
    """
    skeleton = result_dict["skeleton"]
    graph = result_dict["skeleton_graph"]
    node_types = result_dict["node_types"]
    search_by_node = result_dict["search_by_node"] 
    
    junction_color = "red"
    other_color = "gray"

    node_options = {
        "node_size": 100,
    }

    edge_options = {
        "width": 2.0, 
        "alpha": 0.5, 
    }

    legend_options = {
        "xdata": [0],
        "ydata": [0],
        "marker": 'o',
        "markersize": 10,
        "linewidth": 0,
    }

    # get nodes that are junctions and non-junctions
    junction_locations = list(np.where(np.array(node_types)=="J")[0])
    other_locations = list(np.where(np.array(node_types)!="J")[0])

    # subset node location dict for each type of node
    junction_nodes_dict = dict([item for item in search_by_node.items() if item[0] in junction_locations])
    other_nodes_dict = dict([item for item in search_by_node.items() if item[0] not in junction_locations])

    # reverse node locations for plotting in matplotlib
    node_locations_plotting = dict([(k, [v[1], v[0]]) for (k, v) in search_by_node.items()])
    junction_locations_plotting = dict([(k, [v[1], v[0]]) for (k, v) in junction_nodes_dict.items()])
    other_locations_plotting = dict([(k, [v[1], v[0]]) for (k, v) in other_nodes_dict.items()])

    # plot skeleton
    fig, ax = plt.subplots(figsize=(7, 7))
   
    ax.imshow(skeleton, cmap="gray")

    # plot nodes by type
    # non-junction nodes
    nx.draw_networkx_nodes(graph, pos=node_locations_plotting, nodelist=other_locations, node_color=other_color, **node_options)
    nx.draw_networkx_edges(graph, pos=node_locations_plotting, edgelist=nx.to_edgelist(graph, other_locations), edge_color=other_color, **edge_options)

    # junction nodes
    if(len(junction_locations) != 0):
        nx.draw_networkx_nodes(graph, pos=node_locations_plotting, nodelist=junction_locations, node_color=junction_color, **node_options)
        nx.draw_networkx_edges(graph, pos=node_locations_plotting, edgelist=nx.to_edgelist(graph, junction_locations), edge_color=junction_color, **edge_options)

    # add node labels
    nx.draw_networkx_labels(graph, pos=node_locations_plotting, font_size=8)

    # add custom legend elements
    non_junction_label = Line2D(color="gray", markerfacecolor="gray", label='NON-JUNCTION', **legend_options)
    junction_label = Line2D(color='red', markerfacecolor="red", label='JUNCTION', **legend_options)
    
    legend_elements = [non_junction_label, junction_label]
    ax.legend(handles=legend_elements)

    # title, save figure
    figtitle = "node_example_{}_idx_{}".format(label, idx)
    ax.set_title(figtitle)

    plt.axis("off")
    plt.tight_layout()

    if(save_fig is True):
        if not(os.path.isdir(save_dir)):
            os.mkdir(save_dir)
        plt.savefig(save_dir+figtitle+".pdf", bbox_inches='tight', format='pdf')      
        # close figure to save on memory if saving many figures at once
        plt.close(fig)
    else:
        plt.show()


def plot_cliques(result_dict, label, idx, save_fig=False, save_dir="./") -> None:
    """
    Overlay a NetworkX graph onto a skeletonized image. Then, color all junction nodes (nodes with 3+ connections)
    a separate color based on how many nodes there are in a junction cluster.

    Parameters:
        result_dict: a dictionary of processed attributes from a call to padded_adjacency()

        label: what digit, character, shape, etc. does result_dict represent

        idx: the index of where the digit, character, shape, etc. is found in a dataset

    Returns:
        None
    
    """
    skeleton = result_dict["skeleton"]
    graph = result_dict["skeleton_graph"]
    node_types = result_dict["node_types"]
    search_by_node = result_dict["search_by_node"]    
    #cliques = result_dict["cliques"]
    unique_cliques = result_dict["cliques_unique"]
    
    junction_color = "red"
    other_color = "gray"

    node_options = {
        "node_size": 100,
        "alpha": 0.4,
    }

    edge_options = {
        "width": 2.0, 
        "alpha": 0.4, 
    }

    legend_options = {
        "xdata": [0],
        "ydata": [0],
        "marker": 'o',
        "markersize": 10,
        "linewidth": 0,
    }

    # get nodes that are junctions and non-junctions
    junction_locations = list(np.where(np.array(node_types)=="J")[0])
    other_locations = list(np.where(np.array(node_types)!="J")[0])

    # subset node location dict for each type of node
    junction_nodes_dict = dict([item for item in search_by_node.items() if item[0] in junction_locations])
    other_nodes_dict = dict([item for item in search_by_node.items() if item[0] not in junction_locations])

    # reverse node locations for plotting in matplotlib
    node_locations_plotting = dict([(k, [v[1], v[0]]) for (k, v) in search_by_node.items()])
    junction_locations_plotting = dict([(k, [v[1], v[0]]) for (k, v) in junction_nodes_dict.items()])
    other_locations_plotting = dict([(k, [v[1], v[0]]) for (k, v) in other_nodes_dict.items()])

    # plot skeleton
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(skeleton, cmap="gray")

    # plot nodes by type
    # non-junction nodes
    nx.draw_networkx_nodes(graph, pos=node_locations_plotting, nodelist=other_locations, node_color=other_color, **node_options)
    nx.draw_networkx_edges(graph, pos=node_locations_plotting, edgelist=nx.to_edgelist(graph, other_locations), edge_color=other_color, **edge_options)

    # junction nodes
    if(len(unique_cliques) != 0):
        for i, clique in enumerate(unique_cliques):
            nx.draw_networkx_nodes(graph, pos=node_locations_plotting, nodelist=clique, node_color=junction_color, **node_options)
            nx.draw_networkx_edges(graph, pos=node_locations_plotting, edgelist=nx.to_edgelist(graph, clique), edge_color=junction_color, **edge_options)

    # add node labels
    nx.draw_networkx_labels(graph, pos=node_locations_plotting, font_size=8)

    # legend
    # add custom legend elements
    non_junction_label = Line2D(color="gray", markerfacecolor="gray", label='NON-JUNCTION', **legend_options)
    clique_label = Line2D(color='red', markerfacecolor="red", label='CLIQUE', **legend_options)
    
    legend_elements = [non_junction_label, clique_label]
    ax.legend(handles=legend_elements)
    
    # title, save figure
    figtitle = "node_example_{}_idx_{}".format(label, idx)
    ax.set_title(figtitle)

    plt.axis("off")
    plt.tight_layout()
    
    if(save_fig is True):
        if not(os.path.isdir(save_dir)):
            os.mkdir(save_dir)
        plt.savefig(save_dir+figtitle+".pdf", bbox_inches='tight', format='pdf')      
        # close figure to save on memory if saving many figures at once
        plt.close(fig)
    else:
        plt.show()


def plot_primary_junctions(result_dict, label, idx, save_fig=False, save_dir="./") -> None:
    """
    Overlay a NetworkX graph onto a skeletonized image. Color all junction nodes (nodes with 3+ connections)
    a separate color so that they can be easily identified. Then, mark all primary junction nodes (all nodes that
    make up a right triangle in a NetworkX clique of length 3) in a separate color from the other junction nodes.

    Parameters:
        result_dict: a dictionary of processed attributes from a call to padded_adjacency()

        label: what digit, character, shape, etc. does result_dict represent

        idx: the index of where the digit, character, shape, etc. is found in a dataset

    Returns:
        None
    """
    skeleton = result_dict["skeleton"]
    graph = result_dict["skeleton_graph"]
    node_types = result_dict["node_types"]
    search_by_node = result_dict["search_by_node"]
    cliques = result_dict["cliques"]
    primary_junctions = result_dict["junctions_primary"]

    junction_color = "red"
    other_color = "gray"
    
    node_options = {
        "node_size": 100,
    }
    
    edge_options = {
        "width": 2.0, 
        #"alpha": 0.5, 
    }
    
    primary_options = {
        "node_size": 100,
        "node_shape": "o", #options: ‘so^>v<dph8’.
        "edgecolors": "green",
        "linewidths": 2,
        #"alpha": 1,
    }
    
    legend_options = {
        "xdata": [0],
        "ydata": [0],
        "marker": 'o',
        "markersize": 10,
        "linewidth": 0,
    }

    # get nodes that are junctions and non-junctions
    junction_locations = list(np.where(np.array(node_types)=="J")[0])
    other_locations = list(np.where(np.array(node_types)!="J")[0])

    # subset node location dict for each type of node
    junction_nodes_dict = dict([item for item in search_by_node.items() if item[0] in junction_locations])
    other_nodes_dict = dict([item for item in search_by_node.items() if item[0] not in junction_locations])

    # reverse node locations for plotting in matplotlib
    node_locations_plotting = dict([(k, [v[1], v[0]]) for (k, v) in search_by_node.items()])
    junction_locations_plotting = dict([(k, [v[1], v[0]]) for (k, v) in junction_nodes_dict.items()])
    other_locations_plotting = dict([(k, [v[1], v[0]]) for (k, v) in other_nodes_dict.items()])

    # plot skeleton
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(skeleton, cmap="gray")

    # plot nodes by type
    # non-junction nodes
    nx.draw_networkx_nodes(graph, pos=node_locations_plotting, nodelist=other_locations, node_color=other_color, label="NON-JUNCTION", **node_options)
    nx.draw_networkx_edges(graph, pos=node_locations_plotting, edgelist=nx.to_edgelist(graph, other_locations), edge_color=other_color, **edge_options)
    
    # all junction nodes
    if(len(cliques) != 0):
        for clique in cliques:
            nx.draw_networkx_nodes(graph, pos=node_locations_plotting, nodelist=clique, node_color=junction_color, **node_options)
            nx.draw_networkx_edges(graph, pos=node_locations_plotting, edgelist=nx.to_edgelist(graph, clique), edge_color=junction_color, **edge_options)

    # only primary junction nodes
    if(len(primary_junctions) != 0):
        for junction in primary_junctions:
            nx.draw_networkx_nodes(graph, pos=node_locations_plotting, nodelist=primary_junctions, node_color=junction_color, **primary_options)
            
    # add node labels
    nx.draw_networkx_labels(graph, pos=node_locations_plotting, font_size=8)

    # add custom legend elements
    non_junction_label = Line2D(color="gray", markerfacecolor="gray", label='NON-JUNCTION', **legend_options)
    junction_label = Line2D(color='red', markerfacecolor="red", label='JUNCTION', **legend_options)
    primary_junction_label = Line2D(color='red', markerfacecolor="red", markeredgecolor="green", markeredgewidth=2.0, label='PRIMARY JUNCTION', **legend_options)
    
    legend_elements = [non_junction_label, junction_label, primary_junction_label]
    ax.legend(handles=legend_elements)
    
    # title, save figure
    figtitle = f"node_example_{label}_idx_{idx}"
    ax.set_title(figtitle)

    plt.axis("off")
    plt.tight_layout()

    if(save_fig is True):
        if not(os.path.isdir(save_dir)):
            os.mkdir(save_dir)
        plt.savefig(save_dir+figtitle+".pdf", bbox_inches='tight', format='pdf')      
        # close figure to save on memory if saving many figures at once
        plt.close(fig)
    else:
        plt.show()
    

def plot_removed_edges(result_dict, label, idx, save_fig=False, save_dir="./") -> None:
    """
    Plot the edges removed from a graph before path segmentation.

    Parameters:
        result_dict: a dictionary of processed attributes from a call to padded_adjacency()

        label: what digit, character, shape, etc. does result_dict represent

        idx: the index of where the digit, character, shape, etc. is found in a dataset

    Returns:
        None
    """
    skeleton = result_dict["skeleton"]
    path_seg_graph = result_dict["skeleton_graph_path_seg"]
    removed_edges = result_dict["removed_edges"]
    search_by_node = result_dict["search_by_node"]
    coordinates = result_dict["skeleton_coordinates"]

    node_options = {
        "node_size": 100,
        "font_size": 8,
    }

    edge_options = {
        "width": 2.0,
        "edge_color": "red",
    }

    legend_options = {
            "xdata": [0],
            "ydata": [0],
            "linewidth": 4,
            "color": "red",
        }

    plot_coords = reverse_coordinates(coordinates)
    node_locations_plotting = dict([(k, [v[1], v[0]]) for (k, v) in search_by_node.items()])

    fig, ax = plt.subplots(figsize=(7, 7))

    # draw skeletons
    ax.imshow(skeleton, cmap="gray")

    # overlay graphs
    nx.draw_networkx(path_seg_graph, pos=plot_coords, ax=ax, **node_options)

    # show deleted edges
    nx.draw_networkx_edges(path_seg_graph, pos=node_locations_plotting, edgelist=removed_edges, **edge_options)

    # add custom legend elements
    removed_edges_label = Line2D(label='REMOVED EDGES', **legend_options)
    legend_elements = [removed_edges_label]
    ax.legend(handles=legend_elements)
    
    # title, save figure
    figtitle = f"removed_edges_{label}_idx_{idx}"
    ax.set_title(figtitle)

    ax.axis('off')
    plt.tight_layout()

    if(save_fig is True):
        if not(os.path.isdir(save_dir)):
            os.mkdir(save_dir)
        plt.savefig(save_dir+figtitle+".pdf", bbox_inches='tight', format='pdf')      
        # close figure to save on memory if saving many figures at once
        plt.close(fig)
    else:
        plt.show()


def plot_junctions_and_terminals(result_dict, label, idx, save_fig=False, save_dir="./") -> None:
    """
    Plot junctions and terminals (path segmentation start- and endpoints).

    Parameters:
        result_dict: a dictionary of processed attributes from a call to padded_adjacency()

        label: what digit, character, shape, etc. does result_dict represent

        idx: the index of where the digit, character, shape, etc. is found in a dataset

    Returns:
        None
    """  
    skeleton = result_dict["skeleton"]
    path_seg_graph = result_dict["skeleton_graph_path_seg"]
    removed_edges = result_dict["removed_edges"]
    search_by_node = result_dict["search_by_node"]
    coordinates = result_dict["skeleton_coordinates"]
    path_seg_endpoints = result_dict["endpoints_path_seg"]

    junction_color = "red"
    other_color = "gray"

    node_options = {
        "node_size": 100,
    }

    edge_options = {
        "width": 2.0, 
        "alpha": 0.5, 
    }

    legend_options = {
        "xdata": [0],
        "ydata": [0],
        "marker": 'o',
        "markersize": 10,
        "linewidth": 0,
    }

    # get nodes that are junctions and non-junctions
    other_locations = [node for node in path_seg_graph.nodes() if node not in path_seg_endpoints]

    # subset node location dict for each type of node
    junction_nodes_dict = dict([item for item in search_by_node.items() if item[0] in path_seg_endpoints])
    other_nodes_dict = dict([item for item in search_by_node.items() if item[0] not in path_seg_endpoints])

    # reverse node locations for plotting in matplotlib
    node_locations_plotting = dict([(k, [v[1], v[0]]) for (k, v) in search_by_node.items()])
    junction_locations_plotting = dict([(k, [v[1], v[0]]) for (k, v) in junction_nodes_dict.items()])
    other_locations_plotting = dict([(k, [v[1], v[0]]) for (k, v) in other_nodes_dict.items()])

    # plot skeleton
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.imshow(skeleton, cmap="gray")

    # plot nodes by type
    # non-junction nodes
    nx.draw_networkx_nodes(path_seg_graph, pos=node_locations_plotting, nodelist=other_locations, node_color=other_color, **node_options)
    nx.draw_networkx_edges(path_seg_graph, pos=node_locations_plotting, edgelist=nx.to_edgelist(path_seg_graph, other_locations), edge_color=other_color, **edge_options)

    # junction/terminal nodes
    if(len(path_seg_endpoints) != 0):
        nx.draw_networkx_nodes(path_seg_graph, pos=node_locations_plotting, nodelist=path_seg_endpoints, node_color=junction_color, **node_options)
        nx.draw_networkx_edges(path_seg_graph, pos=node_locations_plotting, edgelist=nx.to_edgelist(path_seg_graph, path_seg_endpoints), edge_color=junction_color, **edge_options)

    # add node labels
    nx.draw_networkx_labels(path_seg_graph, pos=node_locations_plotting, font_size=8)

    # add custom legend elements
    non_junction_label = Line2D(color="gray", markerfacecolor="gray", label='NON-JUNCTION', **legend_options)
    junction_label = Line2D(color='red', markerfacecolor="red", label='JUNCTION/TERMINAL', **legend_options)

    legend_elements = [non_junction_label, junction_label]
    ax.legend(handles=legend_elements)

    # title, save figure
    figtitle = f"junctions_and_terminals_{label}_idx_{idx}"
    ax.set_title(figtitle)

    plt.axis("off")
    plt.tight_layout()

    if(save_fig is True):
        if not(os.path.isdir(save_dir)):
            os.mkdir(save_dir)
        plt.savefig(save_dir+figtitle+".pdf", bbox_inches='tight', format='pdf')      
        # close figure to save on memory if saving many figures at once
        plt.close(fig)
    else:
        plt.show()


def plot_graph_paths(result_dict, label, idx, save_fig=False, save_dir="./") -> None:
    """
    Plot paths in a NetworkX graph.

    Parameters:
        result_dict: a dictionary of processed attributes from a call to padded_adjacency()

        label: what digit, character, shape, etc. does result_dict represent

        idx: the index of where the digit, character, shape, etc. is found in a dataset

    Returns:
        None
    """
    graph = result_dict["skeleton_graph"]
    path_seg_graph = result_dict["skeleton_graph_path_seg"]
    skeleton = result_dict["skeleton"]
    paths_list = result_dict["paths_list"]
    endpoints_list = result_dict["endpoints_path_seg"]
    search_by_node = result_dict["search_by_node"]

    # create custom colormap with Set3 as the base
    base_colormap = cm.Set3
    
    # replace lighter colors that are hard to see
    new_colors = [color for color in base_colormap.colors]
    new_colors[1] = (colors.to_rgb("firebrick"))
    new_colors[2] = (colors.to_rgb("darkorchid"))
    new_colors[7] = (colors.to_rgb("lightpink"))
    new_colors[8] = (colors.to_rgb("darkgrey"))
    new_colors[10] = (colors.to_rgb("teal"))
    new_colors[-1] = (colors.to_rgb("gold"))
    new_colors = tuple(new_colors)
    custom_cmap = ListedColormap(colors=new_colors, name="Path Segmentation")

    num_colors = len(custom_cmap.colors)
    random_idx = np.random.choice(num_colors, size=len(paths_list), replace=False)

    outline_color = "red"
    
    node_options = {
        "node_size": 100,
    }

    edge_options = {
        "width": 2.0, 
        "alpha": 0.5, 
    }
    
    endpoint_options = {
        "node_size": 100,
        "node_shape": "o", #options: ‘so^>v<dph8’.
        "linewidths": 2,
        "alpha": 1,
    }

    node_locations_plotting = dict([(k, [v[1], v[0]]) for (k, v) in search_by_node.items()])

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.imshow(skeleton, cmap='gray')

    # nodes
    for i, path in enumerate(paths_list):
        random_color = custom_cmap(random_idx[i])
        color_hex = colors.to_hex(random_color, keep_alpha=False)
        nx.draw_networkx_nodes(graph, pos=node_locations_plotting, nodelist=path, node_color=color_hex, **node_options)
        nx.draw_networkx_edges(graph, pos=node_locations_plotting, edgelist=nx.to_edgelist(path_seg_graph, path), edge_color=color_hex, **edge_options)
        
    # only primary junction nodes
    if(len(endpoints_list) != 0):
        for endpoint in endpoints_list:
            nx.draw_networkx_nodes(graph, pos=node_locations_plotting, nodelist=endpoints_list, node_color=outline_color, **endpoint_options)
    
    # add node labels
    nx.draw_networkx_labels(path_seg_graph, pos=node_locations_plotting, font_size=8)

    # title, save figure
    figtitle = f"path_segmentation_{label}_idx_{idx}"
    ax.set_title(figtitle)
    
    plt.axis("off")
    plt.tight_layout()

    if(save_fig is True):
        if not(os.path.isdir(save_dir)):
            os.mkdir(save_dir)
        plt.savefig(save_dir+figtitle+".pdf", bbox_inches='tight', format='pdf')      
        # close figure to save on memory if saving many figures at once
        plt.close(fig)
    else:
        plt.show()
