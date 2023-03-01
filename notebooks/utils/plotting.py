import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib import colors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap

import numpy as np
import networkx as nx
# this type alias is for type checking
nxGraph = nx.classes.graph.Graph

from utils.process import reverse_coordinates


def plot_graph(skeleton: np.ndarray, graph: nxGraph, coordinates: list, search_by_node: dict, label: str = "", node_size: int = 100, node_labels: bool = True, label_size: int = 12, save_fig:bool = False, save_dir: str = "./", **kwargs) -> None:
    """
    Draw a NetworkX Graph object, with the option of overlaying it onto an image using Matplotlib's imshow() method.
    
    Parameters:
        skeleton: an image skeleton

        graph: a NetworkX graph
    
        coordinates: where the nodes should be placed when plotted (i.e., the coordinates of the nodes from the input image)

        search_by_node: a dictionary with nodes as keys and coordinates as values

        label: what digit, character, shape, etc. does result_dict represent

        node_size: an integer, how large you want the nodes to look on the graph (100 is good for small images, 30 better for bigger
                
        node_labels: boolean, whether to draw the node labels on the figure (does not look good for large graphs

        label_size: int, size of the font for labeling node numbers

        save_fig: boolean, whether to save the figure or not
        
        save_dir: string path for where to save the figure to
        
    Returns:
        None
    
    """
    node_options = {
        "node_size": node_size,
    }

    edge_options = {
        "edge_color": "black",
        "width": 2.0,
    }

    label_options = {
        "font_size": label_size,
    }

    plot_coords = reverse_coordinates(coordinates)
    node_locations_plotting = dict([(k, [v[1], v[0]]) for (k, v) in search_by_node.items()])

    fig, ax = plt.subplots(figsize=(7, 7))

    # draw skeletons
    ax.imshow(skeleton, cmap="gray")

    # draw graph node and edges
    nx.draw_networkx_nodes(graph, pos=plot_coords, ax=ax, **node_options)
    nx.draw_networkx_edges(graph, pos=node_locations_plotting, edgelist=nx.to_edgelist(graph, search_by_node), **edge_options)

    # add node labels
    if(node_labels):
        nx.draw_networkx_labels(graph, pos=node_locations_plotting, **label_options)
    
    plt.axis("off")
    plt.tight_layout()
    plt.margins(0)
    
    if(save_fig is True):
        if not(os.path.isdir(save_dir)):
            os.mkdir(save_dir)
        # title, save figure
        if(label != ""):
            figtitle = "original_graph_{}".format(label)
        else: 
            figtitle = "original_graph"
            
        plt.savefig(save_dir+figtitle+".png", format="png", dpi=300, bbox_inches='tight')      
        # close figure to save on memory if saving many figures at once
        plt.close(fig)
    else:
        plt.show()


def plot_cliques(result_dict: dict, label: str = "", node_size:int = 100, node_labels:bool = True, label_size: int = 8, save_fig: bool = False, save_dir: str = "./") -> None:
    """
    Overlay a NetworkX graph onto a skeletonized image. Then, color all junction nodes (nodes with 3+ connections)
    a separate color based on how many nodes there are in a junction cluster.

    Parameters:
        result_dict: a dictionary of processed attributes from a call to TGGLinesPlus()

        label: what digit, character, shape, etc. does result_dict represent

        node_size: an integer, how large you want the nodes to look on the graph (100 is good for small images, 30 better for bigger
                
        node_labels: boolean, whether to draw the node labels on the figure (does not look good for large graphs

        label_size: int, size of the font for labeling node numbers

        save_fig: boolean, whether to save the figure or not
        
        save_dir: string path for where to save the figure to

    Returns:
        None
    
    """
    skeleton = result_dict["skeleton"]
    graph = result_dict["skeleton_graph"]
    node_types = result_dict["node_types"]
    search_by_node = result_dict["search_by_node"]    
    cliques = result_dict["cliques"]
    
    junction_color = "red"
    other_color = "gray"

    node_options = {
        "node_size": node_size,
        "alpha": 0.4,
    }

    edge_options = {
        "width": 2.0, 
        "alpha": 0.4, 
    }

    label_options = {
        "font_size": label_size,
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
    if(len(cliques) != 0):
        for i, clique in enumerate(cliques):
            nx.draw_networkx_nodes(graph, pos=node_locations_plotting, nodelist=clique, node_color=junction_color, **node_options)
            nx.draw_networkx_edges(graph, pos=node_locations_plotting, edgelist=nx.to_edgelist(graph, clique), edge_color=junction_color, **edge_options)

    # add node labels
    if(node_labels):
        nx.draw_networkx_labels(graph, pos=node_locations_plotting, **label_options)

    # legend
    # add custom legend elements
    non_junction_label = Line2D(color="gray", markerfacecolor="gray", label='NODE', **legend_options)
    clique_label = Line2D(color='red', markerfacecolor="red", label='CLIQUE', **legend_options)
    
    legend_elements = [non_junction_label, clique_label]
    ax.legend(handles=legend_elements, bbox_to_anchor=(0.95, 0.95))
    
    plt.axis("off")
    plt.tight_layout()
    
    if(save_fig is True):
        if not(os.path.isdir(save_dir)):
            os.mkdir(save_dir)
        # title, save figure
        if(label != ""):
            figtitle = "cliques_{}".format(label)
        else: 
            figtitle = "cliques"

        plt.savefig(save_dir+figtitle+".png", format="png", dpi=300, bbox_inches='tight')      
        # close figure to save on memory if saving many figures at once
        plt.close(fig)
    else:
        plt.show()
    

def plot_removed_edges(result_dict: dict, label: str = "", node_size:int = 100, node_labels: bool = True, label_size: int = 8, save_fig: bool = False, save_dir: str = "./") -> None:
    """
    Plot the edges removed from a graph before path segmentation.

    Parameters:
        result_dict: a dictionary of processed attributes from a call to TGGLinesPlus()

        label: what digit, character, shape, etc. does result_dict represent

        node_size: an integer, how large you want the nodes to look on the graph (100 is good for small images, 30 better for bigger
                
        node_labels: boolean, whether to draw the node labels on the figure (does not look good for large graphs

        label_size: int, size of the font for labeling node numbers

        save_fig: boolean, whether to save the figure or not
        
        save_dir: string path for where to save the figure to

    Returns:
        None
    """
    skeleton = result_dict["skeleton"]
    graph = result_dict["simple_graph"]
    removed_edges = result_dict["removed_edges"]
    search_by_node = result_dict["search_by_node"]
    coordinates = result_dict["skeleton_coordinates"]

    node_options = {
        "node_size": node_size,
    }

    edge_options = {
        "edge_color": "black",
        "width": 2.0,
    }

    removed_edge_options = {
        "edge_color": "red",
        "width": 2.0,
    }

    label_options = {
        "font_size": label_size,
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

    # draw graph node and edges
    nx.draw_networkx_nodes(graph, pos=plot_coords, ax=ax, **node_options)
    nx.draw_networkx_edges(graph, pos=node_locations_plotting, edgelist=nx.to_edgelist(graph, search_by_node), **edge_options)

    # overlay deleted edges
    nx.draw_networkx_edges(graph, pos=node_locations_plotting, edgelist=removed_edges, **removed_edge_options)

    # add node labels
    if(node_labels):
        nx.draw_networkx_labels(graph, pos=node_locations_plotting, **label_options)

    # add custom legend elements
    removed_edges_label = Line2D(label='REMOVED EDGES', **legend_options)
    legend_elements = [removed_edges_label]
    ax.legend(handles=legend_elements, bbox_to_anchor=(0.95, 0.95))

    ax.axis('off')
    plt.tight_layout()

    if(save_fig is True):
        if not(os.path.isdir(save_dir)):
            os.mkdir(save_dir)
        # title, save figure
        if(label != ""):
            figtitle = f"removed_edges_{label}"
        else: 
            figtitle = "removed_edges"

        plt.savefig(save_dir+figtitle+".png", format='pdf', dpi=300, bbox_inches='tight')      
        # close figure to save on memory if saving many figures at once
        plt.close(fig)
    else:
        plt.show()


def plot_simplified_graph(result_dict: dict, label: str = "", node_size: int = 100, node_labels: bool = True, label_size: int = 8, save_fig: bool = False, save_dir: str = "./") -> None:
    """
    Plot NetworkX graph after it has been simplified, i.e., after removing 45 degree edges in cliques.

    Parameters:
        result_dict: a dictionary of processed attributes from a call to TGGLinesPlus()

        label: what digit, character, shape, etc. does result_dict represent

        node_size: an integer, how large you want the nodes to look on the graph (100 is good for small images, 30 better for bigger
                
        node_labels: boolean, whether to draw the node labels on the figure (does not look good for large graphs

        label_size: int, size of the font for labeling node numbers

        save_fig: boolean, whether to save the figure or not
        
        save_dir: string path for where to save the figure to

    Returns:
        None
    
    """
    skeleton = result_dict["skeleton"]
    simple_graph = result_dict["simple_graph"]
    search_by_node = result_dict["search_by_node"]    
    coordinates = result_dict["skeleton_coordinates"]

    node_options = {
        "node_size": node_size,
    }

    edge_options = {
        "edge_color": "black",
        "width": 2,
    }

    label_options = {
        "font_size": label_size,
    }

    plot_coords = reverse_coordinates(coordinates)
    node_locations_plotting = dict([(k, [v[1], v[0]]) for (k, v) in search_by_node.items()])

    fig, ax = plt.subplots(figsize=(7, 7))

    # draw skeletons
    ax.imshow(skeleton, cmap="gray")

    # draw graph node and edges
    nx.draw_networkx_nodes(simple_graph, pos=plot_coords, ax=ax, **node_options)
    nx.draw_networkx_edges(simple_graph, pos=node_locations_plotting, edgelist=nx.to_edgelist(simple_graph, search_by_node), **edge_options)

    # add node labels
    if(node_labels):
        nx.draw_networkx_labels(simple_graph, pos=node_locations_plotting, **label_options)
    
    plt.axis("off")
    plt.tight_layout()
    
    if(save_fig is True):
        if not(os.path.isdir(save_dir)):
            os.mkdir(save_dir)
        # title, save figure
        if(label != ""):
            figtitle = "simplified_graph_{}".format(label)
        else: 
            figtitle = "simplified_graph"

        plt.savefig(save_dir+figtitle+".png", format="png", dpi=300, bbox_inches='tight')      
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


def plot_junctions(result_dict: dict, label: str = "", node_size: int = 100, node_labels: bool = True, label_size: int = 8, save_fig: bool = False, save_dir: str = "./") -> None:
    """
    Overlay a NetworkX graph onto a skeletonized image. Then, color all junction nodes (nodes with 3+ connections)
    a separate color so that they can be easily identified.

    Parameters:
        result_dict: a dictionary of processed attributes from a call to TGGLinesPlus()

        label: what digit, character, shape, etc. does result_dict represent

        node_size: an integer, how large you want the nodes to look on the graph (100 is good for small images, 30 better for bigger
                
        node_labels: boolean, whether to draw the node labels on the figure (does not look good for large graphs

        label_size: int, size of the font for labeling node numbers

        save_fig: boolean, whether to save the figure or not
        
        save_dir: string path for where to save the figure to

    Returns:
        None
    """
    skeleton = result_dict["skeleton"]
    simple_graph = result_dict["simple_graph"]
    search_by_node = result_dict["search_by_node"]
    junction_nodes = result_dict["junction_nodes"]
    
    junction_color = "red"
    other_color = "gray"

    node_options = {
        "node_size": node_size,
    }

    edge_options = {
        "width": 2.0, 
        "alpha": 0.5, 
    }

    label_options = {
        "font_size": label_size,
    }

    legend_options = {
        "xdata": [0],
        "ydata": [0],
        "marker": 'o',
        "markersize": 10,
        "linewidth": 0,
    }

    # get nodes that are junctions and non-junctions
    other_nodes = [node for node in simple_graph.nodes() if node not in junction_nodes]

    # subset node location dict for each type of node
    junction_nodes_dict = dict([item for item in search_by_node.items() if item[0] in junction_nodes])
    other_nodes_dict = dict([item for item in search_by_node.items() if item[0] not in junction_nodes])

    # reverse node locations for plotting in matplotlib
    node_locations_plotting = dict([(k, [v[1], v[0]]) for (k, v) in search_by_node.items()])
    junction_locations_plotting = dict([(k, [v[1], v[0]]) for (k, v) in junction_nodes_dict.items()])
    other_locations_plotting = dict([(k, [v[1], v[0]]) for (k, v) in other_nodes_dict.items()])

    # plot skeleton
    fig, ax = plt.subplots(figsize=(7, 7))
   
    ax.imshow(skeleton, cmap="gray")

    # plot nodes by type
    # non-junction nodes
    nx.draw_networkx_nodes(simple_graph, pos=node_locations_plotting, nodelist=other_nodes, node_color=other_color, **node_options)
    nx.draw_networkx_edges(simple_graph, pos=node_locations_plotting, edgelist=nx.to_edgelist(simple_graph, other_nodes), edge_color=other_color, **edge_options)

    # junction nodes
    if(len(junction_nodes) != 0):
        nx.draw_networkx_nodes(simple_graph, pos=node_locations_plotting, nodelist=junction_nodes, node_color=junction_color, **node_options)
        nx.draw_networkx_edges(simple_graph, pos=node_locations_plotting, edgelist=nx.to_edgelist(simple_graph, junction_nodes), edge_color=junction_color, **edge_options)

    # add node labels
    if(node_labels):
        nx.draw_networkx_labels(simple_graph, pos=node_locations_plotting, **label_options)

    # add custom legend elements
    junction_label = Line2D(color='red', markerfacecolor="red", label='JUNCTION', **legend_options)
    
    legend_elements = [junction_label]
    ax.legend(handles=legend_elements, bbox_to_anchor=(0.95, 0.95))

    plt.axis("off")
    plt.tight_layout()

    if(save_fig is True):
        if not(os.path.isdir(save_dir)):
            os.mkdir(save_dir)
        # title, save figure
        if(label != ""):
            figtitle = f"junctions_{label}"
        else: 
            figtitle = "junctions"

        plt.savefig(save_dir+figtitle+".png", format='png', dpi=300, bbox_inches='tight')      
        # close figure to save on memory if saving many figures at once
        plt.close(fig)
    else:
        plt.show()


def plot_terminals(result_dict: dict, label: str = "", node_size: int = 100, node_labels: bool = True, label_size: int = 8, save_fig: bool = False, save_dir: str = "./") -> None:
    """
    Overlay a NetworkX graph onto a skeletonized image. Then, color all terminal nodes a separate color so that they can be easily identified.

    Parameters:
        result_dict: a dictionary of processed attributes from a call to TGGLinesPlus()

        label: what digit, character, shape, etc. does result_dict represent

        node_size: an integer, how large you want the nodes to look on the graph (100 is good for small images, 30 better for bigger
                
        node_labels: boolean, whether to draw the node labels on the figure (does not look good for large graphs

        label_size: int, size of the font for labeling node numbers

        save_fig: boolean, whether to save the figure or not
        
        save_dir: string path for where to save the figure to

    Returns:
        None
    """
    skeleton = result_dict["skeleton"]
    simple_graph = result_dict["simple_graph"]
    search_by_node = result_dict["search_by_node"] 
    end_nodes = result_dict["end_nodes"]
    
    end_color = "red"
    other_color = "gray"

    node_options = {
        "node_size": node_size,
    }

    edge_options = {
        "width": 2.0, 
        "alpha": 0.5, 
    }

    label_options = {
        "font_size": label_size,
    }

    legend_options = {
        "xdata": [0],
        "ydata": [0],
        "marker": 'o',
        "markersize": 10,
        "linewidth": 0,
    }

    # get nodes that are junctions and non-junctions
    other_nodes = [node for node in simple_graph.nodes() if node not in end_nodes]

    # subset node location dict for each type of node
    end_nodes_dict = dict([item for item in search_by_node.items() if item[0] in end_nodes])
    other_nodes_dict = dict([item for item in search_by_node.items() if item[0] not in end_nodes])

    # reverse node locations for plotting in matplotlib
    node_locations_plotting = dict([(k, [v[1], v[0]]) for (k, v) in search_by_node.items()])
    end_locations_plotting = dict([(k, [v[1], v[0]]) for (k, v) in end_nodes_dict.items()])
    other_locations_plotting = dict([(k, [v[1], v[0]]) for (k, v) in other_nodes_dict.items()])

    # plot skeleton
    fig, ax = plt.subplots(figsize=(7, 7))
   
    ax.imshow(skeleton, cmap="gray")

    # plot nodes by type
    # non-end nodes
    nx.draw_networkx_nodes(simple_graph, pos=node_locations_plotting, nodelist=other_nodes, node_color=other_color, **node_options)
    nx.draw_networkx_edges(simple_graph, pos=node_locations_plotting, edgelist=nx.to_edgelist(simple_graph, other_nodes), edge_color=other_color, **edge_options)

    # end nodes
    if(len(end_nodes) != 0):
        nx.draw_networkx_nodes(simple_graph, pos=node_locations_plotting, nodelist=end_nodes, node_color=end_color, **node_options)
        nx.draw_networkx_edges(simple_graph, pos=node_locations_plotting, edgelist=nx.to_edgelist(simple_graph, end_nodes), edge_color=end_color, **edge_options)

    # add node labels
    if(node_labels):
        nx.draw_networkx_labels(simple_graph, pos=node_locations_plotting, **label_options)

    # add custom legend elements
    end_label = Line2D(color='red', markerfacecolor="red", label='TERMINAL', **legend_options)
    
    legend_elements = [end_label]
    ax.legend(handles=legend_elements, bbox_to_anchor=(0.95, 0.95))

    plt.axis("off")
    plt.tight_layout()

    if(save_fig is True):
        if not(os.path.isdir(save_dir)):
            os.mkdir(save_dir)
        # title, save figure
        if(label != ""):
            figtitle = f"junctions_{label}"
        else: 
            figtitle = "junctions"

        plt.savefig(save_dir+figtitle+".png", format='png', dpi=300, bbox_inches='tight')      
        # close figure to save on memory if saving many figures at once
        plt.close(fig)
    else:
        plt.show()


def plot_pathseg_endpoints(result_dict: dict, label: str = "", node_size: int = 100, node_labels: bool = True, label_size: int = 8, save_fig: bool = False, save_dir: str = "./") -> None:
    """
    Plot junctions and terminals (path segmentation start- and endpoints). The only difference between this method and plot_junctions()
    is that it also overlays terminal nodes onto the graph.

    Parameters:
        result_dict: a dictionary of processed attributes from a call to TGGLinesPlus()

        label: what digit, character, shape, etc. does result_dict represent

        node_size: an integer, how large you want the nodes to look on the graph (100 is good for small images, 30 better for bigger
                
        node_labels: boolean, whether to draw the node labels on the figure (does not look good for large graphs

        label_size: int, size of the font for labeling node numbers

        save_fig: boolean, whether to save the figure or not
        
        save_dir: string path for where to save the figure to

    Returns:
        None
    """  
    skeleton = result_dict["skeleton"]
    simple_graph = result_dict["simple_graph"]
    search_by_node = result_dict["search_by_node"]
    coordinates = result_dict["skeleton_coordinates"]
    path_seg_endpoints = result_dict["path_seg_endpoints"]
    
    junction_end_color = "red"
    other_color = "gray"

    node_options = {
        "node_size": node_size,
    }

    edge_options = {
        "width": 2.0, 
        "alpha": 0.5, 
    }

    label_options = {
        "font_size": label_size,
    }

    legend_options = {
        "xdata": [0],
        "ydata": [0],
        "marker": 'o',
        "markersize": 10,
        "linewidth": 0,
    }

    # get nodes that are junctions and non-junctions
    other_locations = [node for node in simple_graph.nodes() if node not in path_seg_endpoints]

    # subset node location dict for each type of node
    endpoint_nodes_dict = dict([item for item in search_by_node.items() if item[0] in path_seg_endpoints])
    other_nodes_dict = dict([item for item in search_by_node.items() if item[0] not in path_seg_endpoints])

    # reverse node locations for plotting in matplotlib
    node_locations_plotting = dict([(k, [v[1], v[0]]) for (k, v) in search_by_node.items()])
    enpoint_locations_plotting = dict([(k, [v[1], v[0]]) for (k, v) in endpoint_nodes_dict.items()])
    other_locations_plotting = dict([(k, [v[1], v[0]]) for (k, v) in other_nodes_dict.items()])

    # plot skeleton
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.imshow(skeleton, cmap="gray")

    # plot nodes by type
    # non-path seg endpoint nodes
    nx.draw_networkx_nodes(simple_graph, pos=node_locations_plotting, nodelist=other_locations, node_color=other_color, **node_options)
    nx.draw_networkx_edges(simple_graph, pos=node_locations_plotting, edgelist=nx.to_edgelist(simple_graph, other_locations), edge_color=other_color, **edge_options)

    # junction and end nodes
    if(len(path_seg_endpoints) != 0):
        nx.draw_networkx_nodes(simple_graph, pos=node_locations_plotting, nodelist=path_seg_endpoints, node_color=junction_end_color, **node_options)
        nx.draw_networkx_edges(simple_graph, pos=node_locations_plotting, edgelist=nx.to_edgelist(simple_graph, path_seg_endpoints), edge_color=junction_end_color, **edge_options)

    # add node labels
    if(node_labels):
        nx.draw_networkx_labels(simple_graph, pos=node_locations_plotting, **label_options)

    # add custom legend elements
    junction_end_label = Line2D(color='red', markerfacecolor="red", label='JUNCTION + END', **legend_options)
    legend_elements = [junction_end_label]
    ax.legend(handles=legend_elements, bbox_to_anchor=(0.95, 0.95))

    plt.axis("off")
    plt.tight_layout()

    if(save_fig is True):
        if not(os.path.isdir(save_dir)):
            os.mkdir(save_dir)
        # title, save figure
        if(label != ""):
            figtitle = f"junctions_and_terminals_{label}"
        else: 
            figtitle = "junctions_and_terminals"
        plt.savefig(save_dir+figtitle+".png", format='png', dpi=300, bbox_inches='tight')      
        # close figure to save on memory if saving many figures at once
        plt.close(fig)
    else:
        plt.show()


def plot_graph_paths(result_dict: dict, label: str = "", node_size: int = 100, plot_endpoints: bool = True, node_labels: bool = True, label_size: int = 8, show_legend: bool = False, save_fig: bool = False, save_dir: str = "./") -> None:
    """
    Plot paths in a NetworkX graph.

    Parameters:
        result_dict: a dictionary of processed attributes from a call to TGGLinesPlus()

        label: what digit, character, shape, filename, etc. does result_dict represent
        
        node_size: an integer, how large you want the nodes to look on the graph (100 is good for small images, 30 better for bigger
        
        plot_endpoints: boolean, whether to plot path_seg_endpoints on top of graph paths to better see end points and junctions
        
        node_labels: boolean, whether to draw the node labels on the figure (does not look good for large graphs
        
        label_size: int, size of the font for labeling node numbers

        show_legend: boolean, whether to plot the legend on the figure

        save_fig: boolean, whether to save the figure or not
        
        save_dir: string path for where to save the figure to

    Returns:
        None
    """    
    skeleton = result_dict["skeleton"]
    simple_graph = result_dict["simple_graph"]
    search_by_node = result_dict["search_by_node"]
    path_seg_endpoints = result_dict["path_seg_endpoints"]
    paths_list = result_dict["paths_list"]
    
    # create custom colormap here
    # should not contain colors close to cherry red
    custom_color_list = ["#ef6351", "#ff9f1c", "#ffd700", # yellows and oranges
                         "#90be6d",                       # greens
                        "#a2d2ff", "#0077b6", "#0d47a1",  # blues
                         "#ffc8dd", "#8e7dbe", "#6a4c93", # purples
                        "#bbbbbb", "#f7c59f"]             # grey and brown

    custom_cmap = ListedColormap(custom_color_list, name="Path Segmentation")
    num_colors = len(custom_cmap.colors)

    # choose a random color from our custom color map
    # we want to try and make it so that colors do not repeat on adjacent paths, if possible
    if(len(paths_list) > num_colors):
        random_idx = np.random.choice(num_colors, size=len(paths_list), replace=True)
    else:
        random_idx = np.random.choice(num_colors, size=len(paths_list), replace=False)

    junction_end_color = "red"
    
    node_options = {
        "node_size": node_size,
    }

    edge_options = {
        "width": 2.0, 
        "alpha": 0.5, 
    }

    label_options = {
        "font_size": label_size,
    }
    
    endpoint_options = {
        "node_size": node_size,
        "node_shape": "o", #options: ‘so^>v<dph8’.
        "linewidths": 2,
        "alpha": 1,
    }

    legend_options = {
        "xdata": [0],
        "ydata": [0],
        "marker": 'o',
        "markersize": 10,
        "linewidth": 0,
    }

    node_locations_plotting = dict([(k, [v[1], v[0]]) for (k, v) in search_by_node.items()])

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.imshow(skeleton, cmap='gray')
        
    # nodes in each path
    for i, path in enumerate(paths_list):
        color = custom_cmap(random_idx[i])
        color_hex = colors.to_hex(color, keep_alpha=True)
        nx.draw_networkx_nodes(simple_graph, pos=node_locations_plotting, nodelist=path, node_color=color_hex, **node_options)
        nx.draw_networkx_edges(simple_graph, pos=node_locations_plotting, edgelist=nx.to_edgelist(simple_graph, path), edge_color=color_hex, **edge_options)
    
    # junction and end nodes
    if(plot_endpoints):
        if(len(path_seg_endpoints) != 0):
            nx.draw_networkx_nodes(simple_graph, pos=node_locations_plotting, nodelist=path_seg_endpoints, node_color=junction_end_color, **node_options)
            nx.draw_networkx_edges(simple_graph, pos=node_locations_plotting, edgelist=nx.to_edgelist(simple_graph, path_seg_endpoints), edge_color=junction_end_color, **edge_options)

    #add node labels
    if(node_labels):
        nx.draw_networkx_labels(simple_graph, pos=node_locations_plotting, **label_options)
    
    # add custom legend elements
    if(show_legend):
        junction_end_label = Line2D(color='red', markerfacecolor="red", label='JUNCTION + END', **legend_options)
        legend_elements = [junction_end_label]
        ax.legend(handles=legend_elements, bbox_to_anchor=(0.95, 0.95))

    plt.axis("off")
    plt.margins(0)
    plt.tight_layout()

    if(save_fig is True):
        if not(os.path.isdir(save_dir)):
            os.mkdir(save_dir)
        # title, save figure
        if(label != ""):
            figtitle = f"path_segmentation_{label}"
        else: 
            figtitle = "path_segmentation"
        plt.savefig(save_dir+figtitle+".png", format='png', dpi=300, bbox_inches='tight')      
        # close figure to save on memory if saving many figures at once
        plt.close(fig)
    else:
        plt.show()
