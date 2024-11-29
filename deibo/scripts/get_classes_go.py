# Importing necessary libraries
import sys  # For system-specific parameters and functions
import pickle  # For serializing and deserializing Python objects
import pandas as pd  # For handling tabular data
import numpy as np  # For numerical computations
from multiprocessing import Pool  # For parallel processing
import os  # For interacting with the operating system
from goatools.base import get_godag  # To retrieve the Gene Ontology (GO) DAG
from goatools.semantic import TermCounts, get_info_content  # For GO term semantic operations
from goatools.associations import dnld_assc  # For downloading GO associations
from goatools.gosubdag.gosubdag import GoSubDag  # For working with sub-DAGs of GO terms

# Function to calculate the size of the subgraph for each node in a graph
def calculate_subgraph_size(df):
    """
    Calculate the size of subgraphs for nodes in a graph and return their details.

    Parameters:
    df (pd.DataFrame): DataFrame with nodes and their associated children.

    Returns:
    tuple: A dictionary of subgraph sizes and a dictionary of subgraph contents for each node.
    """
    # Initialize dictionaries to store subgraph sizes and subgraph node lists
    subgraph_sizes = {node: 0 for node in df.index}
    subgraphs = {node: [] for node in df.index}

    stack = []  # Stack for depth-first traversal
    visited = set()  # Set to track visited nodes

    # Iterate through all nodes in the DataFrame
    for node in df.index:
        if node not in visited:  # Process unvisited nodes
            stack.append((node, False))  # Push node onto the stack
            visited.add(node)

            while stack:
                current, processed = stack.pop()  # Process nodes in stack
                if not processed:  # If the node is not yet processed
                    stack.append((current, True))  # Mark the node as processed
                    # Add children to the stack for further exploration
                    if len(df.at[current, 'children']) > 0:
                        for child in df.at[current, 'children']:
                            if child not in visited and child in df.index:
                                stack.append((child, False))
                                visited.add(child)
                else:
                    # Calculate subgraph size and contents for the current node
                    subgraph_size = 1
                    subgraph = [current]
                    for child in df.at[current, 'children']:
                        if child in df.index:
                            subgraph_size += subgraph_sizes[child]
                            subgraph = subgraph + subgraphs[child]
                    subgraph_sizes[current] = subgraph_size
                    subgraphs[current] = subgraph

    return subgraph_sizes, subgraphs

# Function to extract GO classes based on a subgraph size threshold
def get_classes_go(go_path, classes_path, set_classes_path, subgo):
    """
    Extract GO classes with significant subgraph sizes and save the results.

    Parameters:
    go_path (str): Path to the input GO term data file.
    classes_path (str): Path to save the filtered GO classes.
    set_classes_path (str): Path to save the set of GO class subgraphs.
    subgo (str): GO namespace to filter by (e.g., 'biological_process').
    """
    # Load the Gene Ontology DAG
    go = get_godag('go-basic.obo', optional_attrs=['relationship','def'])

    # Define the path to the GAF (Gene Association File) (not used in the final implementation)
    fin_gaf = os.path.join(os.getcwd(), "tair.gaf")

    # Extract GO term IDs and names for the specified namespace
    ids = [go[i].id for i in go if go[i].namespace == subgo]
    names = [go[i].name for i in go if go[i].namespace == subgo]
    dict_names = dict([(i, j) for i, j in zip(names, ids)])

    # Load input data into a DataFrame
    df = pd.read_csv(go_path)
    df['name'] = ids
    df = df.set_index(['name'])
    df = df.drop_duplicates().transpose()

    # Prepare a DataFrame containing GO terms and their children
    df_go = []
    for g in go:
        if go[g].id in ids:
            df_go.append([go[g].name, go[g].id])
    df_go = pd.DataFrame(df_go, columns=['name', 'id'])
    df_go = df_go.drop_duplicates()
    df_go['children'] = df_go['id'].apply(lambda x: [go[y.id].id for y in go[x].children])
    df_go = df_go.set_index('id')
    df_go['children'] = df_go['children'].apply(lambda x: x if x != '[]' else [])

    # Calculate subgraph sizes and contents
    subgraph_sizes, subgraphs = calculate_subgraph_size(df_go)

    # Add subgraph size information to the DataFrame
    df_go['subgraph_size'] = df_go.index.map(subgraph_sizes.get)

    # Filter classes based on subgraph size threshold
    classes = df_go[df_go['subgraph_size'] >= 10].index.to_list()

    # Remove root-level terms for the specified namespace
    if subgo == 'molecular_function':
        classes.remove('GO:0003674')  # Root for molecular_function
    elif subgo == 'biological_process':
        classes.remove('GO:0008150')  # Root for biological_process
    else:
        classes.remove('GO:0005575')  # Root for cellular_component

    # Generate the set of subgraphs for the filtered classes
    set_classes = {c: subgraphs[c] for c in classes}

    # Save the filtered classes and their subgraphs to files
    with open(classes_path, 'wb') as file:
        pickle.dump(classes, file)
        
    with open(set_classes_path, 'wb') as file:
        pickle.dump(set_classes, file)

# Main script execution
if __name__ == '__main__':
    # Call the function with parameters provided by Snakemake
    get_classes_go(snakemake.input[0], snakemake.output[0],
                   snakemake.output[1],
                   snakemake.params['sub'])