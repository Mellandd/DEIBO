# Importing necessary libraries
import sys  # For system-specific parameters and functions
from pyhpo import Ontology  # To interact with the Human Phenotype Ontology (HPO)
import pickle  # For serializing and deserializing Python objects
import pandas as pd  # For handling tabular data
import numpy as np  # For numerical computations
from multiprocessing import Pool  # For parallel processing

# Function to calculate the size of subgraphs for nodes in a graph
def calculate_subgraph_size(df):
    """
    Calculate the size of subgraphs for nodes in a graph and return their sizes.

    Parameters:
    df (pd.DataFrame): DataFrame with nodes and their associated children.

    Returns:
    dict: A dictionary where keys are nodes and values are their subgraph sizes.
    """
    # Initialize a dictionary to store subgraph sizes for each node
    subgraph_sizes = {node: 0 for node in df.index}

    stack = []  # Stack for depth-first traversal
    visited = set()  # Set to track visited nodes

    # Iterate through all nodes in the DataFrame
    for node in df.index:
        if node not in visited:  # Process unvisited nodes
            stack.append((node, False))  # Push node onto the stack
            visited.add(node)

            while stack:
                current, processed = stack.pop()  # Process nodes in the stack
                if not processed:  # If the node is not yet processed
                    stack.append((current, True))  # Mark the node as processed
                    # Add children to the stack for further exploration
                    for child in df.at[current, 'children']:
                        if child not in visited and child in df.index:
                            stack.append((child, False))
                            visited.add(child)
                else:
                    # Calculate subgraph size for the current node
                    subgraph_size = 1  # Count the current node
                    for child in df.at[current, 'children']:
                        if child in df.index:
                            subgraph_size += subgraph_sizes[child]
                    subgraph_sizes[current] = subgraph_size

    return subgraph_sizes

# Function to generate classes from phenotype data and the HPO
def generate_classes(phen_map_path, phen_path, classes_path, set_classes_path):
    """
    Generate classes based on phenotype data and HPO, and save results.

    Parameters:
    phen_map_path (str): Path to a pickle file mapping phenotypes to HPO terms.
    phen_path (str): Path to a CSV file with phenotype data.
    classes_path (str): Path to save the filtered HPO classes.
    set_classes_path (str): Path to save the set of HPO class subgraphs.
    """
    # Initialize the HPO Ontology
    _ = Ontology(data_folder='data/hpo/hpo_ontology')

    # Load the phenotype-to-HPO mapping from a pickle file
    with open(phen_map_path, 'rb') as file:
        phen_map = pickle.load(file)

    # Reverse the phenotype-to-HPO mapping
    phen_map = {v: k for k, v in phen_map.items()}

    # Create a mapping from HPO IDs to their names
    hpo_map = {v: Ontology.get_hpo_object(v).name for v in phen_map.values()}

    # Load the HPO ontology as a DataFrame
    df_hpo = Ontology().to_dataframe()

    # Filter out obsolete HPO terms
    df_hpo = df_hpo[~df_hpo['name'].str.contains('obsolete')]

    # Filter to include only terms in the intersection of the HPO DataFrame and phenotype mapping
    my_list = list(set(df_hpo.index.to_list()).intersection(set(phen_map.values())))
    df_hpo = df_hpo[df_hpo.index.isin(my_list)]

    # Parse children terms for each node
    df_hpo['children'] = df_hpo['children'].apply(lambda x: x.split('|') if (x != '') else [])

    # Load the phenotype data and map column names to HPO terms
    df = pd.read_csv(phen_path).transpose().rename(columns=phen_map).rename(columns=hpo_map)

    # Calculate subgraph sizes for the HPO ontology
    subgraph_sizes = calculate_subgraph_size(df_hpo)

    # Add subgraph size information to the HPO DataFrame
    df_hpo['subgraph_size'] = df_hpo.index.map(subgraph_sizes.get)

    # Filter classes based on a subgraph size threshold
    classes = df_hpo[df_hpo['subgraph_size'] >= 10].index.to_list()

    # Remove the root node ('All') from the classes
    classes.remove(Ontology.get_hpo_object('All').id)

    # Initialize a dictionary to store subgraphs for each class
    set_classes = {}
    terms = df_hpo.index.to_list()

    # Populate the set of subgraphs for each class
    for term in classes:
        if Ontology.get_hpo_object(term).name != 'All':  # Skip the root node
            term_h = Ontology.get_hpo_object(term)
            arr = []
            for t in terms:
                if t == term or Ontology.get_hpo_object(t).child_of(term_h):
                    arr.append(Ontology.get_hpo_object(t).name)
            set_classes[term] = arr

    # Save the filtered classes and their subgraphs to files
    with open(classes_path, 'wb') as file:
        pickle.dump(classes, file)

    with open(set_classes_path, 'wb') as file:
        pickle.dump(set_classes, file)

# Main script execution
if __name__ == '__main__':
    # Call the function with parameters provided by Snakemake
    generate_classes(snakemake.input[0],
                     snakemake.input[1],
                     snakemake.output[0],
                     snakemake.output[1])