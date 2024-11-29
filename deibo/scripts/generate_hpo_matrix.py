import sys
from pyhpo import Ontology
import pickle
import pandas as pd
import numpy as np
import math
from multiprocessing import Pool
from scipy.spatial import distance
from scipy.stats import norm, zscore
from scipy.integrate import quad

# Constant used to handle small variance scenarios
VAR_CONST = 0.01

def bhattacharya(p, q):
    """
    Compute the Bhattacharyya distance between two distributions.

    The Bhattacharyya distance is a measure of the overlap between two statistical samples. 
    This function assumes normal distributions for both inputs.

    Args:
        p (array-like): First distribution.
        q (array-like): Second distribution.

    Returns:
        float: Bhattacharyya distance between the two distributions.
    """
    vars = (p.var() / q.var()) + (q.var() / p.var()) + 2  # Variance term
    means = (p.mean() - q.mean())**2 / (p.var() + q.var())  # Mean difference term
    return 0.25 * np.log(0.25 * vars) + 0.25 * means

def hellinger(p, q):
    """
    Compute the Hellinger distance between two distributions.

    The Hellinger distance measures the similarity between two probability distributions. 
    This function assumes normal distributions for both inputs.

    Args:
        p (array-like): First distribution.
        q (array-like): Second distribution.

    Returns:
        float: Hellinger distance between the two distributions.
    """
    root = np.sqrt((2 * q.std() * p.std()) / (q.var() + p.var()))  # Root term
    ex = np.exp(-0.25 * ((p.mean() - q.mean())**2 / (p.var() + q.var())))  # Exponential term
    return 1 - (root * ex)

def kl(p, q):
    """
    Compute the Kullback-Leibler (KL) divergence between two distributions.

    The KL divergence measures how one probability distribution diverges from a second, expected probability distribution.

    Args:
        p (array-like): First distribution.
        q (array-like): Second distribution.

    Returns:
        float: Kullback-Leibler divergence between the two distributions.
    """
    std0 = p.std()
    std1 = q.std()
    var0 = p.var()
    var1 = q.var()
    mean0 = p.mean()
    mean1 = q.mean()
    return np.log(std1 / std0) + ((var0 + (mean0 - mean1)**2) / (2 * var1)) - 0.5

def jeffreys(p, q):
    """
    Compute Jeffrey's divergence (symmetric KL divergence) between two distributions.

    Jeffrey's divergence is the average of the KL divergence from p to q and from q to p.

    Args:
        p (array-like): First distribution.
        q (array-like): Second distribution.

    Returns:
        float: Jeffrey's divergence between the two distributions.
    """
    return (kl(p, q) + kl(q, p)) / 2

def js(p, q):
    """
    Compute the Jensen-Shannon (JS) divergence between two distributions.

    The JS divergence is a symmetrized version of the Kullback-Leibler divergence and is bounded.

    Args:
        p (array-like): First distribution.
        q (array-like): Second distribution.

    Returns:
        float: Jensen-Shannon divergence between the two distributions.
    """
    mu_1, sigma_1 = np.mean(p), np.std(p)
    mu_2, sigma_2 = np.mean(q), np.std(q)
    
    # Define the probability density functions for both distributions
    def pdf_1(x):
        return norm.pdf(x, mu_1, sigma_1)
    def pdf_2(x):
        return norm.pdf(x, mu_2, sigma_2)
    
    # Create a range of values to evaluate the PDFs
    x = np.linspace(min(mu_1 - 5 * sigma_1, mu_2 - 5 * sigma_2), 
                    max(mu_1 + 5 * sigma_1, mu_2 + 5 * sigma_2), 
                    num=1000)
    px = np.array([pdf_1(a) for a in x])  # PDF values for p
    qx = np.array([pdf_2(a) for a in x])  # PDF values for q
    
    # Compute the Jensen-Shannon divergence
    return distance.jensenshannon(px, qx)

def tvm(p, q):
    """
    Compute the Total Variation Metric (TVM) between two distributions.

    The Total Variation Metric is a measure of the distance between two probability distributions. 
    It is computed by integrating the absolute difference between the probability density functions.

    Args:
        p (array-like): First distribution.
        q (array-like): Second distribution.

    Returns:
        float: Total Variation Metric between the two distributions.
    """
    mu_1, sigma_1 = np.mean(p), np.std(p)
    mu_2, sigma_2 = np.mean(q), np.std(q)
    
    # Define the probability density functions for both distributions
    def pdf_1(x):
        return norm.pdf(x, mu_1, sigma_1)
    def pdf_2(x):
        return norm.pdf(x, mu_2, sigma_2)
    
    # Function to calculate the absolute difference between PDFs
    def tv_distance(x):
        return np.abs(pdf_1(x) - pdf_2(x))
    
    # Define the integration range
    lower_bound = min(mu_1 - 5 * sigma_1, mu_2 - 5 * sigma_2)
    upper_bound = max(mu_1 + 5 * sigma_1, mu_2 + 5 * sigma_2)
    
    # Compute the total variation distance
    tv_distance_value, _ = quad(tv_distance, lower_bound, upper_bound, limit=1000)
    tv_distance_value /= 2  # Normalize according to the definition
    return tv_distance_value

def compute_q(args):
    """
    Compute the distance (Q value) for a given feature and class.

    Args:
        args (tuple): A tuple containing the following elements:
            - i (int): Feature index.
            - j (int): Class index.
            - df (pd.DataFrame): DataFrame containing feature values.
            - classes (list): List of class names.
            - set_classes (dict): Dictionary mapping each class to associated terms.
            - dist (str): Distance metric to use ('kl', 'js', 'hell', etc.).

    Returns:
        tuple: A tuple containing:
            - i (int): Feature index.
            - j (int): Class index.
            - value (float): Calculated distance (Q value).
    """
    i, j, df, classes, set_classes, dist = args
    term = classes[j]
    d = set_classes[term]
    row = df.iloc[i]
    
    # Split the feature values into positive and negative sets based on class
    class_pos = row[[k in d for k in row.index]]
    class_neg = row[[k not in d for k in row.index]]
    
    # If the variance of the positive set is below the threshold, return 0
    if np.array(class_pos).var() <= VAR_CONST:
        return i, j, 0.0
    
    # Compute the specified distance metric
    if dist == 'jef':
        return i, j, jeffreys(class_pos, class_neg)
    elif dist == 'kl':
        return i, j, kl(class_pos, class_neg)
    elif dist == 'hell':
        return i, j, hellinger(class_pos, class_neg)
    elif dist == 'js':
        return i, j, js(class_pos, class_neg)
    elif dist == 'tvm':
        return i, j, tvm(class_pos, class_neg)
    elif dist == 'bha':
        return i, j, bhattacharya(class_pos, class_neg)

def generate_matrix(phen_map_path, phen_path, classes_path, set_classes_path, dist, out):
    """
    Generate a distance matrix (Q-matrix) between features and classes based on specified distance metrics.

    The function processes phenotype data, computes distances between features and classes, 
    and stores the resulting matrix in a pickle file.

    Args:
        phen_map_path (str): Path to the pickle file containing the phenotype map.
        phen_path (str): Path to the CSV file containing phenotypic data.
        classes_path (str): Path to the pickle file containing the list of classes.
        set_classes_path (str): Path to the pickle file containing the class-term mapping.
        dist (str): Distance metric to use ('kl', 'js', 'hell', etc.).
        out (str): Path to save the output Q-matrix.
    """
    # Load the ontology
    _ = Ontology(data_folder='data/hpo/hpo_ontology')
    
    # Load and invert the phenotype map
    with open(phen_map_path, 'rb') as file:
        phen_map = pickle.load(file)
    phen_map = {v: k for k, v in phen_map.items()}
    
    # Map HPO IDs to term names
    hpo_map = {v: Ontology.get_hpo_object(v).name for v in phen_map.values()}
    
    # Prepare HPO DataFrame and filter out obsolete terms
    df_hpo = Ontology().to_dataframe()
    df_hpo = df_hpo[~df_hpo['name'].str.contains('obsolete')]  # Remove obsolete terms
    my_list = list(set(df_hpo.index.to_list()).intersection(set(phen_map.values())))
    df_hpo = df_hpo[df_hpo.index.isin(my_list)]
    df_hpo['children'] = df_hpo['children'].apply(lambda x: x.split('|') if x != '' else [])
    
    # Load phenotype data and standardize
    df = pd.read_csv(phen_path).transpose().rename(columns=phen_map).rename(columns=hpo_map)
    df = df.apply(lambda row: zscore(row), axis=1)
    
    # Load class data
    with open(classes_path, 'rb') as file:
        classes = pickle.load(file)
    with open(set_classes_path, 'rb') as file:
        set_classes = pickle.load(file)
    
    # Initialize the Q-matrix (rows for features, columns for classes)
    Q = np.zeros((768, len(classes)))  # Adjust this size based on your dataset
    
    # Prepare arguments for parallel computation
    args_list = [(i, j, df, classes, set_classes, dist) for i in range(768) for j in range(len(classes))]
    
    # Perform parallel computation using multiprocessing
    with Pool() as pool:
        results = pool.map(compute_q, args_list)
    
    # Populate the Q-matrix with the computed values
    for i, j, value in results:
        Q[i, j] = value
    
    # Save the Q-matrix to a pickle file
    with open(out, 'wb') as f:
        pickle.dump(Q, f)
        
# Main script execution
if __name__ == '__main__':
    # Call the function with parameters provided by Snakemake
    generate_matrix(snakemake.input[0], snakemake.input[1], 
                    snakemake.input[2], snakemake.input[3], 
                    snakemake.params['dist'], snakemake.output[0])