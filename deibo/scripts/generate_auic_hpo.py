import sys
from pyhpo import Ontology
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from multiprocessing import Pool


def generate_random_matrix():
    """
    Generates a random matrix with values between 0 and 1.
    
    Returns:
        np.ndarray: A random matrix of shape (768, len(classes)).
    """
    Q = np.random.rand(768, len(classes))  # Create a matrix with random values
    return Q

def inter_score_ont(dim, lamb, df_q):
    """
    Computes interpretability scores for a specific dimension.
    
    Args:
        dim (int): The dimension index.
        lamb (int): The scaling parameter λ.
        df_q (pd.DataFrame): Dataframe containing scores for terms.
        
    Returns:
        list: A list of interpretability scores.
    """
    term = classes[df_q.iloc[dim].argmax()]  # Get the class with the highest score
    terms = set_classes[term]  # Retrieve associated terms
    nj = len(terms)  # Number of terms for the class
    
    # Identify top λ*nj terms for positive side
    v_i = df.iloc[dim].argsort()[:(lamb * nj)]
    scoresplus = []
    for t in v_i:
        if hpo_map[phen_map[t]] in terms:  # Check if term belongs to the class
            scoresplus.append(1)
        else:
            scoresplus.append(0)
    
    # Identify top λ*nj terms for negative side
    v_i = df.iloc[dim].argsort()[::-1][:(lamb * nj)]
    scoresneg = []
    for t in v_i:
        if hpo_map[phen_map[t]] in terms:
            scoresneg.append(1)
        else:
            scoresneg.append(0)
    
    # Compute scores for varying λ values
    scores = []
    for i in range(1, lamb + 1):
        score_p = np.array(scoresplus[:(i * nj)])
        scoreplus = np.sum(score_p)  # Compute score for positive side
        score_n = np.array(scoresneg[:(i * nj)])
        scoreneg = np.sum(score_n)  # Compute score for negative side
        scores.append((max(scoreplus, scoreneg) * 100) / nj)
    
    return scores


def inter_score_global_ont(lamb, df_q):
    """
    Computes the average interpretability score across all dimensions.
    
    Args:
        lamb (int): The scaling parameter λ.
        df_q (pd.DataFrame): Dataframe containing scores for terms.
        
    Returns:
        np.ndarray: Average interpretability scores across dimensions.
    """
    scores = []
    for i in range(768):
        scores.append(inter_score_ont(i, lamb, df_q))  # Compute scores per dimension
    scores = np.array(scores)
    return np.average(scores, axis=0)  # Average across dimensions


def generate_auic(phen_map_path, phen_path, classes_path, set_classes_path,
                  q_jef, q_js, q_k, q_h, q_tvm, q_bha, out_path):
    """
    Generates and visualizes the Area Under the Interpretability Curve (AUIC) 
    for various divergence measures.
    
    Args:
        phen_map_path (str): Path to the phenotype mapping file.
        phen_path (str): Path to the phenotype data file.
        classes_path (str): Path to the classes file.
        set_classes_path (str): Path to the set of classes file.
        q_jef (str): Path to Jeffreys divergence matrix.
        q_js (str): Path to Jensen-Shannon divergence matrix.
        q_k (str): Path to Kullback-Leibler divergence matrix.
        q_h (str): Path to Hellinger divergence matrix.
        q_tvm (str): Path to Total Variation distance matrix.
        q_bha (str): Path to Bhattacharyya distance matrix.
        out_path (str): Path to save the resulting plot.
    """
    # Load ontology and mappings
    np.random.seed(0)
    _ = Ontology(data_folder='data/hpo/hpo_ontology')
    with open(phen_map_path, 'rb') as file:
        global phen_map
        phen_map = pickle.load(file)
    phen_map = {v: k for k, v in phen_map.items()}  # Reverse mapping
    
    global hpo_map
    hpo_map = {v: Ontology.get_hpo_object(v).name for v in phen_map.values()}
    global df
    df = pd.read_csv(phen_path).transpose().rename(columns=phen_map).rename(columns=hpo_map)
    
    # Load divergence matrices
    matrices = {}
    for name, path in zip(['Q_h', 'Q_jef', 'Q_js', 'Q_k', 'Q_tvm', 'Q_bha'], 
                          [q_h, q_jef, q_js, q_k, q_tvm, q_bha]):
        with open(path, 'rb') as file:
            matrices[name] = pd.DataFrame(pickle.load(file))
    
    # Load class information
    with open(classes_path, 'rb') as file:
        global classes
        classes = pickle.load(file)
    
    with open(set_classes_path, 'rb') as file:
        global set_classes
        set_classes = pickle.load(file)
    
    # Generate random comparison matrices
    Q_rands = [generate_random_matrix() for _ in range(10)]
    
    # Compute interpretability scores
    scores = {key: inter_score_global_ont(100, mat) for key, mat in matrices.items()}
    scores['Random'] = [inter_score_global_ont(100, pd.DataFrame(Q)) for Q in Q_rands]
    
    # Compute AUIC for all measures
    areas = {key: np.trapz(score, range(1, 101)) / (100 * 100) for key, score in scores.items()}
    
    # Plot the interpretability curves
    plt.figure(figsize=(8, 6), dpi=300)
    for key, score in scores.items():
        plt.plot(range(1, 101), score, label=f'{key}, AUIC= {np.round(areas[key], 2)}')
    
    plt.xlabel(r'$\lambda$', fontsize=14)
    plt.ylabel('Interpretability Score (IS)', fontsize=14)
    plt.ylim(0, 100)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == '__main__':
    generate_auic(snakemake.input[0],
                  snakemake.input[1],
                  snakemake.input[2],
                  snakemake.input[3],
                  snakemake.input[4],
                  snakemake.input[5],
                  snakemake.input[6],
                  snakemake.input[7],
                  snakemake.input[8],
                  snakemake.input[9],
                  snakemake.output[0])