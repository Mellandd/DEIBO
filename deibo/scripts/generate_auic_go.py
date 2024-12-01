import sys
from goatools.base import get_godag
from goatools.semantic import deepest_common_ancestor, min_branch_length
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

def calcular_distancia(nombre1, nombre2):
    """
    Calculates the semantic distance between two GO terms using 
    their deepest common ancestor (DCA) and branch lengths.
    
    Args:
        nombre1 (str): The name of the first GO term.
        nombre2 (str): The name of the second GO term.
    
    Returns:
        float: The computed semantic distance between the two GO terms.
    """
    term_1 = go[nombre1]  # Retrieve term 1 from GO DAG
    term_2 = go[nombre2]  # Retrieve term 2 from GO DAG
    ca = deepest_common_ancestor([nombre1, nombre2], go)  # Find DCA
    c1 = min_branch_length(nombre1, ca, go, 1)  # Distance from term 1 to DCA
    c2 = min_branch_length(nombre2, ca, go, 1)  # Distance from term 2 to DCA
    h = go[ca].level + 1  # Depth of DCA in GO hierarchy
    return (2 * h) / (c1 + c2 + (2 * h))  # Semantic distance formula

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
    term = classes[df_q.iloc[dim].argmax()]  # Class with highest score in the dimension
    terms = set_classes[term]  # Retrieve associated GO terms for the class
    nj = len(terms)  # Number of terms in the class

    # Positive side computation
    v_i = df.iloc[dim].argsort()[:(lamb * nj)]
    scoresplus = []
    for t in v_i:
        if go[ids[t]].id in terms:  # Check if term belongs to the class
            scoresplus.append(1)
        else:
            scoresplus.append(0)

    # Negative side computation
    v_i = df.iloc[dim].argsort()[::-1][:(lamb * nj)]
    scoresneg = []
    for t in v_i:
        if go[ids[t]].id in terms:
            scoresneg.append(1)
        else:
            scoresneg.append(0)

    # Calculate scores for different λ values
    scores = []
    for i in range(1, lamb + 1):
        score_p = np.array(scoresplus[:(i * nj)])
        scoreplus = np.sum(score_p)  # Sum for positive side
        score_n = np.array(scoresneg[:(i * nj)])
        scoreneg = np.sum(score_n)  # Sum for negative side
        scores.append((max(scoreplus, scoreneg) * 100) / nj)  # Normalize score
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
    for i in range(768):  # Iterate over all dimensions
        scores.append(inter_score_ont(i, lamb, df_q))  # Compute scores per dimension
    scores = np.array(scores)
    return np.average(scores, axis=0)  # Average scores across dimensions

def generate_auic(gobp_embs_path, classes_path, set_classes_path,
                  q_jef, q_js, q_k, q_h, q_tvm, q_bha, out_path, sub):
    """
    Generates and visualizes the Area Under the Interpretability Curve (AUIC) 
    for various divergence measures.
    
    Args:
        gobp_embs_path (str): Path to the GO-BP embeddings file.
        classes_path (str): Path to the classes file.
        set_classes_path (str): Path to the set of classes file.
        q_jef (str): Path to Jeffreys divergence matrix.
        q_js (str): Path to Jensen-Shannon divergence matrix.
        q_k (str): Path to Kullback-Leibler divergence matrix.
        q_h (str): Path to Hellinger divergence matrix.
        q_tvm (str): Path to Total Variation distance matrix.
        q_bha (str): Path to Bhattacharyya distance matrix.
        out_path (str): Path to save the resulting plot.
        sub (str): Sub-ontology to use (e.g., biological process, molecular function).
    """
    np.random.seed(0)
    global go
    go = get_godag('go-basic.obo', optional_attrs=['relationship', 'def'])  # Load GO DAG
    global ids
    ids = [go[i].id for i in go if go[i].namespace == sub]  # Filter terms by sub-ontology
    names = [go[i].name for i in go if go[i].namespace == sub]
    dict_names = dict([(i, j) for i, j in zip(names, ids)])

    global df
    df = pd.read_csv(gobp_embs_path)
    df['name'] = ids
    df = df.set_index(['name']).drop_duplicates().transpose()  # Prepare embeddings

    # Load divergence matrices
    with open(q_h, 'rb') as file:
        Q_h = pickle.load(file)
    df_Qh = pd.DataFrame(Q_h)

    with open(q_jef, 'rb') as file:
        Q_jef = pickle.load(file)
    df_QJef = pd.DataFrame(Q_jef)

    with open(q_js, 'rb') as file:
        Q_js = pickle.load(file)
    df_QJs = pd.DataFrame(Q_js)

    with open(q_k, 'rb') as file:
        Q_kl = pickle.load(file)
    df_Qkl = pd.DataFrame(Q_kl)

    with open(q_tvm, 'rb') as file:
        Q_tvm = pickle.load(file)
    df_Qtvm = pd.DataFrame(Q_tvm)

    with open(q_bha, 'rb') as file:
        Q_bha = pickle.load(file)
    df_Qbha = pd.DataFrame(Q_bha)

    # Load class mappings
    with open(classes_path, 'rb') as file:
        global classes
        classes = pickle.load(file)

    with open(set_classes_path, 'rb') as file:
        global set_classes
        set_classes = pickle.load(file)

    # Generate random comparison matrices
    Q_rands = [generate_random_matrix() for _ in range(10)]

    # Compute interpretability scores for all measures
    scores_jef = inter_score_global_ont(100, df_QJef)
    scores_h = inter_score_global_ont(100, df_Qh)
    scores_k = inter_score_global_ont(100, df_Qkl)
    scores_js = inter_score_global_ont(100, df_QJs)
    scores_tvm = inter_score_global_ont(100, df_Qtvm)
    scores_bha = inter_score_global_ont(100, df_Qbha)
    scores_rands = [inter_score_global_ont(100, pd.DataFrame(Q)) for Q in Q_rands]

    # Compute AUIC values
    areas = [
        (np.trapz(score, range(1, 101)) / (100 * 100), score, label)
        for score, label in zip(
            [scores_h, scores_jef, scores_k, scores_js, scores_tvm, scores_bha],
            ["Hellinger", "Jeffreys", "Kullback-Leibler", "Jensen-Shannon", "TV Distance", "Bhattacharya"]
        )
    ]

    # Add random scores with error bars
    mean_scores = np.average(scores_rands, axis=0)
    yerror = np.std(scores_rands, axis=0)

    # Plot interpretability curves
    plt.figure(figsize=(8, 6), dpi=300)
    for area, score, label in sorted(areas, key=lambda x: x[0], reverse=True):
        plt.plot(range(1, 101), score, label=f'{label}, AUIC={np.round(area, 2)}')

    plt.fill_between(range(1, 101), mean_scores - yerror, mean_scores + yerror, color='C0', alpha=0.3, label='Random')
    plt.xlabel(r'$\lambda$', fontsize=14)
    plt.ylabel('Interpretability Score (IS)', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()  # Adjust layout to avoid clipping

    # Save and display the plot
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.show()