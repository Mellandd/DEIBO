import sys
from goatools.base import get_godag  # Load Gene Ontology DAG structure
from goatools.semantic import deepest_common_ancestor, min_branch_length  # Functions for semantic distance
import pickle  # For serializing and deserializing data structures
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
from matplotlib import pyplot as plt  # For plotting
from multiprocessing import Pool  # To enable parallel processing

def generate_random_matrix():
    """
    Generate a random matrix of dimensions 768 x len(classes).
    This represents random embeddings for interpretability comparison.
    """
    Q = np.random.rand(768, len(classes))
    return Q

def calcular_distancia(nombre1, nombre2):
    """
    Calculate semantic distance between two Gene Ontology terms using their 
    deepest common ancestor and branch lengths.

    Args:
        nombre1 (str): First GO term ID.
        nombre2 (str): Second GO term ID.

    Returns:
        float: Normalized distance metric.
    """
    term_1 = go[nombre1]  # Retrieve first GO term details
    term_2 = go[nombre2]  # Retrieve second GO term details
    ca = deepest_common_ancestor([nombre1, nombre2], go)  # Find the deepest common ancestor
    c1 = min_branch_length(nombre1, ca, go, 1)  # Distance from term_1 to ancestor
    c2 = min_branch_length(nombre2, ca, go, 1)  # Distance from term_2 to ancestor
    h = go[ca].level + 1  # Height of the ancestor level, +1 ensures no zero division
    return (2 * h) / (c1 + c2 + (2 * h))  # Normalized distance computation

def inter_score_ont(dim, lamb, df_q, df):
    """
    Compute interpretability score for a single dimension of the embedding.

    Args:
        dim (int): Dimension index to evaluate.
        lamb (int): Scaling factor for interpretability.
        df_q (pd.DataFrame): Query embedding matrix.
        df (pd.DataFrame): Embedding matrix.

    Returns:
        list: Interpretability scores at different lambda levels.
    """
    term = classes[df_q.iloc[dim].argmax()]  # Identify the most relevant class
    terms = set_classes[term]  # Fetch GO terms associated with the class
    nj = len(terms)  # Number of terms in the class

    # Positive side: Sort scores and evaluate for top lambda * nj elements
    v_i = df.iloc[dim].argsort()[:(lamb * nj)]
    scoresplus = []
    for t in v_i:
        if go[ids[t]].id in terms:
            scoresplus.append(1)  # Exact match
        else:
            scoresplus.append(0)  # Non-match

    # Negative side: Evaluate for lowest lambda * nj elements
    v_i = df.iloc[dim].argsort()[::-1][:(lamb * nj):]
    scoresneg = []
    for t in v_i:
        if go[ids[t]].id in terms:
            scoresneg.append(1)
        else:
            scoresneg.append(0)

    # Aggregate scores across multiple thresholds
    scores = []
    for i in range(1, lamb + 1):
        score_p = np.array(scoresplus[:(i * nj)])  # Positive matches
        scoreplus = np.sum(score_p)
        score_n = np.array(scoresneg[:(i * nj)])  # Negative matches
        scoreneg = np.sum(score_n)
        scores.append((max(scoreplus, scoreneg) * 100) / nj)  # Normalize score
    return scores

def inter_score_global_ont(lamb, df_q, df):
    """
    Compute global interpretability scores across all embedding dimensions.

    Args:
        lamb (int): Scaling factor for interpretability.
        df_q (pd.DataFrame): Query embedding matrix.
        df (pd.DataFrame): Embedding matrix.

    Returns:
        np.array: Averaged interpretability scores for all dimensions.
    """
    scores = []
    for i in range(768):  # Iterate over all dimensions
        scores.append(inter_score_ont(i, lamb, df_q, df))  # Compute scores for each dimension
    scores = np.array(scores)  # Convert to NumPy array for aggregation
    return np.average(scores, axis=0)  # Average scores across dimensions

def generate_auic(go_embs_path, go_embs_path_mpnet, go_embs_path_lord, go_embs_path_edu, 
                  classes_path, set_classes_path, q_b, q_b_mpnet, q_b_biolord, q_b_edu, out_path, sub):
    """
    Generate AUIC (Area Under the Interpretability Curve) for various embedding techniques.

    Args:
        go_embs_path (str): Path to GO embeddings (BioBERT).
        go_embs_path_mpnet (str): Path to MPNet embeddings.
        go_embs_path_lord (str): Path to BioLORD embeddings.
        go_embs_path_edu (str): Path to Education embeddings.
        classes_path (str): Path to class information.
        set_classes_path (str): Path to GO term sets for each class.
        q_b, q_b_mpnet, q_b_biolord, q_b_edu (str): Query matrices for respective embeddings.
        out_path (str): Output path for saving the plot.
        sub (str): Sub-ontology to focus on (e.g., biological process).
    """
    np.random.seed(0)  # Fix random seed for reproducibility

    # Load GO DAG and IDs specific to the given sub-ontology
    global go
    go = get_godag('go-basic.obo', optional_attrs=['relationship', 'def'])
    global ids
    ids = [go[i].id for i in go if go[i].namespace == sub]  # Filter IDs by namespace
    names = [go[i].name for i in go if go[i].namespace == sub]  # Filter names by namespace
    dict_names = dict([(i, j) for i, j in zip(names, ids)])  # Map names to IDs

    # Load and preprocess embedding matrices for all models
    global df
    df = pd.read_csv(go_embs_path)
    df['name'] = ids
    df = df.set_index(['name'])
    df = df.drop_duplicates().transpose()  # Transpose for correct orientation

    # Similar processing for other embeddings (MPNet, BioLORD, Education)
    df_mpnet = pd.read_csv(go_embs_path_mpnet)
    df_mpnet['name'] = ids
    df_mpnet = df_mpnet.set_index(['name']).drop_duplicates().transpose()

    df_biolord = pd.read_csv(go_embs_path_lord)
    df_biolord['name'] = ids
    df_biolord = df_biolord.set_index(['name']).drop_duplicates().transpose()

    df_edu = pd.read_csv(go_embs_path_edu)
    df_edu['name'] = ids
    df_edu = df_edu.set_index(['name']).drop_duplicates().transpose()

    # Load query matrices for each embedding type
    with open(q_b, 'rb') as file:
        Q_bat = pickle.load(file)
    df_QB = pd.DataFrame(Q_bat)

    # Repeat for other query matrices
    with open(q_b_mpnet, 'rb') as file:
        Q_bat_mpnet = pickle.load(file)
    df_QBmpnet = pd.DataFrame(Q_bat_mpnet)

    with open(q_b_biolord, 'rb') as file:
        Q_bat_bio = pickle.load(file)
    df_QBbio = pd.DataFrame(Q_bat_bio)

    with open(q_b_edu, 'rb') as file:
        Q_bat_edu = pickle.load(file)
    df_QBedu = pd.DataFrame(Q_bat_edu)

    # Load class and term set data
    with open(classes_path, 'rb') as file:
        global classes
        classes = pickle.load(file)

    with open(set_classes_path, 'rb') as file:
        global set_classes
        set_classes = pickle.load(file)

    # Compute interpretability scores for all embeddings
    scores_b = inter_score_global_ont(100, df_QB, df)
    scores_bm = inter_score_global_ont(100, df_QBmpnet, df_mpnet)
    scores_bio = inter_score_global_ont(100, df_QBbio, df_biolord)
    scores_edu = inter_score_global_ont(100, df_QBedu, df_edu)

    # Compute AUIC for each method
    area1 = np.trapz(scores_bm, range(1, 101)) / (100 * 100)
    area2 = np.trapz(scores_b, range(1, 101)) / (100 * 100)
    area3 = np.trapz(scores_bio, range(1, 101)) / (100 * 100)
    area4 = np.trapz(scores_edu, range(1, 101)) / (100 * 100)

    # Package results with labels for visualization
    areas = [
        (area1, scores_bm, 'all-mpnet-base-v2, AUIC= ' + str(np.round(area1, 2)), 'C1', '-'),
        (area2, scores_b, 'BioBERT, AUIC= ' + str(np.round(area2, 2)), 'C2', '--'),
        (area3, scores_bio, 'BioLORD-2023, AUIC= ' + str(np.round(area3, 2)), 'C3', '-.'),
        (area4, scores_edu, 'BERT Education, AUIC= ' + str(np.round(area4, 2)), 'C4', ':'),
    ]
        # Sort areas in descending order for visualization
    areas_sorted = sorted(areas, key=lambda x: x[0], reverse=True)

    # Create the plot
    plt.figure(figsize=(8, 6), dpi=300)

    # Plot interpretability curves
    for area, score, label, color, linestyle in areas_sorted:
        plt.plot(range(1, 101), score, label=label, color=color, linestyle=linestyle, linewidth=2)

    # Configure axis labels
    plt.xlabel(r'$\lambda$', fontsize=14)
    plt.ylabel('Interpretability Score (IS)', fontsize=14)
    plt.ylim(0, 100)
    # Add legend sorted by AUIC
    plt.legend(loc='lower right', fontsize=12)
    # Add grid for better readability
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # Optimize layout for saving
    plt.tight_layout()
    # Save plot to the specified output path
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    # Display the plot
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
                  snakemake.output[0],
                  snakemake.params['sub'])