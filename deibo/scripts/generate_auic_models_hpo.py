from pyhpo import Ontology
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from multiprocessing import Pool

def generate_random_matrix():
    """
    Generates a random matrix of dimensions 768 x len(classes).

    Returns:
        np.ndarray: A random matrix with values sampled uniformly from [0, 1).
    """
    Q = np.random.rand(768, len(classes))
    return Q

def permute_dataframe(df, num_permutations=1):
    """
    Permutes the values within each column of a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        num_permutations (int): The number of permuted DataFrames to generate.

    Returns:
        list of pd.DataFrame: A list containing permuted DataFrames.
    """
    permuted_dfs = []
    for _ in range(num_permutations):
        permuted_df = df.copy()
        for column in permuted_df.columns:
            permuted_df[column] = np.random.permutation(permuted_df[column].values)
        permuted_dfs.append(permuted_df)
    return permuted_dfs

def calcular_distancia(nombre1, nombre2):
    """
    Calculates a distance metric based on shared ontology between two terms.

    Args:
        nombre1 (str): Name of the first ontology term.
        nombre2 (str): Name of the second ontology term.

    Returns:
        float: The calculated distance metric. Returns 0 if one of the terms is 'All' or if no common ancestors exist.
    """
    if nombre1 == 'All' or nombre2 == 'All':
        return 0
    else:
        term_1 = Ontology.get_hpo_object(nombre1)
        term_2 = Ontology.get_hpo_object(nombre2)
        ca = term_1.common_ancestors(term_2)
        if len(ca) == 0:
            return 0
        else:
            h = -1
            t = None
            for term in ca:
                if term.shortest_path_to_root() > h:
                    t = term
                    h = term.shortest_path_to_root()
            c1 = term_1.shortest_path_to_parent(t)[0]
            c2 = term_2.shortest_path_to_parent(t)[0]
            return 1 - ((2 * h) / (c1 + c2 + (2 * h)))

def inter_score_ont(dim, lamb, df_q, df):
    """
    Calculates interpretability scores for a given dimension.

    Args:
        dim (int): The dimension being evaluated.
        lamb (int): A scaling parameter.
        df_q (pd.DataFrame): Query DataFrame with scores.
        df (pd.DataFrame): Target DataFrame with ontology terms.

    Returns:
        list of float: A list of interpretability scores for different lambda values.
    """
    term = classes[df_q.iloc[dim].argmax()]
    terms = set_classes[term]
    nj = len(terms)

    # Positive side evaluation
    v_i = df.iloc[dim].argsort()[:(lamb * nj)]
    scoresplus = []
    for t in v_i:
        if hpo_map[phen_map[t]] in terms:
            scoresplus.append(1)
        else:
            scoresplus.append(0)

    # Negative side evaluation
    v_i = df.iloc[dim].argsort()[::-1][:(lamb * nj)]
    scoresneg = []
    for t in v_i:
        if hpo_map[phen_map[t]] in terms:
            scoresneg.append(1)
        else:
            scoresneg.append(0)

    # Aggregate scores
    scores = []
    for i in range(1, lamb + 1):
        score_p = np.array(scoresplus[:(i * nj)])
        scoreplus = np.sum(score_p)
        score_n = np.array(scoresneg[:(i * nj)])
        scoreneg = np.sum(score_n)
        scores.append((max(scoreplus, scoreneg) * 100) / nj)
    return scores

def inter_score_global_ont(lamb, df_q, df):
    """
    Calculates global interpretability scores across all dimensions.

    Args:
        lamb (int): A scaling parameter.
        df_q (pd.DataFrame): Query DataFrame with scores.
        df (pd.DataFrame): Target DataFrame with ontology terms.

    Returns:
        np.ndarray: An array of averaged interpretability scores.
    """
    scores = []
    for i in range(768):
        scores.append(inter_score_ont(i, lamb, df_q, df))
    scores = np.array(scores)
    return np.average(scores, axis=0)

def generate_auic(phen_map_path, phen_path_bio, phen_path_mpnet, phen_path_lord, phen_path_edu, 
                  classes_path, set_classes_path, q_b, q_b_mpnet, q_b_biolord, q_b_edu, out_path):
    """
    Generates an AUIC plot comparing interpretability scores across different models.

    Args:
        phen_map_path (str): Path to the phenotype map file.
        phen_path_bio (str): Path to the BioBERT phenotype scores.
        phen_path_mpnet (str): Path to the MPNet phenotype scores.
        phen_path_lord (str): Path to the BioLORD phenotype scores.
        phen_path_edu (str): Path to the BERT Education phenotype scores.
        classes_path (str): Path to the classes file.
        set_classes_path (str): Path to the set of classes file.
        q_b (str): Path to the BioBERT query file.
        q_b_mpnet (str): Path to the MPNet query file.
        q_b_biolord (str): Path to the BioLORD query file.
        q_b_edu (str): Path to the BERT Education query file.
        out_path (str): Path to save the output plot.

    Returns:
        None
    """
    np.random.seed(0)
    _ = Ontology(data_folder='data/hpo/hpo_ontology/')

    # Load data and mappings
    with open(phen_map_path, 'rb') as file:
        global phen_map
        phen_map = pickle.load(file)
    phen_map = {v: k for k, v in phen_map.items()}
    global hpo_map
    hpo_map = {v: Ontology.get_hpo_object(v).name for v in phen_map.values()}

    # Load phenotype data
    df_bio = pd.read_csv(phen_path_bio).transpose().rename(columns=phen_map).rename(columns=hpo_map)
    df_mpnet = pd.read_csv(phen_path_mpnet).transpose().rename(columns=phen_map).rename(columns=hpo_map)
    df_biolord = pd.read_csv(phen_path_lord).transpose().rename(columns=phen_map).rename(columns=hpo_map)
    df_edu = pd.read_csv(phen_path_edu).transpose().rename(columns=phen_map).rename(columns=hpo_map)

    # Load query data
    with open(q_b, 'rb') as file:
        Q_bat = pickle.load(file)
    df_QB = pd.DataFrame(Q_bat)

    with open(q_b_mpnet, 'rb') as file:
        Q_bat_mpnet = pickle.load(file)
    df_QBmpnet = pd.DataFrame(Q_bat_mpnet)

    with open(q_b_biolord, 'rb') as file:
        Q_bat_bio = pickle.load(file)
    df_QBbio = pd.DataFrame(Q_bat_bio)

    with open(q_b_edu, 'rb') as file:
        Q_bat_edu = pickle.load(file)
    df_QBedu = pd.DataFrame(Q_bat_edu)

    # Load ontology and classes
    df_hpo = Ontology().to_dataframe()
    df_hpo = df_hpo[~df_hpo['name'].str.contains('obsolete')]
    my_list = list(set(df_hpo.index.to_list()).intersection(set(phen_map.values())))
    df_hpo = df_hpo[df_hpo.index.isin(my_list)]
    df_hpo['children'] = df_hpo['children'].apply(lambda x: x.split('|') if x != '' else [])

    with open(classes_path, 'rb') as file:
        global classes
        classes = pickle.load(file)

    with open(set_classes_path, 'rb') as file:
        global set_classes
        set_classes = pickle.load(file)

    # Generate random matrices and calculate scores
    Q_rands = [generate_random_matrix() for _ in range(5)]
    scores_b = inter_score_global_ont(100, df_QB, df_bio)
    scores_bm = inter_score_global_ont(100, df_QBmpnet, df_mpnet)
    scores_bio = inter_score_global_ont(100, df_QBbio, df_biolord)
    scores_edu = inter_score_global_ont(100, df_QBedu, df_edu)

    # Compute AUIC areas
    area1 = np.trapz(scores_bm, range(1, 101)) / (100 * 100)
    area2 = np.trapz(scores_b, range(1, 101)) / (100 * 100)
    area3 = np.trapz(scores_bio, range(1, 101)) / (100 * 100)
    area4 = np.trapz(scores_edu, range(1, 101)) / (100 * 100)
    
    # Prepare data for plotting
    areas = [
        (area1, scores_bm, 'all-mpnet-base-v2, AUIC= ' + str(np.round(area1, 2)), 'C1', '-'),
        (area2, scores_b, 'BioBERT, AUIC= ' + str(np.round(area2, 2)), 'C2', '--'),
        (area3, scores_bio, 'BioLORD-2023, AUIC= ' + str(np.round(area3, 2)), 'C3', '-.'),
        (area4, scores_edu, 'BERT Education, AUIC= ' + str(np.round(area4, 2)), 'C4', ':'),
    ]

    # Sort areas in descending order for plotting
    areas_sorted = sorted(areas, key=lambda x: x[0], reverse=True)

    # Create a plot
    plt.figure(figsize=(8, 6), dpi=300)

    # Plot each AUIC curve
    for area, score, label, color, linestyle in areas_sorted:
        plt.plot(range(1, 101), score, label=label, color=color, linestyle=linestyle, linewidth=2)

    # Set plot labels, limits, and grid
    plt.xlabel(r'$\lambda$', fontsize=14)
    plt.ylabel('Interpretability Score (IS)', fontsize=14)
    plt.ylim(0, 100)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # Save the figure and show it
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == '__main__':
    # Execute the AUIC generation process with inputs from Snakemake
    generate_auic(
        snakemake.input[0],
        snakemake.input[1],
        snakemake.input[2],
        snakemake.input[3],
        snakemake.input[4],
        snakemake.input[5],
        snakemake.input[6],
        snakemake.input[7],
        snakemake.input[8],
        snakemake.input[9],
        snakemake.input[10],
        snakemake.output[0]
    )