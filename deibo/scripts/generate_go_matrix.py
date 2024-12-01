import sys
from goatools.base import get_godag
from goatools.semantic import TermCounts, get_info_content
from goatools.associations import dnld_assc
from goatools.gosubdag.gosubdag import GoSubDag
import pickle
import pandas as pd
import numpy as np
from multiprocessing import Pool
from scipy.spatial import distance
from scipy.stats import norm, zscore
from scipy.integrate import quad

VAR_CONST = 0.01  # Threshold for variance consideration


def bhattacharya(p, q):
    """
    Computes the Bhattacharya distance between two probability distributions.

    Parameters:
        p (array-like): Distribution 1
        q (array-like): Distribution 2

    Returns:
        float: Bhattacharya distance
    """
    vars = (p.var() / q.var()) + (q.var() / p.var()) + 2
    means = (p.mean() - q.mean())**2 / (p.var() + q.var())
    return 0.25 * np.log(0.25 * vars) + 0.25 * means


def hellinger(p, q):
    """
    Computes the Hellinger distance between two distributions.

    Parameters:
        p (array-like): Distribution 1
        q (array-like): Distribution 2

    Returns:
        float: Hellinger distance
    """
    root = np.sqrt((2 * q.std() * p.std()) / (q.var() + p.var()))
    ex = np.exp(-0.25 * ((p.mean() - q.mean())**2 / (p.var() + q.var())))
    return 1 - (root * ex)


def kl(p, q):
    """
    Computes the Kullback-Leibler divergence (KL divergence) between two distributions.

    Parameters:
        p (array-like): Distribution 1
        q (array-like): Distribution 2

    Returns:
        float: KL divergence
    """
    std0 = p.std()
    std1 = q.std()
    var0 = p.var()
    var1 = q.var()
    mean0 = p.mean()
    mean1 = q.mean()
    return (np.log(std1 / std0) + ((var0 + (mean0 - mean1)**2) / (2 * var1)) - 0.5)


def jeffreys(p, q):
    """
    Computes the Jeffreys divergence (symmetric KL divergence) between two distributions.

    Parameters:
        p (array-like): Distribution 1
        q (array-like): Distribution 2

    Returns:
        float: Jeffreys divergence
    """
    return (kl(p, q) + kl(q, p)) / 2


def js(p, q):
    """
    Computes the Jensen-Shannon divergence between two distributions.

    Parameters:
        p (array-like): Distribution 1
        q (array-like): Distribution 2

    Returns:
        float: Jensen-Shannon divergence
    """
    mu_1, sigma_1 = np.mean(p), np.std(p)
    mu_2, sigma_2 = np.mean(q), np.std(q)

    def pdf_1(x):
        return norm.pdf(x, mu_1, sigma_1)

    def pdf_2(x):
        return norm.pdf(x, mu_2, sigma_2)

    x = np.linspace(min(mu_1 - 5 * sigma_1, mu_2 - 5 * sigma_2), 
                    max(mu_1 + 5 * sigma_1, mu_2 + 5 * sigma_2), 
                    num=1000)
    px = np.array([pdf_1(a) for a in x])
    qx = np.array([pdf_2(a) for a in x])
    return distance.jensenshannon(px, qx)


def tvm(p, q):
    """
    Computes the Total Variation Metric (TVM) between two distributions.

    Parameters:
        p (array-like): Distribution 1
        q (array-like): Distribution 2

    Returns:
        float: Total Variation Metric
    """
    mu_1, sigma_1 = np.mean(p), np.std(p)
    mu_2, sigma_2 = np.mean(q), np.std(q)

    def pdf_1(x):
        return norm.pdf(x, mu_1, sigma_1)

    def pdf_2(x):
        return norm.pdf(x, mu_2, sigma_2)

    def tv_distance(x):
        return np.abs(pdf_1(x) - pdf_2(x))

    lower_bound = min(mu_1 - 5 * sigma_1, mu_2 - 5 * sigma_2)
    upper_bound = max(mu_1 + 5 * sigma_1, mu_2 + 5 * sigma_2)
    tv_distance_value, _ = quad(tv_distance, lower_bound, upper_bound, 
                                epsabs=1e-9, epsrel=1e-9, limit=1000)
    tv_distance_value /= 2  # Divided by 2 as per definition
    return tv_distance_value


def compute_q(args):
    """
    Computes the divergence or distance between a specific feature and class distributions.

    Parameters:
        args (tuple): Contains index pairs, dataframes, classes, and settings.

    Returns:
        tuple: Computed value and indices (i, j, value)
    """
    i, j, df, classes, set_classes, dist = args
    term = classes[j]
    d = set_classes[term]
    row = df.iloc[i]
    class_pos = row[[k in d for k in row.index]]
    class_neg = row[[k not in d for k in row.index]]

    if np.array(class_pos).var() <= VAR_CONST:
        return i, j, 0.0

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


def generate_matrix(go_embs_path, classes_path, set_classes_path, dist, subgo, output):
    """
    Generates a matrix comparing features and classes using specified distance metrics.

    Parameters:
        go_embs_path (str): Path to the GO BP embeddings CSV.
        classes_path (str): Path to the pickle file of classes.
        set_classes_path (str): Path to the pickle file mapping classes to terms.
        dist (str): Distance metric to use.
        subgo (str): Subontology to focus on (e.g., 'biological_process').
        output (str): Path to the output file for saving the matrix.

    Returns:
        None: Outputs the result to a file.
    """
    go = get_godag('go-basic.obo', optional_attrs=['relationship', 'def'])
    ids = [go[i].id for i in go if go[i].namespace == subgo]
    names = [go[i].name for i in go if go[i].namespace == subgo]
    dict_names = dict(zip(names, ids))

    df = pd.read_csv(go_embs_path)
    df['name'] = ids
    df = df.set_index(['name']).drop_duplicates().transpose()
    df = df.apply(lambda row: zscore(row), axis=1)

    with open(classes_path, 'rb') as file:
        classes = pickle.load(file)

    with open(set_classes_path, 'rb') as file:
        set_classes = pickle.load(file)

    Q = np.zeros((768, len(classes)))
    args_list = [(i, j, df, classes, set_classes, dist) for i in range(768) for j in range(len(classes))]

    with Pool(4) as pool:
        results = pool.map(compute_q, args_list)

    for i, j, value in results:
        Q[i, j] = value

    with open(output, 'wb') as file:
        pickle.dump(Q, file)
        
# Main script execution
if __name__ == '__main__':
    # Call the function with parameters provided by Snakemake
    generate_matrix(snakemake.input[0], snakemake.input[1],
                    snakemake.input[2], snakemake.params['dist'],
                    snakemake.params['sub'], snakemake.output[0])