# Import necessary libraries
from goatools.obo_parser import GODag  # For parsing the Gene Ontology (GO) DAG
import torch  # For tensor operations and computations on GPU
from sentence_transformers import SentenceTransformer  # For generating text embeddings
import numpy as np  # For numerical operations
import os  # For file and path operations
import pandas as pd  # For handling tabular data
from goatools.semantic import TermCounts, get_info_content  # For semantic similarity and GO term information
from goatools.associations import dnld_assc  # For downloading GO term associations
from goatools.base import get_godag  # For loading the Gene Ontology DAG

# Function to load a CSV file and optionally encode its features
def load_node_csv(path, index_col, encoders=None, **kwargs):
    """
    Loads a node CSV file, maps indices, and optionally encodes features.

    Parameters:
    path (str): Path to the CSV file.
    index_col (str): Column to use as the index.
    encoders (dict or None): Dictionary mapping column names to encoders (e.g., BioBERT).
    kwargs: Additional arguments for pandas `read_csv`.

    Returns:
    tuple: (tensor of features, mapping of indices to integers)
    """
    # Load the CSV into a DataFrame
    df = pd.read_csv(path, index_col=index_col, **kwargs)

    # Map unique index values to integer identifiers
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    # Initialize feature matrix
    x = None
    if encoders is not None:
        # Encode specified columns and concatenate their outputs
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)
    else:
        # Default case: use ones as placeholder features
        xs = [np.ones(len(df['gene']))]
        x = torch.cat(xs, dim=-1)

    return x, mapping

# Function to generate embeddings for GO terms
def generate_embs(go_path, embs_path, subgo):
    """
    Generate embeddings for Gene Ontology terms and save them.

    Parameters:
    go_path (str): Path to save the GO term definitions.
    embs_path (str): Path to save the generated embeddings.
    subgo (str): GO namespace to filter terms (e.g., 'biological_process').
    """
    # Load the Gene Ontology DAG
    go = get_godag('go-basic.obo', optional_attrs=['relationship', 'def'])

    # Extract GO term IDs and their definitions for the specified namespace
    ids = [go[i].id for i in go if go[i].namespace == subgo]
    names = [go[i].defn for i in go if go[i].namespace == subgo]

    # Create a DataFrame for the GO terms and their definitions
    gos = pd.DataFrame({'Ids': ids, 'Names': names})

    # Save the DataFrame to a CSV file
    gos.to_csv(go_path, index=False)

    # Load the GO terms and encode their definitions using BioBERT
    pheno_x, pheno_mapping = load_node_csv(
        go_path, index_col='Ids', encoders={
            'Names': SequenceEncoder()  # Use BioBERT to encode the 'Names' column
        })

    # Convert the generated embeddings to a DataFrame
    df = pd.DataFrame(pheno_x.numpy())

    # Save the embeddings to a CSV file
    df.to_csv(embs_path, index=False)

# Main script execution
if __name__ == '__main__':
    global model_path
    model_path = snakemake.params['model']
    # Class for encoding sequences using BioBERT
    class SequenceEncoder(object):
        """
        Encodes text data using a BioBERT-based pre-trained SentenceTransformer model.

        Parameters:
        model_name (str): The name of the BioBERT model from Hugging Face.
        device (str or None): Device to run the model ('cuda' for GPU, 'cpu' for CPU).
        """
        def __init__(self, model_name=model_path, device=None):
            self.device = device
            self.model = SentenceTransformer(model_name, device=device)

        @torch.no_grad()
        def __call__(self, df):
            """
            Encodes the input text data into embeddings.

            Parameters:
            df (pd.Series or list): Text data to be encoded.

            Returns:
            torch.Tensor: Encoded embeddings as a tensor.
            """
            # Generate embeddings and ensure computations are done without gradients for efficiency
            x = self.model.encode(df.values, show_progress_bar=True,
                                convert_to_tensor=True, device=self.device)
            return x.cpu()  # Return tensor on the CPU for consistency
    # Generate embeddings using parameters from Snakemake
    generate_embs(snakemake.output[0], snakemake.output[1], snakemake.params['sub'])