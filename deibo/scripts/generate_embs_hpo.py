# Import necessary libraries
import pandas as pd  # For handling tabular data
from sentence_transformers import SentenceTransformer  # For generating text embeddings
import torch  # For tensor operations
import numpy as np  # For numerical operations

# Function to load a CSV file and optionally encode its features
def load_node_csv(path, index_col, encoders=None, **kwargs):
    """
    Loads a CSV file, maps indices, and encodes specified columns if encoders are provided.

    Parameters:
    path (str): Path to the CSV file.
    index_col (str): Column to use as the index.
    encoders (dict or None): Dictionary mapping column names to encoders (e.g., SentenceTransformer).
    kwargs: Additional arguments for pandas `read_csv`.

    Returns:
    tuple: (tensor of encoded features, mapping of indices to integers)
    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv(path, index_col=index_col, **kwargs)

    # Create a mapping from unique index values to integer identifiers
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None  # Placeholder for the feature matrix
    if encoders is not None:
        # Encode specified columns using provided encoders and concatenate their outputs
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)  # Concatenate along the last dimension
    else:
        # If no encoders are provided, use ones as placeholder features
        xs = [np.ones(len(df['gene']))]
        x = torch.cat(xs, dim=-1)

    return x, mapping

# Function to generate embeddings and save them
def generate_embs(embs_path):
    """
    Generates embeddings for phenotype definitions and saves them to a file.

    Parameters:
    embs_path (str): Path to save the generated embeddings.
    """
    # Path to the phenotype data
    phenotypes = 'data/hpo/phenotypes.csv'

    # Load phenotype data and encode definitions using the SequenceEncoder
    pheno_x, pheno_mapping = load_node_csv(
        phenotypes, index_col='Phenotypes', encoders={
            'Definition': SequenceEncoder()  # Encode the 'Definition' column
        })

    # Convert the encoded features to a DataFrame and save them to a CSV file
    df = pd.DataFrame(pheno_x.numpy())
    df.to_csv(embs_path, index=False)

# Main script execution
if __name__ == '__main__':
    # Load the model path from Snakemake parameters
    global model_path
    model_path = snakemake.params['model']

    # Define a class for encoding sequences using a SentenceTransformer model
    class SequenceEncoder(object):
        """
        Encodes text data using a SentenceTransformer model.

        Parameters:
        model_name (str): Name of the SentenceTransformer model.
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
            df (pd.Series or list): Text data to encode.

            Returns:
            torch.Tensor: Encoded embeddings as a tensor.
            """
            # Generate embeddings and ensure computations are done without gradients
            x = self.model.encode(df.values, show_progress_bar=True,
                                  convert_to_tensor=True, device=self.device)
            return x.cpu()  # Return embeddings as a tensor on the CPU

    # Define a class for identity encoding (returns the input as-is)
    class IdentityEncoder(object):
        """
        Encodes data as-is by converting it into a PyTorch tensor.

        Parameters:
        dtype (torch.dtype or None): Desired data type for the output tensor.
        """
        def __init__(self, dtype=None):
            self.dtype = dtype

        def __call__(self, df):
            """
            Converts input data to a PyTorch tensor.

            Parameters:
            df (pd.Series or list): Data to encode.

            Returns:
            torch.Tensor: Encoded data as a tensor.
            """
            return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)

    # Define a class for constant encoding (returns a tensor of ones)
    class ConstantEncoder(object):
        """
        Encodes data as a constant value (e.g., 1.0) for all inputs.

        Parameters:
        dtype (torch.dtype or None): Desired data type for the output tensor.
        """
        def __init__(self, dtype=None):
            self.dtype = dtype

        def __call__(self, df):
            """
            Generates a tensor of ones for the input data.

            Parameters:
            df (pd.Series or list): Data to encode.

            Returns:
            torch.Tensor: Tensor of ones with the same length as the input.
            """
            ct = np.asarray([1.0 for _ in range(len(df.values))])
            return torch.from_numpy(ct).view(-1, 1).to(self.dtype)

    # Call the generate_embs function with Snakemake's output path
    generate_embs(snakemake.output[0])