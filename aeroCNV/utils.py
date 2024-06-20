import numpy as np
import pandas as pd
import os

import torch

from .logging_config import log

def get_absolute_path(file_name):
    """
    Get the absolute path of a file in the resources directory
    :param file_name:
    :return:
    """
    # Check if the file exists
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'resources', file_name)):
        raise FileNotFoundError("File does not exist")
    return os.path.join(os.path.dirname(__file__), 'resources', file_name)


def load_genemap():
    """
    Load gene map for sorting genes according to their chromosomal location
    """
    chr_gene_dict = {ch: set() for ch in np.arange(1, 25)}
    gene_loc_dict = {}
    path = get_absolute_path('genePos.txt')
    try:
        with open(path) as ifile:
            for line in ifile:
                content = line.strip().split(' ')
                gene, chrom, loc = content[0], content[1][3:], int(content[2])
                if chrom == 'X':
                    ch = 23
                elif chrom == 'Y':
                    ch = 24
                else:
                    ch = int(chrom)

                chr_gene_dict[ch].add(gene)
                gene_loc_dict[gene] = loc
    except FileNotFoundError:
        log.error("Gene map at {} doesn't exist".format(path))
    except ValueError:
        log.error("Invalid gene / chromosome / location name in Gene map at {}".format(path))

    return chr_gene_dict, gene_loc_dict

def order_genes(gene_list):
    """
    Order genes according to their chromosomal location
    :param gene_list: (list) List of genes
    :return genes_sorted: (pd.DataFrame) Sorted genes and corresponding chromosomes,
            genes_not_found: (list) Genes not found in the gene map
    """

    chr_gene_dict, gene_loc_dict = load_genemap()

    gene_level = []
    chr_level = []
    for ch in np.arange(1, 25):
        genes_unsorted = [gene for gene in gene_list if gene in chr_gene_dict[ch]]
        if len(genes_unsorted) > 0:
            positions = np.vectorize(lambda gene: gene_loc_dict[gene])(genes_unsorted)
            genes_sorted = [list(t) for t in zip(*sorted(zip(positions, genes_unsorted)))][1]
            gene_level.extend(genes_sorted)
            chr_level.extend([ch] * len(genes_sorted))
    genes_not_found = list(set(gene_list) - set(gene_level))
    genes_sorted = pd.DataFrame({'Chromosome': chr_level, 'Gene': gene_level})
    if len(genes_not_found) > 0:
        log.info(f'Could not find {len(genes_not_found)} genes in the gene map: {genes_not_found}')
    log.info(f'Sorted {len(gene_level)} genes according to their chromosomal location')
    return genes_sorted, genes_not_found

def generate_default_transitions(genes, epsilon=0.1, offset=0):
    """
    Generate default transition matrix for a set of genes
    :param genes: (list) Genes to generate transitions for
    :param epsilon: (float) Probability of off-diagonal transition
    :param offset: (int) Offset for off-diagonal transition
    :return: (pd.DataFrame) Transition matrix for genes containing epsilon and offset columns
    """
    default_transitions = pd.DataFrame(index=genes)
    default_transitions['epsilon'] = epsilon
    default_transitions['offset'] = offset
    return default_transitions


def create_transition_matrix(n_states, epsilon, offset):
    """
    Create a transition matrix for a given number of states with specified epsilon and offset.

    :param n_states: (int) The number of states in the transition matrix.
    :param epsilon: (float) The probability value used to fill non-diagonal elements or offset elements.
    :param offset: (int) The offset value indicating the main transition direction.
                   - offset = 0: Main transition on the diagonal.
                   - offset > 0: Main transition one or more positions above the diagonal.
                   - offset < 0: Main transition one or more positions below the diagonal.
    :return: (torch.Tensor) The generated transition matrix of size (n_states, n_states).
    """

    # Initialize the matrix with epsilon / (n_states - 1) to ensure probabilities sum to 1
    matrix = torch.full((n_states, n_states), epsilon / (n_states - 1), dtype=torch.float64)

    if offset == 0:
        # For zero offset, set the diagonal elements to 1 - epsilon
        diag_idx = torch.arange(n_states)
        matrix[diag_idx, diag_idx] = 1 - epsilon
    else:
        # For non-zero offsets, adjust the matrix to set the main transition direction
        for i in range(n_states):
            # Calculate the column index with the specified offset, ensuring it wraps around
            j = (i + offset) % n_states

            # Set the main transition probability
            matrix[i, int(j)] = 1 - epsilon

            # Create a mask to set other values in the row to epsilon / (n_states - 1)
            mask = torch.ones(n_states, dtype=bool)
            mask[int(j)] = False
            matrix[i, mask] = epsilon / (n_states - 1)

    return matrix

def generate_default_celltype_params(genes, parameter_priors):
    """
    Generate default celltype parameters for a set of genes. Parameters used for optimizing celltype-specific
    gene expression and response are: diploid_mean and alpha, alpha_std to allow optimization of the gene expression r
    esponse to CNV. Alpha parameters are drawn from a Normal distribution with specified priors.
    :param genes: (list) Genes to generate parameters for
    :param parameter_priors: (dict) Dictionary containing the priors for the parameters. Must contain the following keys:
            - diploid_mean: (float) Mean of the diploid gene expression
            - diploid_std: (float) Standard deviation of the diploid gene expression
            - alpha: (float) Mean of the gene expression response to CNV
            - alpha_std: (float) Standard deviation of the gene expression response to CNV
    :return: pd.DataFrame containing the generated parameters for the genes.
    """
    default_params = pd.DataFrame(index=genes)
    default_params['diploid'] = parameter_priors['diploid_mean']
    default_params['alpha'] = np.random.normal(parameter_priors['alpha_mean'], parameter_priors['alpha_std'], len(genes))
    return default_params

def generate_default_celltype_priors(genes, parameter_priors):
    """
    Generate default celltype priors for a set of genes. Priors used for optimizing celltype-specific gene expression
    and response are: diploid_mean, diploid_std and alpha, alpha_std to allow optimization of the gene expression response to CNV.
    :param genes: (list) Genes to generate priors for
    :param parameter_priors: (dict) Dictionary containing the priors for the parameters. Must contain the following keys:
            - diploid_mean: (float) Mean of the diploid gene expression
            - diploid_std: (float) Standard deviation of the diploid gene expression
            - alpha: (float) Mean of the gene expression response to CNV
            - alpha_std: (float) Standard deviation of the gene expression response to CNV
    :return: (pd.DataFrame) containing the columns 'diploid_mean', 'diploid_std', 'alpha_mean' and 'alpha_std'
    """
    default_priors = pd.DataFrame(index=genes)
    default_priors['diploid_mean'] = parameter_priors['diploid_mean']
    default_priors['diploid_std'] = parameter_priors['diploid_std']
    default_priors['alpha_mean'] = parameter_priors['alpha_mean']
    default_priors['alpha_std'] = parameter_priors['alpha_std']
    return default_priors

def filter_genes(expression, leg_threshold=10):
    """
    Remove low expressed genes from the expression data
    :param expression: (pd.DataFrame) Expression data
    :param leg_threshold: (float) Percentile threshold for filtering genes. The lower the threshold, the more genes are
    retained. (Default: 10, so that the 10% of genes with the lowest expression are removed)
    :return: (pd.DataFrame) Filtered expression data
    """

    # Calculate the mean expression for each gene across all cells
    gene_means = expression.mean(axis=0)

    # Determine the removal threshold based on the specified percentile
    removal_threshold = np.percentile(gene_means, leg_threshold)

    # Identify genes to keep, i.e., genes with mean expression above the threshold
    genes_to_keep = gene_means >= removal_threshold

    # Check if all genes are to be removed
    if len(genes_to_keep) == 0:
        log.info('Preprocessing attempting to remove all genes. Falling back to genes with non-zero expression.')
        genes_to_keep = gene_means > 0
        if len(genes_to_keep) == 0:
            raise ValueError('All genes have zero expression. Please check the input data.')
    log.info(f'Removed {len(genes_to_keep) - genes_to_keep.sum()} genes due to low expression.')
    return expression.loc[:, genes_to_keep]

def _separate_chromosomes(df, reset_character=-1):
    """
    Modify data and transition matrix to insert a reset state at the beginning of each chromosome (so that
    transitions are reset between chromosomes/non-contiguous regions of the genome).
    :param: df (pd.DataFrame) - DataFrame of data to modify. Must contain a MultiIndex with levels 'Chromosome' and
            'Gene'
    :param: reset_character (int) - character to use for reset state
    :return: (pd.DataFrame) Modified DataFrame with reset states inserted at the beginning of each chromosome
    """
    # Get the unique chromosomes
    unique_chromosomes = df.columns.get_level_values('Chromosome').unique()

    for chrom in unique_chromosomes:
        # Create the new column with the label "RESET_CHR_i"
        reset_column = pd.MultiIndex.from_tuples([(chrom, f'RESET_CHR_{chrom}')], names=['Chromosome', 'Gene'])
        reset_data = pd.DataFrame(reset_character, index=df.index, columns=reset_column)

        # Concatenate the new column with the original DataFrame
        df = pd.concat([reset_data, df], axis=1)

        # Reorder the columns to ensure the new column is the first for its chromosome
        all_columns = df.columns.to_list()
        chrom_columns = [col for col in all_columns if col[0] == chrom]
        new_order = [col for col in all_columns if col not in chrom_columns] + chrom_columns

        df = df.reindex(columns=new_order)

    return df

def _pad_transition_matrix(matrix, n_states):
    """
    Pad the transition matrix to include transitions to and from the reset state.
    The first column of the transition matrix is assumed to be the reset state.
    :param matrix: (torch.Tensor) Transition matrix to pad
    :param n_states: (int) Number of states in the original matrix
    :return: (torch.Tensor) Padded transition matrix
    """
    # Create a new matrix with additional rows and columns for the reset state
    padded_matrix = torch.zeros(n_states + 1, n_states + 1, dtype=matrix.dtype)
    padded_matrix[1:, 1:] = matrix
    # Set the transitions from the reset state to all other states to be uniform
    padded_matrix[0, 1:] = 1 / n_states
    return padded_matrix

def _zero_out_diagonal_offsets(matrix, offset):
    """
    Zero out specific diagonal offsets in a n_steps x n_states x n_states matrix.

    :param matrix (torch.Tensor): Input tensor of shape (n_states, n_states).
    :param offsets (int): Offset to zero out.

    :return torch.Tensor: The modified matrix with specific diagonals zeroed out.
    """
    n_states, _ = matrix.shape

    if offset < 0:
        for i in range(n_states + offset):
            matrix[ i - offset, i] = 0
    elif offset > 0:
        for i in range(n_states - offset):
            matrix[i, i + offset] = 0
    else:
        for i in range(n_states):
            matrix[i, i] = 0

    return matrix
