import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import os
from collections import Counter

from . import utils
from . import preprocess as pp
def count_values(row):
    """

    :param row:
    :return:
    """
    return row.value_counts()

def describe_profiles_A():
    path = utils.get_absolute_path('CNV_A_Profiles.csv')
    cnvs = pd.read_csv(path, header=[0], index_col=0)
    cnvs = pp.remove_cycle_genes(cnvs)

    plt.figure(figsize=(5,1))
    sns.heatmap(cnvs, cmap='coolwarm', vmin=0, center=2)
    plt.suptitle('Copy Number Profiles for Each Sample')
    plt.show()
    plt.close()

    # Apply the function to each row
    row_counts = cnvs.apply(count_values, axis=1).fillna(0).astype(int)
    print(f'The number of genes with each copy number in each sample are: \n{row_counts}')

    path2 = utils.get_absolute_path('expression_A_profiles.csv')
    df = pd.read_csv(path2, index_col=[0,1], header=[0])
    df = pp.remove_cycle_genes(df)

    sample_counts = Counter(df.index.get_level_values(0))
    print(f'There are {sample_counts["Control_Brain_Met_36"]} diploid cells (Control_Brain_Met_36) and {sample_counts["Control_Primary_5"]} aneuploid cells (Control_Primary_5).')

    return cnvs, df

def describe_profiles_B():
    path = utils.get_absolute_path('CNV_B_Profiles.csv')
    cnvs = pd.read_csv(path, header=[0], index_col=0)
    cnvs = pp.remove_cycle_genes(cnvs)
    plt.figure(figsize=(5,1))
    sns.heatmap(cnvs, cmap='coolwarm', vmin=0, center=2)
    plt.suptitle('Copy Number Profiles for Each Sample')
    plt.show()
    plt.close()

    # Apply the function to each row
    row_counts = cnvs.apply(count_values, axis=1).fillna(0).astype(int)
    print(f'The number of genes with each copy number in each sample are: \n{row_counts}')

    path2 = utils.get_absolute_path('expression_B_profiles.csv')
    df = pd.read_csv(path2, index_col=[0,1], header=[0])
    df = pp.remove_cycle_genes(df)
    sample_counts = Counter(df.index.get_level_values(0))
    print(f'There are {sample_counts["Control_Brain_Met_31"]} diploid cells (Control_Brain_Met_31) and {sample_counts["Control_Primary_2"]} normal cells (Control_Primary_2).')

    return cnvs, df


def one_clone_piecewise_from_real_data(n_tumour_cells, n_normal_cells, n_reference_cells, alteration_widths,
                                       total_genome_length):
    """
    Simulate a non-normal clone from real data by rearranging the genes in clone_profiles_A to generate
    ground truth CNV profiles and expression levels.
    :param n_tumour_cells: (int)
    :param n_normal_cells:
    :param n_reference_cells: (int)
    :param alteration_widths:
    :param total_genome_length:
    :return:
    """

    genes, labels = get_genes_and_labels(alteration_widths, total_genome_length)

    print(f'Using {len(genes)} genes for the simulation.')

    path = utils.get_absolute_path('expression_A_profiles.csv')
    expression = pd.read_csv(path, index_col=[0, 1], header=[0])
    expression = pp.remove_cycle_genes(expression)
    expression = expression[genes]

    sample_counts = Counter(expression.index.get_level_values(0))
    # Get the correct number of tumour and normal cells
    max_tumour_cells = sample_counts["Control_Primary_5"]
    max_normal_cells = sample_counts["Control_Brain_Met_36"]

    assert n_tumour_cells <= max_tumour_cells, f'There are only {max_tumour_cells} tumour cells in the dataset, cannot have {n_tumour_cells} tumour cells'
    assert n_normal_cells + n_reference_cells <= max_normal_cells, f'There are only {max_normal_cells} normal cells in the dataset, cannot have {n_normal_cells} normal cells and {n_reference_cells} reference cells'

    # Randomly select the appropriate number of tumour cells
    tumour_cells = np.random.choice(expression.index[expression.index.get_level_values(0) == "Control_Primary_5"],
                                    size=n_tumour_cells, replace=False)
    normal_cells = np.random.choice(expression.index[expression.index.get_level_values(0) == "Control_Brain_Met_36"],
                                    size=n_normal_cells + n_reference_cells, replace=False)

    sampled_cells = np.concatenate([tumour_cells, normal_cells[:n_normal_cells]])

    print(
        f'Sampled {n_tumour_cells} tumour cells and {n_normal_cells} normal cells from the dataset, giving a total of {len(sampled_cells)} cells.')
    reference_cells = normal_cells[n_normal_cells:]

    sampled_expression = expression.loc[sampled_cells]
    reference_expression = expression.loc[reference_cells]

    # Concatenate two levels of the index to get a unique identifier for each cell
    sampled_expression.index = sampled_expression.index.map(lambda x: '_'.join(map(str, x)))
    reference_expression.index = reference_expression.index.map(lambda x: '_'.join(map(str, x)))

    # Construct the ground truth copy number profiles
    labels = [labels] * len(tumour_cells) + [[2] * len(labels)] * n_normal_cells
    labels = pd.DataFrame(labels, index=sampled_expression.index, columns=sampled_expression.columns)

    return reference_expression, sampled_expression, labels


def get_genes_and_labels(alteration_widths, total_genome_length):
    """
    Get a list of genes and their corresponding labels for a given set of alteration widths and total genome length
    :param alteration_widths: (list) A list of integers representing the width of each alteration in the genome
    :param total_genome_length: (int) The total length of the genome
    :return: (list, list) A list of genes and a list of CNV labels for each gene
    """

    path = utils.get_absolute_path('CNV_A_Profiles.csv')
    cnvs = pd.read_csv(path, header=[0], index_col=0)
    cnvs = pp.remove_cycle_genes(cnvs)

    altered_genes = list(cnvs.columns[cnvs.loc['Control_Primary_5'] == 3])
    np.random.shuffle(altered_genes)
    max_alterations = len(altered_genes)

    n_altered_genes = sum(alteration_widths)
    altered_genes = altered_genes[:n_altered_genes]

    # Check to make sure the genome length is less that the number of genes we have data for
    assert n_altered_genes < max_alterations, 'The total length of the alterations is greater than the number of genes in the dataset'
    assert n_altered_genes < total_genome_length, 'The total length of the alterations is greater than the number of normal genes'

    n_normal_genes = total_genome_length - n_altered_genes
    print(f'In total we have {n_normal_genes} normal genes and {n_altered_genes} altered genes')

    # Randomly partition the normal genes into len(alteration_widths) + 1 groups
    normal_genes = list(cnvs.columns[cnvs.loc['Control_Primary_5'] == 2])

    assert len(normal_genes) >= n_normal_genes, 'There are not enough normal genes in the dataset to cover the total genome length'

    # Shuffle the order of the gene_lists
    np.random.shuffle(normal_genes)
    # Only keep enough normal genes to cover the total genome length
    normal_genes = normal_genes[:n_normal_genes]

    partitions = sorted(np.random.choice(len(normal_genes), size=len(alteration_widths), replace=False))
    partitions = [0] + partitions + [len(normal_genes)]
    alteration_widths = np.cumsum([0] + alteration_widths)

    normal_chunks = [normal_genes[partitions[i]:partitions[i + 1]] for i in range(len(partitions) - 1)]
    altered_chunks = [altered_genes[alteration_widths[i]:alteration_widths[i + 1]] for i in
                      range(len(alteration_widths) - 1)]

    genes = []
    labels = []
    for i in range(len(partitions) - 2):
        genes.append(normal_chunks[i])
        genes.append(altered_chunks[i])
        labels.append([2] * len(normal_chunks[i]))
        labels.append([3] * len(altered_chunks[i]))

    genes.append(normal_chunks[-1])
    labels.append([2] * len(normal_chunks[-1]))
    genes = sum(genes, [])
    labels = sum(labels, [])

    return genes, labels
