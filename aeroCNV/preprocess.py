import pandas as pd

from . import utils
from .logging_config import log

def _normalize_over_genes(expression, diploid_expression, genes):
    """
    Normalize the expression data by the mean expression of each gene
    :param expression: (pd.DataFrame) Expression data, index=cells, columns=genes
    :param diploid_expression: (pd.DataFrame) Expression data for diploid genes, index=cells, columns=genes
    :param genes: (pd.Index) Genes to use for normalization
    :return:

    """
    cell_metadata = pd.DataFrame(index=expression.index)
    cell_metadata['Normalization'] = expression[genes].mean(axis=1)

    diploid_expression = diploid_expression.copy()

    diploid_ref = diploid_expression / diploid_expression[genes].mean(axis=1).values[:, None]
    if cell_metadata['Normalization'].min() == 0:
        log.warning('Some cells have a normalization factor of 0')
        # Remove cells with a normalization factor of 0
        cell_metadata = cell_metadata[cell_metadata['Normalization'] > 0]
        log.warning(f'Removed {len(expression) - len(cell_metadata)} cells with a normalization factor of 0')

    return cell_metadata, diploid_ref.mean(axis=0)

def _normalize_by_total_reads(expression, diploid_expression):
    """
    Normalize the expression data by the mean number of reads in each cell

    :param expression: (pd.DataFrame) Expression data, index=cells, columns=genes
    :param diploid_expression: (pd.DataFrame) Expression data for diploid genes, index=cells, columns=genes
    :return: (pd.DataFrame, pd.DataFrame) Normalization constant for expression data, normalized diploid expression data
    """
    return _normalize_over_genes(expression, diploid_expression, expression.columns)


def _normalize_by_median(expression, diploid_expression, percentile_bounds=(.25,.75)):
    """
    Normalize the expression data by the mean gene expression in the (default) 25th to 75th percentile range for each cell.
    :param expression: (pd.DataFrame) Expression data, index=cells, columns=genes
    :param diploid_expression: (pd.DataFrame) Expression data for diploid genes, index=cells, columns=genes
    :return: (pd.DataFrame, pd.DataFrame) Normalization constant for expression data, normalized diploid expression data
    """
    cell_metadata = pd.DataFrame(index=expression.index)

    lower_percentile, upper_percentile = percentile_bounds
    assert 0 <= lower_percentile < upper_percentile <= 11, 'Percentile bounds must be in the range [0, 1]'

    # Step 1: Calculate the mean expression level for each gene across all cells
    gene_means = expression.mean(axis=0)

    # Step 2: Determine the 25th and 75th percentile expression levels
    lower_percentile = gene_means.quantile(lower_percentile)
    upper_percentile = gene_means.quantile(upper_percentile)

    # Step 3: Identify the genes that fall within the 25th to 75th percentile range
    genes_in_range = gene_means[(gene_means >= lower_percentile) & (gene_means <= upper_percentile)].index

    return _normalize_over_genes(expression, diploid_expression, genes_in_range)

def _get_gene_variance_ratio(expression, diploid_expression):
    """
    Compute the variance of gene expression in diploid cells and all cells, then divide the two to get a ratio.
    We will use this ratio to identify genes with higher variance than expected in the diploid cells.
    :param expression: (pd.DataFrame) Expression data, index=cells, columns=genes
    :param diploid_expression: (pd.DataFrame) Expression data for diploid genes, index=cells, columns=genes
    :return: (pd.DataFrame) Gene variance ratio, index=genes, columns=['Diploid', 'Expression', 'Ratio']
    """

    # Normalize both the expression data and the diploid expression data by the total number of reads in each cell
    expression_norm = expression.div(expression.mean(axis=1), axis=0)
    diploid_expression_norm = diploid_expression.div(diploid_expression.mean(axis=1), axis=0)

    # Compute the variance for each gene
    diploid_variance = diploid_expression_norm.var(axis=0)
    variance = expression_norm.var(axis=0)

    gene_variances = pd.concat([diploid_variance, variance], axis=1)
    gene_variances.columns = ['Diploid', 'Expression']

    # Divide the variance by the diploid variance for each gene
    gene_variances['Ratio'] = gene_variances['Expression'] / gene_variances['Diploid']
    return gene_variances

def _infer_diploid_genes(gene_variances, low_variance_percent):
    """
    Infer diploid genes by selecting the genes with the lowest variance ratio
    :param gene_variances: (pd.DataFrame) Gene variance ratio, index=genes, columns=['Diploid', 'Expression', 'Ratio']
    :param low_variance_percent: (float (0,100) ) Percentage of genes to select with the lowest variance
    :return: (pd.Index) Index of inferred diploid genes
    """
    assert 0 < low_variance_percent < 100, 'low_variance_percent must be between 0 and 100'
    cutoff = gene_variances['Ratio'].quantile(low_variance_percent / 100)
    return gene_variances[gene_variances['Ratio'] < cutoff].index

def _normalize_by_gene_variance(expression, diploid_expression, low_variance_percent=10):
    """
    Normalize the expression data by the mean expression of genes with low variance.
    We first normalize the data by the median gene expression in each cell, then identify genes with low variance

    :param expression: (pd.DataFrame) Expression data, index=cells, columns=genes
    :param diploid_expression: (pd.DataFrame) Expression data for diploid genes, index=cells, columns=genes
    :param low_variance_percent: (float (0,100) ) Percentage of genes to select with the lowest variance
    :return:
    """
    gene_variances = _get_gene_variance_ratio(expression, diploid_expression)
    inferred_diploid_genes = _infer_diploid_genes(gene_variances, low_variance_percent)

    # Sum the expression of the inferred diploid genes in non-normalized data
    return _normalize_over_genes(expression, diploid_expression, inferred_diploid_genes)


def normalize_inputs(expression, diploid_expression, flavour='gene_variance', low_variance_percent=20):
    """
    Normalize the expression data by the mean expression of genes with low variance.
    :param expression: (pd.DataFrame) Expression data, index=cells, columns=genes
    :param diploid_expression: (pd.DataFrame) Expression data for diploid genes, index=cells, columns=genes
    :param flavour: (str) Normalization approach to use. Options are 'total_reads', 'median', 'gene_variance'
    :param low_variance_percent: (float (0,100) ) Percentage of genes to select with the lowest variance
    :return: (pd.DataFrame, pd.DataFrame) Normalization constant for expression data, normalized diploid expression data
    """
    flavour = flavour.lower()
    if flavour == 'total_reads':
        return _normalize_by_total_reads(expression, diploid_expression)
    elif flavour == 'median':
        return _normalize_by_median(expression, diploid_expression)
    elif flavour == 'gene_variance':
        return _normalize_by_gene_variance(expression, diploid_expression, low_variance_percent)
    else:
        raise ValueError(f'Normalization flavour {flavour} not recognized')

def remove_cycle_genes(df):
    """
    Remove cell cycle genes from the expression data
    :param df: (pd.DataFrame) Expression data, index=cells, columns=genes
    """
    path = utils.get_absolute_path('cyclegenes.csv')
    cycle_genes = pd.read_csv(path)['Gene'].values
    # Select the genes that are not in the cell cycle gene list
    return df.loc[:, ~df.columns.isin(cycle_genes)]


