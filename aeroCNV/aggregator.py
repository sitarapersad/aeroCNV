import scanpy as sc
import pandas as pd
import numpy as np

from .logging_config import log
import SEACells

def compute_PCA_embeddings(ad, normal_genes=None, n_hvg=2000):
    """
    Compute PCA embeddings for SEACell aggregation. If normal_genes are known, normalize against them and then remove them from the variables.
    Otherwise, normalize against total library size. Select highly variable genes and perform PCA on these.
    :param ad: (AnnData) Annotated data matrix
    :param normal_genes: (list) List of normal genes
    :param n_hvg: (int) Number of highly variable genes to select
    :return: (AnnData) Annotated data matrix with PCA embeddings
    """
    ad = ad.copy()
    # If normal genes are known, normalize against them
    if normal_genes is not None:
        log.debug('Normalizing against normal genes')
        ad.obs['total_counts'] = ad[:, normal_genes].X.mean(axis=1)
        ad.X /= ad.obs['total_counts'].values[:, None]
        ad = ad[:, ~ad.var_names.isin(normal_genes)].copy()
    else:
        ad.obs['total_counts'] = ad.X.sum(axis=1)
        sc.pp.normalize_per_cell(ad)

    sc.pp.log1p(ad)
    # Select highly variable genes
    sc.pp.highly_variable_genes(ad, n_top_genes=n_hvg)
    ad = ad[:, ad.var.highly_variable]

    # Compute PCA embeddings
    sc.tl.pca(ad)

    # Plot UMAP coloured by clones
    if 'clone' in ad.obs:
        sc.pp.neighbors(ad, n_neighbors=15)
        sc.tl.umap(ad)
        sc.pl.umap(ad, color='clone', s=20, cmap='Set1')

    return ad


def compute_SEACells(ad, min_avg_cells_per_SEACell=20, min_avg_proportion_per_SEACell=0.05):
    """
    Compute SEACells for an anndata; assumes the anndata has PCA embeddings pre-computed
    :param ad: (AnnData) Annotated data matrix
    :param min_avg_cells_per_SEACell: (int) Minimum average number of cells per SEACell
    :param min_avg_proportion_per_SEACell: (float) Minimum average proportion of total cells per SEACell
    :return: (pd.DataFrame) SEACells labels for each cell
    """
    # We would like to have roughly 20 cells per SEACell, or 5% of the total number of cells, whichever is smaller.
    # We expect that we will not be able to reliably aggregate clones present in less than 5% of the cells anyway.
    assert 'X_pca' in ad.obsm, 'PCA embeddings not found in AnnData. Please ensure that PCA embeddings are computed'
    assert 0 < min_avg_proportion_per_SEACell < 1, 'min_avg_proportion_per_SEACell must be between 0 and 1'
    assert min_avg_cells_per_SEACell > 0, 'min_avg_cells_per_SEACell must be greater than 0'

    cells_per_seacell = min(min_avg_proportion_per_SEACell * ad.shape[0], min_avg_cells_per_SEACell)
    n_SEACells = int(ad.shape[0] / cells_per_seacell)
    print(f'Aggregating into {n_SEACells} SEACells with ~{cells_per_seacell} cells each')
    model = SEACells.core.SEACells(ad,
                                   n_SEACells=n_SEACells,
                                   build_kernel_on='X_pca')

    model.construct_kernel_matrix()
    model.initialize_archetypes()
    model.fit(min_iter=10, max_iter=50)
    return model.get_hard_assignments()


def compute_clonal_purity(ad, seacells):
    """
    Compute the clonal purity of each SEACell - we construct a matrix that contains the proportion of each clone in each SEACell
    :param ad: (AnnData) Annotated data matrix
    :param seacells: (pd.DataFrame) SEACells labels for each cell
    :return: (pd.DataFrame) Clonal purity of each SEACell
    """
    if 'clone' not in ad.obs:
        raise ValueError(
            'Clone labels not found in AnnData. Please ensure that clone labels are present in the obs dataframe')
    if 'SEACell' in ad.obs:
        ad.obs.drop('SEACell', axis=1, inplace=True)
    ad.obs = ad.obs.join(seacells, how='inner')
    # For each SEACell, compute the count of each clone
    clone_counts = ad.obs.groupby('SEACell').clone.value_counts().unstack().fillna(0)
    clone_counts = clone_counts.div(clone_counts.sum(axis=1), axis=0)
    return clone_counts


def run_aggregation_pipeline(ad, normal_genes, min_avg_cells_per_SEACell=20, min_avg_proportion_per_SEACell=0.05):
    """
    Run the aggregation pipeline for an experiment
    :param ad: (AnnData) Annotated data matrix
    :param normal_genes: (list) List of normal genes
    :param min_avg_cells_per_SEACell: (int) Minimum average number of cells per SEACell
    :param min_avg_proportion_per_SEACell: (float) Minimum average proportion of total cells per SEACell
    :return: (annData) Annotated data matrix with PCA embeddings, SEACells labels and clonal purity,
             (annData) SEACell aggregated anndata, with clone purity in obs and clone identity in obs
             (pd.DataFrame) Confusion matrix of clonal purity if clone labels are present in the data
    """
    assert 'raw' in ad.layers, 'Raw counts not found in AnnData. Please ensure that raw counts are stored in the raw layer'
    ad_normalized = compute_PCA_embeddings(ad, normal_genes)
    seacell_labels = compute_SEACells(ad_normalized,
                                      min_avg_cells_per_SEACell=min_avg_cells_per_SEACell,
                                      min_avg_proportion_per_SEACell=min_avg_proportion_per_SEACell)
    # Keep only the cells that are assigned to a SEACell
    ad = ad[seacell_labels.index]
    if 'clone' in ad.obs:
        clonal_purity_cm = compute_clonal_purity(ad, seacell_labels)
        clone_purity = pd.DataFrame(clonal_purity_cm.max(axis=1), columns=['clone_purity'])
        if 'clone_purity' in ad.obs:
            ad.obs.drop('clone_purity', axis=1, inplace=True)
        ad.obs = ad.obs.merge(clone_purity, left_on='SEACell', right_index=True)
    else:
        clonal_purity_cm = None
    SEACell_ad = SEACells.core.summarize_by_SEACell(ad, SEACells_label='SEACell', summarize_layer='raw')
    if 'clone' in ad.obs:
        SEACell_clone = pd.DataFrame(clonal_purity_cm.idxmax(axis=1), columns=['clone'])
        SEACell_ad.obs = SEACell_ad.obs.join(SEACell_clone, how='inner')
        SEACell_ad.obs['clone'] = SEACell_ad.obs['clone'].astype('category')
        SEACell_ad.obs = SEACell_ad.obs.join(clone_purity, how='inner')

    return ad, SEACell_ad, clonal_purity_cm