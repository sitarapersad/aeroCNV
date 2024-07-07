import scanpy as sc
import pandas as pd
import numpy as np

from .logging_config import log
from .preprocess import normalize_inputs
from .cnvHMM import cnvHMM

from picasso import encode_cnvs_as_ternary, Picasso
def run_aero(ad,
             n_states,
             diploid_means,
             output_directory,
             max_HMM_iters=200,
             max_tree_depth=None,
             min_cells_per_clone=5, ):
    """

    :param ad: (AnnData) Annotated data matrix containing expression data as raw counts stored in the .layers['counts'].
    This can be either the single-cell expression matrix or the aggregated expression matrix.
    (Run .agg.run_aggregation_pipeline() to aggregate the single-cell expression matrix)
    :param n_states: (int) Number of states in the HMM
    :param diploid_means: (pd.Series) Series containing the mean expression of each gene in the diploid state
    :param output_directory: (str) Directory to save the output files
    :param max_HMM_iters: (int) Maximum number of iterations to run the HMM
    :param max_tree_depth: (int) Maximum depth of the clone tree
    :param min_cells_per_clone: (int) Minimum number of cells per clone

    :return:
    """
    assert 'counts' in ad.layers, 'Expression matrix must contain raw counts in the "counts" layer'
    log.info(f'Initial expression matrix contains {ad.shape[0]} cells and {ad.shape[1]} genes')
    log.info(f'Initial diploid means contains {len(diploid_means)} genes')
    # Filter out cells with expression in less than 10% of genes
    sc.pp.filter_cells(ad, min_genes=ad.shape[1] * 0.1)
    # Filter out genes expressed in less than 5% of cells
    sc.pp.filter_genes(ad, min_cells=ad.shape[0] * 0.05)
    log.info(f'Filtered expression matrix contains {ad.shape[0]} cells and {ad.shape[1]} genes')

    # Remove any genes with 0 mean expression
    diploid_means = diploid_means[diploid_means > 0]
    log.info(f'Using {len(diploid_means)} genes for the analysis after filtering out genes with 0 mean expression')

    # Keep only genes in both the diploid and expression data
    common_genes = ad.var_names.intersection(diploid_means.index)
    log.info(f'Keeping {len(common_genes)} genes common to both the diploid and expression data')
    ad = ad[:, common_genes]
    diploid_means = diploid_means.loc[common_genes]

    expression = ad.to_df(layer='counts')
    # Compute cell metadata
    cell_metadata, diploid_ref = normalize_inputs(ad, diploid_means, flavour='total_reads')
    # If .obs contains 'clone' or '.cell_type', use this as the cell type
    cell_metadata['celltype'] = ad.obs['celltype'] if 'celltype' in ad.obs else 'UnknownCellType'
    cell_metadata['clone'] = ad.obs['clone'] if 'clone' in ad.obs else 'UnknownClone'

    diploid_mean = pd.DataFrame(columns=cell_metadata['celltype'].unique(), index=expression.columns)
    for celltype in cell_metadata['celltype'].unique():
        diploid_mean.loc[:, celltype] = diploid_ref.copy()

    model = cnvHMM(expression=expression,
                   observation_metadata=cell_metadata,
                   n_states=n_states,
                   diploid_means=diploid_mean)
    model.verify()
    model.fit(freeze_emissions=True, max_iters=max_HMM_iters)
    viterbi = model.predict()

    iterate = True
    while iterate:
        # Run picasso to infer the clone profiles
        log.info('Running Picasso to infer clone profiles')
        cm = encode_cnvs_as_ternary(viterbi)
        log.debug('Encoding CNVs as ternary matrix')
        character_matrix = character_matrix.loc[:,
                           (character_matrix.values == character_matrix.mode(axis=0).values).mean(axis=0) < 0.99]
        log.debug('Filtered out uninformative alterations genes with more than 99% agreement between cells')
        tree_model = Picasso(cm,
                             min_depth=1,
                             max_depth=2,
                             min_clone_size=5,
                             terminate_by='bic',
                             assignment_confidence_threshold=0.8,
                             assignment_confidence_proportion=0.9)
        tree_model.fit()
        clone_assignments = tree_model.get_clone_assignments()
        log.info(f'Estimated {len(clone_assignments)} clones.')

        # Re-run cnvHMM to infer the CNV states given new clone profiles
        cell_metadata['clone'] = clone_assignments
        model = cnvHMM(expression=expression,
                       observation_metadata=cell_metadata,
                       n_states=n_states,
                       diploid_means=diploid_mean)
        model.verify()
        model.fit(freeze_emissions=True, max_iters=max_HMM_iters)
        viterbi = model.predict()

        # Define priors over transitions
        phylogenetic_transition_priors = model.infer_transition_priors()

        # Run cnvHMM to infer cleaned Viterbi paths

        pass
    return ad

