import time

import numpy as np
import pandas as pd
import torch

from .states import State, OurPoisson, Reset
from .dynamicHMM import HiddenMarkovModel as HMM
from . import utils
from . import optimize_likelihood as opt
from .logging_config import log

class cnvHMM:
    """
    Class for a CNV HMM model. This class contains the observations, metadata, and parameters for the model.
    Copy number predictions can be made after running .fit() on the model, by calling .predict().
    """
    def __init__(self,
                 observations,
                 observation_metadata,
                 n_states,
                 diploid_means,
                 clone_transitions = None,
                 celltype_prior_defaults={'alpha_mean': 1, 'alpha_std': 0.5, 'diploid_std': 0.1},
                 clone_prior_defaults={'epsilon': 0.01, 'offset': 0},
                 gexp_percentile_threshold=10,):
        """
        Initialize the CNV HMM model.
        :param observations: (pd.DataFrame) Observations of the cells. Rows are cells and columns are genes.
        :param observation_metadata: (pd.DataFrame) Metadata for the observations. Rows are cells and columns are
                                    metadata. Metadata includes celltype, clone, and normalization factors (used as
                                    modifiers for Poisson means)
        :param n_states: (int) Number of copy number states in the HMM - 0, 1, diploid, 3, 4, ...
        :param diploid_means: (pd.DataFrame) Diploid means for each gene in the observations. Rows are genes and columns
                                are celltypes.
        :param clone_transitions: (pd.DataFrame) Transition matrix for the clones. Rows are genes and columns are a
                                  MultiIndex with clones as the first level then (epsilon, diagonal_offset) as the second
                                  level. The second level encodes which transitions are likely at each time step (for
                                  each clone).

        :param celltype_prior_defaults: (dict) Default values for the prior distributions of the cluster parameters.
        :param clone_prior_defaults: (dict) Default values for the prior distributions of the clone parameters.
        :param gexp__percentile_threshold: (int) Percentile threshold for filtering lowly expressed genes. Default is 10.
        """
        observation_metadata = observation_metadata.copy()
        observations = observations.copy()
        diploid_means = diploid_means.copy()

        observation_metadata.columns = [col.lower() for col in observation_metadata.columns]
        assert 'celltype' in observation_metadata.columns, 'Metadata must contain a column for celltype'
        assert 'clone' in observation_metadata.columns, 'Metadata must contain a column for clone'
        assert 'normalization' in observation_metadata.columns, 'Metadata must contain a column for normalization'

        genes = diploid_means.index
        assert set(genes) == set(observations.columns), 'Genes in diploid_means must match genes in observations'

        cells = observations.index
        assert set(cells) == set(observation_metadata.index), ('Cells in observations must match cells in '
                                                               'observation_metadata')
        observation_metadata = observation_metadata.loc[cells]

        assert isinstance(n_states, int), 'n_states must be an integer'
        assert n_states > 0, 'n_states must be greater than 0'
        self.n_states = n_states

        # Check if gene expresion filtering threshold is an integer or float
        assert isinstance(gexp_percentile_threshold, (int, float)), (f'Gene expression filtering threshold must be an '
                                                                      f'integer or float '
                                                                      f'(not {type(gexp_percentile_threshold)})')
        self.leg_threshold = gexp_percentile_threshold

        celltypes = observation_metadata['celltype'].unique()
        assert set(celltypes) == set(diploid_means.columns), 'Celltypes in metadata must match columns in diploid_means'

        self.observations = observations.astype(int)
        self.metadata = observation_metadata
        self.diploid_means = diploid_means.astype(float)
        self.celltype_clone_pairs = {}

        self.clean_observations()

        if isinstance(clone_transitions, pd.DataFrame):
            clone_transitions = clone_transitions.copy()
        self.clone_transitions = clone_transitions
        self.clone_prior_defaults = clone_prior_defaults
        self.transition_matrix_per_clone = {}

        self.celltype_parameters = None
        self.celltype_prior_parameters = {}
        self.celltype_prior_defaults = celltype_prior_defaults
        for celltype in self.get_celltypes():
            defaults = celltype_prior_defaults.copy()
            defaults['diploid_mean'] = self.diploid_means[celltype]
            self.celltype_prior_parameters[celltype] = defaults

        self.verified = False
        self.HMMs = None

        self.observations_separated = None
        self.celltype_parameters_separated = None
        self.separated_chromosomes = False
        self.celltype_prior_distributions = None

        log.info('𓃢 Welcome to aeroCNV! 𓃢')
        return

    def modify_prior_default(self, parameter, value, clone_or_celltype=None):
        """
        Modify the prior distribution for a parameter in the model.
        :param parameter: (str) Parameter to modify
        :param value: (float) New value for the parameter
        :param clone_or_celltype: (str) Clone or celltype to modify the parameter for. If None, modify the parameter for
                                all clones or celltypes.
        :return:
        """
        assert not isinstance(value, str), 'Value must be a number or DataFrame, not a string.'
        if self.verified:
            raise ValueError('Cannot modify prior defaults after model has been verified.')

        if isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
            # Check that the index is the same as the observations
            assert set(value.index) == set(self.observations.columns), 'Index of value must match observations columns'

        if parameter in self.celltype_prior_defaults:
            if 'std' in parameter:
                assert value > 0, f'{parameter} must be greater than 0'
            if clone_or_celltype is not None:
                assert clone_or_celltype in self.get_celltypes(), f'{clone_or_celltype} not in celltypes'
                self.celltype_prior_parameters[clone_or_celltype][parameter] = value
            else:
                for celltype in self.get_celltypes():
                    self.celltype_prior_parameters[celltype][parameter] = value

            self.celltype_prior_defaults[parameter] = value

        elif parameter in self.clone_prior_defaults:
            if parameter == 'epsilon':
                assert value >= 0, f'{parameter} must be greater than or equal to 0'
                assert value <= 1, f'{parameter} must be less than or equal to 1'
            elif parameter == 'offset':
                assert value >= -self.n_states + 1, f'{parameter} must be greater than or equal to {-self.n_states + 1}'
                assert value <= self.n_states - 1, f'{parameter} must be less than or equal to {self.n_states - 1}'
                assert isinstance(value, int), f'{parameter} must be an integer'
            if clone_or_celltype is not None:
                raise NotImplementedError('Modifying clone parameters for a specific clone is not yet implemented')
            else:
                self.clone_prior_defaults[parameter] = value
        else:
            raise RuntimeError(f'Parameter {parameter} not in prior_defaults')

    def verify(self):
        """
        Verify all parameters of the model.
            - Filter genes in the observations and check that the cell type parameters are consistent
            - Verify that the clone transitions are (1) specified for all clones (2) specified for all genes and
                (3) have the correct constraints (e.g. epsilon >= 0, offset in [-n_states+1, n_states-1])
            - Verify the celltype parameters are (1) specified for all celltypes (2) specified for all genes and
                (3) have the correct constraints (e.g. alpha > 0, diploid_mean > 0)
            - Modifies self.verified to True if all parameters are verified, followed by creating the HMMs.
        :return: None
        """
        if self.verified:
            raise RuntimeError('Model has already been verified')

        self.verify_celltype_parameters()
        self.verify_transitions()
        self.verified = True
        return

    def clean_observations(self):
        """
        Clean the observations by removing lowly expressed genes and sorting the genes by chromosomal location.
        :return: None
        """
        observations = utils.filter_genes(self.observations, self.leg_threshold)
        genes_in_order, genes_not_found = utils.order_genes(observations.columns)
        self.discarded_genes = []
        if len(genes_not_found) > 0:
            log.info(f'Access gene information for {len(genes_not_found)} genes at .discarded_genes')
            self.discarded_genes = genes_not_found

        self.observations = observations[genes_in_order['Gene']]
        # Set the observations columns to a MultiIndex  containing the gene names and the chromosomal locations
        # with the chromosomal locations as the first level and the gene names as the second level
        self.observations.columns = pd.MultiIndex.from_frame(genes_in_order)
        # Check the column level names are 'Chromosome' and 'Gene'
        assert self.observations.columns.names == ['Chromosome', 'Gene'], (f'Observations column level names must be '
                                                                           f'Chromosome and Gene, not '
                                                                           f'{self.observations.columns.names}')
        self.metadata = self.metadata.loc[observations.index]

        # Ensure that the diploid means are in the same order as the observations
        self.diploid_means = self.diploid_means.loc[genes_in_order['Gene']]
        self.diploid_means.index = self.observations.columns

        # Extract celltype-clone pairs from the metadata
        self.celltype_clone_pairs = self.metadata.groupby(['celltype', 'clone']).indices

        # Ensure all expression values are positive
        assert self.observations.min().min() >= 0, 'Expression values must be non-negative'
        # Ensure the diploid means are positive
        assert self.diploid_means.min().min() >= 0, 'Diploid means must be non-negative'

        # Ensure neither contains NaN values
        assert not self.observations.isnull().values.any(), 'Observations must not contain NaN values'
        assert not self.diploid_means.isnull().values.any(), 'Diploid means must not contain NaN values'

        return

    def verify_celltype_parameters(self):
        """
        Verify that the celltype parameters are specified for all celltypes and that the alpha and diploid mean
        parameters are within the correct constraints. Also, check that the standard deviations are greater than 0.
        :return:
        """

        if self.celltype_parameters is None:
            self.celltype_parameters = {}
            for celltype in self.get_celltypes():
                prior_defaults = self.celltype_prior_parameters[celltype]
                prior_defaults['diploid_mean'] = self.diploid_means[celltype]
                self.celltype_parameters[celltype] = utils.generate_default_celltype_params(
                    self.get_chrs_and_genes(),
                    prior_defaults)
                self.celltype_prior_parameters[celltype] = utils.generate_default_celltype_priors(
                    self.get_chrs_and_genes(), prior_defaults)

            self.celltype_parameters = pd.concat(self.celltype_parameters, axis=1)
            self.celltype_prior_parameters = pd.concat(self.celltype_prior_parameters, axis=1)

        if len(self.celltype_parameters.index.names) == 1:
            log.debug('Celltype parameters index does not contain chromosome information; assuming genes only.')
            missing_genes = set(self.get_genes()) - set(self.celltype_parameters.index)
            assert len(missing_genes) == 0, (f'Celltype parameters must be specified for all genes; missing genes: '
                                             f'{missing_genes}')

            self.celltype_parameters = self.celltype_parameters.loc[self.get_genes()]
            self.celltype_parameters.index = self.get_chrs_and_genes()
            log.debug('Added chromosome information to celltype parameters index.')

        # Ensure that the celltype parameters genes are the same as the observations
        self.celltype_parameters = self.celltype_parameters.loc[self.get_chrs_and_genes()]

        # Check to make sure that the cluster parameters are specified for all genes
        assert set(self.celltype_parameters.index) == set(self.get_chrs_and_genes()), ('Celltype parameters must be '
                                                                                       'specified for all genes')
        # Check to make sure that the cluster parameters are specified for all celltypes
        celltypes = self.celltype_parameters.columns.get_level_values(0)
        missing_celltypes = set(self.get_celltypes()) - set(celltypes)
        assert set(celltypes) == set(self.get_celltypes()), (f'Cluster parameters must be specified for all celltypes;'
                                                             f' missing parameters for {missing_celltypes}')
        for celltype in self.get_celltypes():
            # Each celltype should have alpha and diploid mean parameters specified
            assert 'alpha' in self.celltype_parameters[celltype].columns, (f'alpha must be specified for celltype '
                                                                       f'{celltype}')
            assert 'diploid' in self.celltype_parameters[celltype].columns, (f'Diploid mean must be specified '
                                                                              f'for celltype {celltype}')
            # Ensure all the diploid means are positive (strictly)
            assert self.celltype_parameters[celltype]['diploid'].min() > 0, (f'Diploid mean must be > 0 for celltype'
                                                                             f' {celltype}')
        # Get all columns with std in name; check that the std is greater than 0
        std_cols = self.celltype_prior_parameters.columns.get_level_values(1).str.contains('std')
        std_cols = self.celltype_prior_parameters.columns[std_cols]
        assert self.celltype_prior_parameters.loc[:, std_cols].min().min() > 0, 'Standard deviations must be > 0'

        return

    def verify_transitions(self):
        """
        Verify that the clone transitions are specified for all clones and that the epsilon and offset parameters are
        within the correct constraints.
        :return: None
        """
        if self.clone_transitions is None:
            # Generate default clone transitions if not specified
            self.clone_transitions = {}
            for clone in self.get_clones():
                self.clone_transitions[clone] = utils.generate_default_transitions(self.get_chrs_and_genes(),
                                                                                   self.clone_prior_defaults['epsilon'],
                                                                                   self.clone_prior_defaults['offset'])
            self.clone_transitions = pd.concat(self.clone_transitions, axis=1)

        # Check if the clone transitions have both genes and chromosomes in the index, or just genes
        if len(self.clone_transitions.index.names) == 1:
            log.debug('Clone transitions index does not contain chromosome information; assuming genes only.')
            missing_genes = set(self.get_genes()) - set(self.clone_transitions.index)
            assert len(missing_genes) == 0, (f'Clone transitions must be specified for all genes; missing genes: '
                                             f'{missing_genes}')

            self.clone_transitions = self.clone_transitions.loc[self.get_genes()]
            self.clone_transitions.index = self.get_chrs_and_genes()
            log.debug('Added chromosome information to clone transitions index.')

        self.clone_transitions = self.clone_transitions.loc[self.get_chrs_and_genes()]

        # Check to make sure that the clone transitions are specified for all clones
        assert set(self.clone_transitions.columns.levels[0]) == set(self.get_clones()), ('Clone transitions must be '
                                                                                            'specified for all clones')
        assert set(self.clone_transitions.index) == set(self.get_chrs_and_genes()), ('Clone transitions must be specified '
                                                                                     'for all genes')
        for clone in self.get_clones():
            # Check to make sure that the epsilon and offset parameters are within the correct constraints and specified
            df = self.clone_transitions[clone]
            assert 'epsilon' in df.columns, f'Epsilon parameter must be specified for clone {clone}'
            assert 'offset' in df.columns, f'Offset parameter must be specified for clone {clone}'
            assert df['epsilon'].min() >= 0, f'Epsilon parameter must be greater than or equal to 0 for clone {clone}'
            assert df['offset'].min() >= -self.n_states + 1, (f'Offset parameter must be greater than or equal to '
                                                                f'{-self.n_states + 1} for clone {clone}')
            assert df['offset'].max() <= self.n_states - 1, (f'Offset parameter must be less than or equal to '
                                                                f'{self.n_states - 1} for clone {clone}')
        return

    def get_chrs_and_genes(self):
        """
        Get the chromosomes and genes in the observations.
        :return: (pd.MultiIndex) Chromosomes and genes in the observations.
        """
        return self.observations.columns

    def get_genes(self):
        """
        Get the genes in the observations.
        :return: (list) Genes in the observations.
        """
        genes = self.observations.columns.get_level_values(1)
        return list(genes)

    def get_n_genes(self):
        """
        Get the number of genes in the observations.
        :return: (int) Number of genes in the observations.
        """
        return len(self.get_genes())

    def get_n_chrs(self):
        """
        Get the number of chromosomes in the observations.
        :return: (int) Number of chromosomes in the observations.
        """
        return len(self.observations.columns.get_level_values(0).unique())

    def get_celltypes(self):
        """
        Get the celltypes in the observations.
        :return: (list) Celltypes in the observations.
        """
        return list(self.metadata['celltype'].unique())

    def get_clones(self):
        """
        Get the clones in the observations.
        :return: (list) Clones in the observations.
        """
        return list(self.metadata['clone'].unique())

    def separate_chromosomes(self):
        """
        Prevent transition information from propogating between chromosomes. In order to do this,
        we add a reset state to the HMMs that will be used to reset the transition probabilities between chromosomes.
        We need to:
        (1) Add the reset emissions to the expression data
        (2) Add the reset parameters to celltype_parameters before constructing the emission distributions
        (3) Construct a transition matrix that includes transitions to the reset state, and transitions to the Reset
            genes (done in _epsilon_to_matrices)
        (4) Add the reset state to the HMMs as the first state (done in create_HMM)
        (5) Create prior distributions over alpha and diploid means, including dummy Normals for the reset state
        :return:
        """
        if not self.verified:
            raise RuntimeError('Model must be verified before separating chromosomes')

        self.observations_separated = utils._separate_chromosomes(self.observations, -1)
        self.celltype_parameters_separated = utils._separate_chromosomes(self.celltype_parameters.T, 1).T
        self.celltype_prior_parameters_separated = utils._separate_chromosomes(self.celltype_prior_parameters.T, 1).T
        self.separated_chromosomes = True

    def create_HMMs(self):
        """
        Create HMMs for each celltype-clone pair in the model.
        :return:
        """
        if not self.separated_chromosomes:
            raise RuntimeError('Model must separate chromosomes before creating HMM(s).')
        self.HMMs = []
        for (celltype, clone) in self.celltype_clone_pairs:
            name = f'HMM_{celltype}_{clone}'
            hmm = self.create_HMM(name, celltype, clone)
            self.HMMs.append(hmm)
        log.debug(f'Created {len(self.HMMs)} HMM(s).')
        return

    def create_HMM(self, name, celltype, clone):
        """
        Create an HMM for a specific celltype-clone pair. Initialize the HMM with the name and add the states for that
        celltype, the transitions for that clone, and the training data for that celltype-clone pair.
        :param name: (str) Name of the HMM
        :param celltype: (str) Celltype to create the HMM for
        :param clone: (str) Clone to create the HMM for
        :return:
        """
        # Add training data to the model
        relevant_cells_ix = self.celltype_clone_pairs[(celltype, clone)]
        observations = self.observations_separated.iloc[relevant_cells_ix]
        modifiers = self.metadata.loc[observations.index]['normalization']

        hmm = HMM(observations, modifiers, name=name)
        # Create Poisson states for the HMM
        states = [State(Reset(-1), name='Reset')] + self._create_states(celltype)
        hmm.add_states(states)
        # Add the transition matrix as well (constructed from the epsilon weights)
        transition_matrix_separated = self.get_transitions_matrix(clone)
        hmm.add_transition_matrix(transition_matrix_separated)
        hmm.verify()
        log.debug(f'Created HMM for {celltype}, clone {clone}.')
        return hmm

    def _create_states(self, celltype):
        """
        Create states for the HMMs. The states are Poisson distributions with means defined by the cluster parameters.
        For state with copy number, k, the (per-gene) mean of the Poisson distribution is defined as:
        mean = (k/2)^alpha * diploid_mean
        :return:
        """
        alpha_weights = self.celltype_parameters_separated[celltype]['alpha'].values.reshape(1, -1)
        diploid_mean = self.celltype_parameters_separated[celltype]['diploid'].values.reshape(1, -1)
        n_features = self.observations_separated.shape[1]
        assert diploid_mean.shape[1] == n_features, (f'Diploid mean must be of the same length '
                                                       f'({diploid_mean.shape[1]}) as the number of genes '
                                                       f'({n_features}) in separated observations.')
        assert alpha_weights.shape[1] == n_features, (f'Alpha weights must be of the same length '
                                                        f'({alpha_weights.shape[1]}) as the number of genes '
                                                        f'({n_features}) in separated observations.')
        states = []
        for state in range(self.n_states):
            if state == 0:
                state_mu = 0.0001
            else:
                state_mu = state
            mean = np.multiply(np.power(state_mu / 2., alpha_weights), diploid_mean)[0]
            if not torch.is_tensor(mean):
                mean = torch.DoubleTensor(mean)
            distribution = OurPoisson(mean)
            states.append(State(distribution, name='State_{0}'.format(state)))
        return states

    def _epsilon_to_matrices(self, clone):
        """
        Convert the epsilon and offset parameters for a clone into a transition matrix.
        :param clone: (str) Clone to convert the parameters for.
        :return: (torch.Tensor of shape n_genes-1+n_chrs x n_states x n_states) Transition matrix for the clone separated
                        by chromosome
        """
        df = self.clone_transitions[clone]
        chrs = df.index.get_level_values('Chromosome').unique()

        # We defined the canonical matrix for transition from any state to the reset state
        transition_to_reset_matrix = torch.zeros((self.n_states+1, self.n_states+1))
        transition_to_reset_matrix[:, 0] = 1

        # We defined the canonical matrix for transitioning from the reset state to any other state with uniform probability
        transition_from_reset_matrix = torch.zeros((self.n_states+1, self.n_states+1))
        transition_from_reset_matrix[0, 1:] = 1/(self.n_states)

        transition_matrices = [transition_from_reset_matrix]
        for chr in chrs:
            # Convert each row into a transition matrix
            df_chr = df.xs(chr, level='Chromosome')
            for gene, row in df_chr.iterrows():
                epsilon = row['epsilon']
                offset = int(row['offset'])  # Ensure offset is an integer
                matrix = utils.create_transition_matrix(self.n_states, epsilon, offset)
                # Pad the matrix to account for the reset state (which is the first state)
                matrix = utils._pad_transition_matrix(matrix, self.n_states)
                transition_matrices.append(matrix)
            # Convert list of matrices into a single tensor, ignoring the last gene (no transitions after the last gene)
            transition_matrices = transition_matrices[:-1]
            # Add the transition matrix from the last gene to the reset state
            transition_matrices.append(transition_to_reset_matrix)
            # Add the transition matrix from the reset state to the first gene on the next chromosome
            # We will use a uniform prior over all (non-Reset) states here.
            transition_matrices.append(transition_from_reset_matrix)

        # For the last chromosome, there is no final reset state, and no transitions to the next chromosome, so we
        # need to remove the last two matrices
        transition_matrices = transition_matrices[:-2]
        # Stack all transition matrices to create a torch tensor
        matrices = torch.stack(transition_matrices)

        assert matrices.shape == (self.get_n_chrs() + self.get_n_genes() - 1, self.n_states + 1, self.n_states + 1), \
            (f'Transition matrices must have shape '
             f'{(self.get_n_chrs() + self.get_n_genes() - 1, self.n_states + 1, self.n_states + 1)}')
        return matrices

    def get_transitions_matrix(self, clone):
        """
        Get the transition matrix for a specific clone.
        :param clone: (str) Clone to get the transition matrix for.
        :return: (torch.Tensor) Transition matrix for the clone.
        """
        assert clone in self.get_clones(), f'Clone {clone} not found in model.'
        if clone not in self.transition_matrix_per_clone:
            self.transition_matrix_per_clone[clone] = self._epsilon_to_matrices(clone)
        return self.transition_matrix_per_clone[clone]
    def __update_transitions(self):
        """
        Update the transition probabilities for the model.
        :return:
        """
        # Rest the transition matrix per clone since we are updating it
        self.transition_matrix_per_clone = {}
        updated_clone_transitions = {}
        clone_expected_transitions = {}

        for hmm in self.HMMs:
            F, B, G = self.matrices[hmm.name]
            celltype, clone = hmm.name.split('_')[1:]
            xi = hmm.expected_transitions(F, B)
            clone_expected_transitions[clone] = clone_expected_transitions.get(clone, []) + [xi]
            clone_expected_transitions[clone] = clone_expected_transitions.get(clone, []) + [xi]

        # Determine which gene is the last gene before a RESET gene; these are a special case where the transition matrix
        # is set to transition to the Reset state (first state) with probability 1
        gene_names = self.observations_separated.columns.get_level_values(1)[:-1]
        reset_gene_ix = np.where(gene_names.str.contains('RESET'))[0]
        pre_reset_gene_ix = reset_gene_ix - 1
        pre_reset_gene_ix = pre_reset_gene_ix[pre_reset_gene_ix >= 0]
        pre_reset_genes = gene_names[pre_reset_gene_ix]

        # Now we will construct the updated transition matrices for each clone
        transition_to_reset_matrix = torch.zeros((self.n_states + 1, self.n_states + 1))
        transition_to_reset_matrix[:, 0] = 1

        for clone in clone_expected_transitions:
            expected_transitions = torch.stack(clone_expected_transitions[clone], dim=0).sum(axis=0)
            main_transition = self.clone_transitions[clone]['offset']

            clone_transitions = {'Chromosome': [], 'Gene': [], 'epsilon': [], 'offset': []}
            updated_matrices = []
            for i, (chr, gene) in enumerate(self.observations_separated.columns[:-1]):
                # We add a small pseudo-count so that we don't divide by zero
                T = expected_transitions[i] + 1e-15
                # We can determine the updated epsilon by counting the expected transitions off the main diagonal and dividing by the total number of transitions
                if 'RESET' not in gene:
                    main_transition = self.clone_transitions[clone]['offset'].loc[(chr, gene)]
                    total_counts = T.sum()
                    epsilon = utils._zero_out_diagonal_offsets(T, main_transition).sum() / total_counts
                    clone_transitions['Chromosome'].append(chr)
                    clone_transitions['Gene'].append(gene)
                    clone_transitions['epsilon'].append(epsilon)
                    clone_transitions['offset'].append(main_transition)

                    if gene in pre_reset_genes:
                        updated_matrices.append(transition_to_reset_matrix)
                    else:
                        updated_T = utils.create_transition_matrix(self.n_states, epsilon, main_transition)
                        updated_T = utils._pad_transition_matrix(updated_T, self.n_states)
                        updated_matrices.append(updated_T)
                else:
                    # We don't compute an epsilon for the reset gene; but we want to normalize the expected transitions to get an updated start distribution over other states
                    updated_matrices.append(T / T.sum(axis=1, keepdim=True))
            # Add a dummy epsilon and offset for the last gene to maintain consistency
            (chr, gene) = self.observations_separated.columns[-1]
            clone_transitions['Chromosome'].append(chr)
            clone_transitions['Gene'].append(gene)
            clone_transitions['epsilon'].append(torch.tensor(0))
            clone_transitions['offset'].append(0)

            clone_transitions['epsilon'] = torch.stack(clone_transitions['epsilon']).numpy()
            clone_transitions = pd.DataFrame(clone_transitions).set_index(['Chromosome', 'Gene'])

            assert set(clone_transitions.index) == set(
                self.observations.columns), 'Epsilon values are not computed for all genes.'
            assert len(updated_matrices) == len(
                self.observations_separated.columns) - 1, 'Transition matrices are not computed for all genes.'

            updated_clone_transitions[clone] = clone_transitions
            updated_matrices = torch.stack(updated_matrices)
            self.transition_matrix_per_clone[clone] = updated_matrices

        updated_clone_transitions = pd.concat(updated_clone_transitions, axis=1)
        self.clone_transitions = updated_clone_transitions
        return

    def __update_emissions(self,):

        celltype_gammas = {}
        celltype_modifiers = {}
        celltype_observations = {}
        for hmm in self.HMMs:
            celltype, clone = hmm.name.split('_')[1:]
            F, B, gamma = self.matrices[hmm.name]
            celltype_gammas[celltype] = celltype_gammas.get(celltype, []) + [gamma]
            celltype_modifiers[celltype] = celltype_modifiers.get(celltype, []) + [hmm.modifiers]
            celltype_observations[celltype] = celltype_observations.get(celltype, []) + [hmm.training_data]

            celltype_gammas[celltype] = celltype_gammas.get(celltype, []) + [gamma]
            celltype_modifiers[celltype] = celltype_modifiers.get(celltype, []) + [hmm.modifiers]
            celltype_observations[celltype] = celltype_observations.get(celltype, []) + [hmm.training_data]

        for celltype in celltype_gammas:
            gammas = torch.cat(celltype_gammas[celltype])
            modifiers = torch.cat(celltype_modifiers[celltype])
            data = torch.cat(celltype_observations[celltype])

            # Update the emission parameters diploid_mean and alpha. We need to optimize the NLL analytically with respect to alpha
            # and diploid
            summaries = opt.compute_Poisson_summaries(data, gammas, modifiers)
            alphas = self.celltype_parameters_separated[celltype]['alpha']
            diploid_means = self.celltype_prior_parameters_separated[celltype]['diploid_mean']
            prior_distributions = self.celltype_prior_parameters_separated[celltype]

            updated_alphas = opt.update_alphas(alphas, diploid_means, summaries, prior_distributions)
            self.celltype_parameters_separated.loc[:, (celltype, 'alpha')] = updated_alphas

        self.celltype_parameters = self.celltype_parameters_separated.loc[self.celltype_parameters.index]
        self.verify_celltype_parameters()

    def converged(self):
        return False
        raise NotImplementedError

    def fit(self, min_iters=10, max_iters=100, freeze_transitions=False, freeze_emissions=False):
        """
        Fit the model to the data.
        :param min_iters: (int) Minimum number of iterations to run before checking for convergence. Default is 10.
        :param max_iters: (int) Maximum number of iterations to run before stopping. Default is None.
        :param freeze_transitions: (bool) If True, do not update transition probabilities. Default is False.
        :param freeze_emissions: (bool) If True, do not update emission probabilities. Default is False.
        :return: None
        """

        if not self.HMMs == 0:
            if not self.separated_chromosomes:
                self.separate_chromosomes()
            self.create_HMMs()

        log.info('Fitting to data.')
        iters = 0
        start_time = time.time()
        while iters <= max_iters:
            log.debug(f'Beginning iteration: {iters}')
            iters += 1
            if iters % 5 == 0:
                log.info(f'Iteration {iters} after {(time.time() - start_time):.2f} seconds')

            self.matrices = {}
            for hmm in self.HMMs:
                F = hmm.forward()
                B = hmm.backward()
                G = hmm.gamma(F,B)
                self.matrices[hmm.name] = (F, B, G)

            if not freeze_transitions:
                self.__update_transitions()
            if not freeze_emissions:
                self.__update_emissions()

            # Create new HMMs with updated parameters
            self.create_HMMs()

            # Check for convergence
            if self.converged() and iters > min_iters:
                log.info(f'Model has converged after {iters} iterations and {(time.time() - start_time):2f} seconds')
                return

        log.warning(f'Model has not converged after {iters-1} iterations.')
        log.warning(f'Terminating training due to max_iters reached at {max_iters} after '
                    f'{(time.time() - start_time):.2f} seconds')
        return

    def predict(self):
        """
        Predict the most likely state sequence for each cell in the model.
        :return: (pd.DataFrame) Predicted state sequences for each cell.
        """
        if not self.HMMs:
            raise RuntimeError('Model must be fit before predicting state sequences.')
        predictions = []
        for hmm in self.HMMs:
            paths = hmm.viterbi()[1]
            paths = pd.DataFrame(paths, index=hmm.cell_ids,
                                 columns=self.observations_separated.columns)
            paths = paths[self.observations.columns]
            predictions.append(paths)
        predictions = pd.concat(predictions, axis=0)

        # Reorder the predictions to match the original order of the observations
        predictions = predictions.loc[self.metadata.index]
        return predictions