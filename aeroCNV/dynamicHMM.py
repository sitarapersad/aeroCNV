import numpy as np
import pandas as pd
import torch

from .states import State, Reset, OurPoisson

import logging as log
log.getLogger().addHandler(log.StreamHandler())

from icecream import ic
class HiddenMarkovModel(object):
    """
    Implements a HiddenMarkovModel for inferring Copy Number Variant (CNV) from scRNA-seq data.

    This model implements a dynamic HMM where:
    (1) transition probabilities are learned from data and are independent at each gene-gene transition
    (2) emission probabilities are modeled by a Poisson distribution and are specific to each gene
    """

    def __init__(self, data, modifiers, name='HMM'):
        """
        Initialize a HiddenMarkovModel object for inferring CNV from scRNA-seq data.
        :param data: (pd.DataFrame) Expression counts (as integers) for each gene in each cell.
                        Rows are cells, columns are genes.
        :param modifiers: (pd.Series) Modifier values for each cell. Rows are cells, values are modifiers.
        :param name: (str) Name of the model
        """
        self.name = name

        assert isinstance(data, pd.DataFrame), 'Training data must be a DataFrame, not {0}'.format(type(data))
        assert isinstance(modifiers, pd.Series), 'Modifiers must be a Series, not {0}'.format(type(modifiers))
        try:
            modifiers = modifiers.loc[data.index]
        except KeyError:
            raise KeyError('Modifiers must have the same index as the training data.')

        # Check that the data are integers
        try:
            data = data.astype(int)
        except ValueError:
            raise ValueError('Data must be integers.')

        # Check that all values are non-negative or -1
        assert (data.values >= -1).all(), 'Data must be non-negative or -1.'

        # Initialize the model with metadata inferred from the observed data
        self.cell_ids = data.index
        self.n_cells = len(self.cell_ids)

        # Now we convert the training data and modifiers into tensors
        data = torch.tensor(data.values)
        n_samples, n_features = data.shape
        data = data.view(n_samples, 1, n_features)

        self.training_data = data
        self.modifiers = torch.tensor(modifiers.values).view(-1, 1)

        # There is 1 fewer steps than there are genes, since we are modeling transitions between genes
        self.n_features = n_features
        self.n_steps = n_features - 1

        # We track whether the model has been verified; once the model is verified, no further changes can be made
        self.model_verified = False

        self.states = []
        self.n_states = 0

        # Prior distribution over all states; this can only be set once all states are added
        self.end_distribution = None

        # Define transition matrix, keep track of whether transitions have already been added.
        # Once transitions are added, no changes can be made to the states in the model.
        self.__transition_matrix = None
        return

    def add_states(self, states):
        """
        Add states to the model.
        :param states: (list) List of State objects
        """
        for state in states:
            self.add_state(state)
        return

    def add_state(self, state):
        """
        Add a state to the model. Checks if (1) the model is already verified and (2) state is valid
        :param state: (State) State object to add to the model
        :return:
        """
        if self.model_verified:
            raise Exception(
                'Model has already been defined. No further changes can be made after calling model.verify()')

        # assert isinstance of State
        if not isinstance(state, State):
            raise TypeError('states must be a State instance.')

        if state.name is None:
            log.debug('No name specified for added state. Creating state with name: State_{0}'.format(self.n_states))
            state.name = 'State_{0}'.format(self.n_states)

        # Verify that the dimension of the state's parameters matches the number of features in the data

        if state in self.states:
            return

        self.states.append(state)
        self.n_states += 1
        return

    def get_states(self):
        """
        Return list of states in the model.
        :return: (list) List of State objects
        """
        return [State(distribution=state.distribution, name=state.name) for state in self.states]

    def clear_states(self):
        '''
        Remove all states from the model. This is useful when updating the model with new states.
        '''
        self.model_verified = False
        self.states = []
        self.n_states = 0
        return

    def clear_transitions(self):
        '''
        Remove transition matrix from the model. This is useful when updating the model with new transitions.
        '''
        self.model_verified = False
        self.__transition_matrix = None
        self.epsilon = None
        return

    def clear_model(self):
        """
        Clear out the model to its initial state. This is useful when updating the model with new states and transitions.
        :return: None
        """
        self.model_verified = False
        self.clear_states()
        self.clear_transitions()
        self.training_data = None
        self.modifiers = None
        self.cell_ids = None
        self.n_cells = None
        return

    def add_transition_matrix(self, matrix):
        '''
        Transition matrix is time-dependent; has shape num_states x num_states x num_steps. The matrix m[t, i,j] is the
        probability of transitioning from state s_i to s_j at step t->t+1.

        @param: matrix (np.array, torch.tensor) - Transition matrix of shape num_steps x num_states x num_states
        @return: None
        '''
        if self.model_verified:
            log.error('Transition cannot be added after model is already verified')
            raise RuntimeError('Transitions cannot be added after model is already verified')

        assert len(matrix.shape) == 3, ('Transition matrix must have shape num_steps x num_states x num_states, '
                                        'not {0}').format(matrix.shape)

        if isinstance(matrix, np.ndarray):
            matrix = torch.from_numpy(matrix)

        # Transitions may be dependent on which step we are at
        n_transitions, n1, n2 = matrix.shape
        assert n2 == n1, f'Transition matrix at each time step must be square, not {n1} x {n2}'
        assert n1 == self.n_states, f'Transition matrix must have as many rows ({n1}) as states ({self.n_states})'
        matrix = matrix.double()

        self.n_steps = n_transitions

        # If any rows have no counts, use a uniform distribution
        if (matrix.sum(2)==0).any():
            iix = np.where(matrix.sum(2)==0)
            matrix[iix] = 1

        # Normalize matrix to ensure rows sum to 1
        matrix_norm = matrix/matrix.sum(2, keepdim=True)

        assert not torch.isnan(matrix_norm).any(), 'Transition matrix must not contain NaNs'
        self.__transition_matrix = matrix_norm
        return None

    def get_transition_matrix(self):
        """
        Return the transition matrix of the model.
        :return: (Tensor) Transition matrix of shape num_steps x num_states x num_states
        """
        return self.__transition_matrix.clone()

    def verify(self):
        """
        Verify the model. Once the model is verified, no further changes can be made.
        The method checks that the training data has been added, and that the
        number of genes in the training data matches the number of steps in the model.

        """
        self.start_distribution = torch.DoubleTensor([1] + [0]*(self.n_states-1)).view(-1,1)
        self.end_distribution = torch.DoubleTensor([1]*self.n_states).view(-1,1)

        assert self.__transition_matrix is not None, 'Training data must be added to the model before verifying.'

        # Ensure that a transition matrix has been added and is specified for all but the last gene
        assert self.__transition_matrix is not None, 'Transition matrix must be added to the model before verifying.'
        n_steps, n_states1, n_states2 = self.__transition_matrix.shape
        assert n_states1 == n_states2, 'Transition matrix must be square.'
        assert n_states1 == self.n_states, 'Transition matrix must have as many rows as there are states.'
        assert n_steps == self.n_steps, 'Transition matrix must have as many rows as there are steps.'

        assert len(self.modifiers) == self.n_cells, ('Number of modifiers must match the number of cells in the training '
                                                     'data.')
        assert self.n_states > 0, 'Model must have at least one state.'

        for state in self.states:
            if isinstance(state.distribution, Reset):
                assert state.distribution.length == 1, 'Reset state must have length 1.'
            else:
                assert state.distribution.length == self.n_features, (f'State length ({state.distribution.length}) must '
                                                                      f'match the number of features in the data '
                                                                      f'({self.n_features}).')

        # We can compute now emission likelihoods for each state at each time step for the training data
        self._compute_emission_loglikelihoods()
        self.model_verified = True
        return

    def _compute_emission_loglikelihoods(self):
        """
        Compute the log likelihood of the training data under each state at each time step.
        """
        lls = []
        for state in self.states:
            if isinstance(state.distribution, Reset):
                ll = state.distribution.log_prob(self.training_data).view(self.n_cells, -1)
            else:
                ll = state.distribution.log_prob(self.training_data, modifiers=self.modifiers).view(self.n_cells,-1)
            lls.append(ll)
        emission_loglik = torch.cat(lls, 1).view(self.n_cells, self.n_states, self.n_features)
        assert not torch.isnan(emission_loglik).any(), 'emission log likelihoods contains NaNs.'
        self.__emissions = emission_loglik
        return
    
    def get_emission_loglikelihoods(self):
        """
        Return the log likelihood of the training data under each state at each time step.
        :return: (Tensor) Log likelihood of the training data under each state at each time step
        """
        return self.__emissions

    def viterbi(self):
        """
        Implements the Viterbi algorithm for decoding the most likely sequence of hidden states given the observed data.
        :return: (Tensor) Viterbi matrix of shape n_cells x n_states x n_observations, (Tensor) Best path of states
        """
        if not self.model_verified:
            raise RuntimeError('Model must be verified before calling viterbi.')

        emission_loglik = self.__emissions
        n_cells, n_states, n_observations = emission_loglik.shape
        assert n_cells == self.n_cells, f'Emission log likelihoods must have as many rows ({n_cells}) as there are cells ({self.n_cells})'
        assert n_states == self.n_states, f'Emission log likelihoods must have as many columns ({n_states}) as there are states ({self.n_states})'

        transition_matrix = self.get_transition_matrix()
        # Initialize empty Viterbi matrix with initial state distribution
        VITERBI = torch.empty((n_cells, n_states, n_observations), dtype=torch.double)
        VITERBI = torch.cat((torch.log(self.start_distribution).repeat(n_cells, 1, 1), VITERBI), 2)
        # Define traceback matrix to track optimal path
        TB = -1 * torch.ones((n_cells, n_states, n_observations))

        # pad the matrix with an identity matrix to represent the transition from initial state distribution (t=-1 -> 0)
        # and the transition from final observation to terminal distribution
        identity = torch.eye(n_states, dtype=torch.double).view(-1, n_states, n_states)
        transition_matrix = torch.cat((identity, transition_matrix), 0)

        # Iteration over each step in observation matrix (column)
        for observation in range(n_observations):
            # Iteration over each state (row)
            for state in range(n_states):
                # Collect all values of "max_{j in States} (V[j,i] + transition_logliks[j,state,observation]" for state
                transition_scores = torch.add(VITERBI[:, :, observation],
                                              torch.log(transition_matrix[observation, :, state]))
                # Determine the best score and best preceding state. Store the best preceding state in a matrix for traceback at the end.
                best_preceding_V, best_preceding_state = transition_scores.max(1)
                VITERBI[:, state, observation + 1] = emission_loglik[:, state, observation] + best_preceding_V
                TB[:, state, observation] = best_preceding_state

        # Remove dummy first column of initial probabilities
        VITERBI = VITERBI[:, :, 1:]
        assert VITERBI.shape == (n_cells, n_states, n_observations), ('Viterbi matrix is of incorrect '
                                                                      'shape: {0}').format(VITERBI.shape)

        try:
            assert not torch.isnan(VITERBI).any(), 'VITERBI contains NaNs.'
        except AssertionError:
            log.error('VITERBI contains NaNs')
            log.error('VITERBI is nan at:')
            log.error(torch.isnan(VITERBI))
            log.error(VITERBI[0])
            log.error(VITERBI.shape)
        # Do traceback to determine indices of states in traceback.
        final_score, final_state = VITERBI[:, :, -1].max(1)

        # -- Traceback to compute best path
        final_state = final_state.view(-1, 1).long()
        path = [final_state]
        for observation in reversed(range(1, n_observations)):
            final_state = TB[:, :, observation].gather(1, final_state).long()

            path.append(final_state)

        best_path = torch.cat(path[::-1], 1)
        best_path -= 1  # Subtract 1 to account for the Reset State
        return VITERBI, best_path

    def forward(self):
        """
        Perform the forward algorithm to compute the probability of the given sequence of observations.

        Mathematical Definition:
        The forward algorithm calculates the probability of an observed sequence, O = {O1, O2, ..., OT}, given the HMM
        parameters.

        Let:
        - N be the number of states in the HMM.
        - T be the length of the observation sequence.
        - A = {a_ij} be the state transition probability matrix, where a_ij = P(S_{t+1} = j | S_t = i).
        - B = {b_jk} be the emission probability matrix, where b_jk = P(O_t = k | S_t = j).
        - π = {π_i} be the initial state distribution, where π_i = P(S_1 = i).
        - α_t(i) be the forward probability at time t for state i, i.e., the probability of the partial observation
        sequence up to time t, ending in state i.

        Initialization:
        α_1(i) = π_i * b_i(O_1), for 1 ≤ i ≤ N

        Induction:
        α_t(j) = [ Σ_{i=1}^N α_{t-1}(i) * a_ij ] * b_j(O_t), for 2 ≤ t ≤ T and 1 ≤ j ≤ N

        Termination:
        P(O | λ) = Σ_{i=1}^N α_T(i)

        Where:
        - α_t(i) is the probability of observing the sequence up to time t and being in state i.
        - Σ denotes the summation over all states.


        WE CAN'T USE LOG TRICK ON SUMMATION, TRY NUMERICALLY STABLE VERSION:
        http://users-cs.au.dk/cstorm/courses/PRiB_f12/slides/hidden-markov-models-2.pdf
        Forward algorithm using scaled values

        :return: (Tensor) Forward matrix of shape n_cells x n_states x n_observations
        """
        if not self.model_verified:
            raise RuntimeError('Model must be verified before calling forward.')

        emission_loglik = self.__emissions

        n_cells, n_states, n_features = emission_loglik.shape
        assert n_cells == self.n_cells, f'Emission log likelihoods must have as many rows ({n_cells}) as there are cells ({self.n_cells})'
        assert n_states == self.n_states, f'Emission log likelihoods must have as many columns ({n_states}) as there are states ({self.n_states})'
        transition_matrix = self.get_transition_matrix()

        # Initialize empty Forward variables matrix.
        FORWARD = -torch.ones((n_cells, self.n_states, n_features-1), dtype=torch.double)
        # We initialize the first column with the product of the start distribution and the emission probabilities
        # at the first time step
        start_distribution = self.start_distribution.repeat(n_cells, 1, 1)
        initial_forward_probs = (start_distribution.squeeze(-1) * torch.exp(emission_loglik[:, :, 0])).unsqueeze(-1)
        FORWARD = torch.cat((initial_forward_probs, FORWARD), 2)

        emission_loglik = emission_loglik[:, :, 1:]
        emission_probs = torch.exp(emission_loglik - emission_loglik.max(1, keepdim=True)[0])

        scale_factors = []
        # Starting from time t=1, we compute the forward recursion as the product of the emission probabilities at time,
        # t and the sum of the product of the forward probabilities at time t-1 and the transition probabilities from
        # state i to state j at time t.
        for observation in range(self.n_steps):
            for state in range(self.n_states):
                # Collect all values of "sum_{j in States} FORWARD[:,j,t] x transition_probs[j,state,t]" for state
                transition_scores = torch.mul(FORWARD[:, :, observation], transition_matrix[observation, :, state])
                FORWARD[:, state, observation + 1] = emission_probs[:, state, observation] * transition_scores.sum(1)

            scale_factor = FORWARD[:, :, observation + 1].sum(1, keepdim=True)
            FORWARD[:, :, observation + 1] /= scale_factor
            scale_factors.append(scale_factor)

        # Use the original scaling factor to recover to original forward matrix
        # We have that F_scaled[sample, state, time] = \prod_1^time 1/scale_factor[t] x F[sample, state, time]
        # Converting to the log space allows us to recover the original F[sample, state, time]
        # log F(state, t) = log F_scale(state, t) + sum_{j=1^t} log scale_factor[j]

        scale_factors = torch.cat(scale_factors, dim=1).view(n_cells, 1, self.n_steps).repeat(1, self.n_states, 1)
        norm = torch.log(scale_factors).cumsum(2)
        FORWARD = torch.log(FORWARD)
        # Ignore start distribution
        FORWARD[:, :, 1:] += norm

        # Add the emission normalization factor
        FORWARD[:, :, 1:] += emission_loglik.max(1, keepdim=True)[0].cumsum(2)

        assert not torch.isnan(FORWARD).any(), 'FORWARD contains NaNs.'
        return FORWARD

    def backward(self):
        """
        Perform the backward algorithm to compute the probability of the given sequence of observations.

        Mathematical Definition:
        The backward algorithm calculates the probability of the ending part of an observed sequence from a given time t to the end, given the HMM parameters.

        Let:
        - N be the number of states in the HMM.
        - T be the length of the observation sequence.
        - A = {a_ij} be the state transition probability matrix, where a_ij = P(S_{t+1} = j | S_t = i).
        - B = {b_jk} be the emission probability matrix, where b_jk = P(O_t = k | S_t = j).
        - π = {π_i} be the initial state distribution, where π_i = P(S_1 = i).
        - β_t(i) be the backward probability at time t for state i, i.e., the probability of the partial observation sequence from time t+1 to T given state i at time t.

        Initialization:
        β_T(i) = 1, for 1 ≤ i ≤ N

        Induction:
        β_t(i) = Σ_{j=1}^N a_ij * b_j(O_{t+1}) * β_{t+1}(j), for t = T-1, T-2, ..., 1 and 1 ≤ i ≤ N

        Termination:
        P(O | λ) = Σ_{i=1}^N π_i * b_i(O_1) * β_1(i)

        Where:
        - β_t(i) is the probability of observing the sequence from time t+1 to T, given that the system is in state i at time t.
        - Σ denotes the summation over all states.

        :param observations: (list or np.ndarray) Sequence of observations (as integers) where each observation corresponds to an emission symbol.
        :return: (tuple) Tuple containing:
                 - (np.ndarray) Backward probabilities matrix β of shape (T, n_states).
                 - (float) Probability of the observed sequence.
        """

        if not self.model_verified:
            raise RuntimeError('Model must be verified before calling backward.')

        emission_loglik = self.__emissions
        n_cells, n_states, n_features = emission_loglik.shape
        assert n_cells == self.n_cells, f'Emission log likelihoods must have as many rows ({n_cells}) as there are cells ({self.n_cells})'
        assert n_states == self.n_states, f'Emission log likelihoods must have as many columns ({n_states}) as there are states ({self.n_states})'

        emission_probs = torch.exp(emission_loglik - emission_loglik.max(1, keepdim=True)[0])
        # Initialize empty backward recursion matrix.
        BACKWARD = - torch.ones((n_cells, self.n_states, n_features), dtype=torch.double)

        # We initialize the last column with the end distribution (probability of transitioning from that state to the
        # end of the sequence)
        initial_backward_probs = self.end_distribution.repeat(n_cells, 1, 1)
        BACKWARD = torch.cat((BACKWARD, initial_backward_probs), 2)

        transition_matrix = self.get_transition_matrix()
        # Starting from time t=T-1, we compute the backward recursion for state j as the sum over all states, l, of the
        # product of (1) the transition probability from state j to state l at time t and (2) the product of the
        # emission probability at time t+1 and the backward probability at time t+1 for state l.

        # Iteration backward over each step in observation matrix, filling in the second to last column, then
        # the third to last column, etc. until the second column. The first column is filled in separately with
        # the start distribution.
        scale_factors = []
        for observation in range(-2, -self.n_steps-2, -1):
            scale_factor = BACKWARD[:, :, observation + 1].sum(1, keepdim=True)
            BACKWARD[:, :, observation + 1] /= scale_factor
            scale_factors.append(scale_factor)

            for state in range(self.n_states):
                backward_times_transition = torch.mul(BACKWARD[:, :, observation + 1],
                                                      transition_matrix[observation + 1, state, :])
                transition_scores = torch.mul(backward_times_transition, emission_probs[:, :, observation + 1])
                # Determine the best score and best following state.
                BACKWARD[:, state, observation] = transition_scores.sum(1)

        # Finally, we incorporate the start distribution probabilities for the first column of the backward matrix
        # We compute the backward probability for the 0th time step as the product of (1) the start distribution
        # (2) the emission probability at time 0 for each state and (3) the backward probability at time 1 for each state
        scale_factor = BACKWARD[:, :, 1].sum(1, keepdim=True)
        scale_factors.append(scale_factor)
        BACKWARD[:, :, 0] = self.start_distribution.squeeze(-1)*emission_probs[:, :, 0]*BACKWARD[:, :, 1]

        # Use the original scaling factor to recover the original backward matrix
        # We have that B_scaled[sample, state, time] = \prod_time^T 1/scale_factor[t] x B[sample, state, time]
        # Converting to the log space allows us to recover the original B[sample, state, time]
        # log F(state, t) = log B_scale(state, t) + sum_{j=t^T} log scale_factor[j]
        scale_factors = torch.cat(scale_factors, dim=1).view(n_cells, 1, self.n_steps+1).repeat(1, self.n_states, 1)
        scale_factors = torch.log(scale_factors)
        norm = scale_factors.cumsum(dim=2).flip(dims=[2])

        BACKWARD = torch.log(BACKWARD)
        BACKWARD[:, :, 1:] += norm

        # Add the emission probs normalization factor
        max_values = emission_loglik.max(dim=1, keepdim=True).values
        norm = max_values.flip(dims=[2]).cumsum(dim=2).flip(dims=[2])
        BACKWARD[:, :, :-1] += norm

        assert not torch.isnan(BACKWARD).any(), """BACKWARD contains NaNs.:
                transitions: {0} \n
                BACKWARD: {1} \n
                emission_probs : {2}
                """.format(transition_matrix, BACKWARD, emission_probs)

        return BACKWARD


    def gamma(self, forward, backward):
        """
        Compute the gamma matrix, which is the probability of being in state i at time t given the observations up to t.

        :param forward: (Tensor) Forward matrix of shape n_cells x n_states x n_observations
        :param backward: (Tensor) Backward matrix of shape n_cells x n_states x n_observations+1) - the first colu
        :return: (Tensor) Gamma matrix of shape n_cells x n_states x n_observations
        """
        if not self.model_verified:
            raise RuntimeError('Model must be verified before calling gamma.')

        forward = forward
        backward = backward[:, :, 1:]
        gamma_matrix = (forward + backward)

        # Use soft-max trick to normalize gamma matrix
        m = gamma_matrix.max(dim=1, keepdim=True)[0]
        gamma_matrix = torch.exp(gamma_matrix - m) / torch.exp(gamma_matrix - m).sum(dim=1, keepdim=True)

        try:
            assert not torch.isnan(gamma_matrix).any(), 'gamma contains nans'
        except:
            log.error('Gamma contains nans')
            log.error('forward', forward[torch.isnan(gamma_matrix)])
            log.error('backward', backward[torch.isnan(gamma_matrix)])
            log.error('forward+backward', forward[torch.isnan(gamma_matrix)]+backward[torch.isnan(gamma_matrix)])
            log.error('gamma before norm', torch.exp(forward[torch.isnan(gamma_matrix)]+backward[torch.isnan(gamma_matrix)]))
            log.error('gamma', gamma_matrix[torch.isnan(gamma_matrix)])
            exit()

        return gamma_matrix

    def expected_transitions(self, forward, backward):
        """
        Compute the expected number of transitions between states for each time step.
        This method assumes the model has time-varying transition probabilities.

        The expected number of transitions from state i to state j at time t is given by:

            E[N_ij(t)] = sum_k gamma_ik(t) * A_ij(t) * beta_j(t+1) * B_j(t+1) / P(X)

        where:
            gamma_ik(t) is the probability of being in state i at time t given the observations up to t.
            A_ij(t) is the transition probability from state i to state j at time t.
            beta_j(t+1) is the backward probability of state j at time t+1.
            B_j(t+1) is the emission probability of the observation at time t+1 given state j.
            P(X) is the total probability of the observation sequence.

        :param forward: (Tensor) Forward matrix of shape n_cells x n_states x n_observations
        :param backward: (Tensor) Backward matrix of shape n_cells x n_states x n_observations
        :return: (Tensor) Expected transitions matrix of shape n_steps x n_states x n_states
        """

        # Check if the model has been verified before computing expected transitions
        if not self.model_verified:
            raise RuntimeError('Model must be verified before computing expected transitions.')

        # Get the emission log-likelihood, forward matrix, backward matrix, and log of transition matrix
        emission_loglik = self.__emissions
        F = forward
        B = backward[:, :, 1:]
        A = torch.log(self.get_transition_matrix())

        # Initialize the updated transition matrix with zeros
        updated_transitions = torch.zeros((self.n_steps, self.n_states, self.n_states), dtype=torch.double)

        # Slice the forward matrix to exclude the last time step
        F_s = F[:, :, :self.n_steps]

        # Use the logsumexp trick to sum over sequence probabilities
        # seqprob represents the log probability of the sequences
        seqprob = F[:, :, self.n_steps].max(1, keepdim=True)[0] + torch.log(
            torch.exp(F[:, :, self.n_steps] - F[:, :, self.n_steps].max(1, keepdim=True)[0]).sum(1).view(-1, 1))
        seqprob = seqprob.view(-1, 1, 1)

        # Compute the contribution of the emission probabilities and backward probabilities
        # x is reshaped to align with the dimensions needed for transition computations
        x = (emission_loglik[:, :, 1:] + B[:, :, 1:] - seqprob).permute(0, 2, 1).view(self.n_cells, self.n_steps, 1, -1)

        # f is reshaped forward matrix to align with the dimensions needed for transition computations
        f = F_s[:, :, :].permute(0, 2, 1).contiguous().view(self.n_cells, self.n_steps, -1, 1)

        # Compute the expected transitions by combining forward, emission, backward, and transition log probabilities
        transitions = (f + x + A)

        # Sum the computed transitions over all cells to get the updated transition matrix
        updated_transitions = torch.exp(transitions).sum(0)

        # Return the updated transition matrix
        return updated_transitions

    def complete_log_likelihood(self):
        raise NotImplementedError

    def learn(self):
        """
        Implements Baum-Welch algorithm for learning the parameters of the model.
        :return:
        """
        F = self.forward()
        B = self.backward()
        gamma = self.gamma(F, B)
        xi = self.expected_transitions(F, B)

        # Update start distribution
        self.start_distribution = gamma[:, :, 0].sum(0, keepdim=True)
        self.start_distribution /= self.start_distribution.sum()

        # Update transition matrix
        self.__transition_matrix = xi / xi.sum(2, keepdim=True)

        # Update emission probabilities; these are Poisson probabilities
        raise NotImplementedError



