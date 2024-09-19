# Hidden Markov Model.

import numpy as np
import torch
import torch.nn.functional as F

class HMM:
    def __init__(self, N, device, target_prob: float = 0.6):
        """
        Initialize the HMM model.

        Args:
            N (int): Number of hidden states.
            emission_probs (np.ndarray): Emission probabilities for each state.
            initial_state (int): Index of the initial state.
            device (torch.device): Device to run the computations on.
        """
        self.N = N
        self.device = device

        # Initialize transition matrix with uniform probabilities
        self.transition_matrix = torch.full((N, N), 1 / N, dtype=torch.float32, device=device)

        # Set emission probabilities
        emission_probs = self.get_emission_probs()
        self.emission_probs = torch.tensor(
            emission_probs, 
            dtype=torch.float32, 
            device=device
        )
        self.initial_state = np.argmin(np.abs(emission_probs - target_prob))
        self.p0 = emission_probs[self.initial_state]
        self.predicted_prob = None

    def get_emission_probs(self) -> np.ndarray:
        k_values = np.arange(self.N)
        emission_probs = (1 / (2 * self.N)) + (k_values / self.N)
        return emission_probs        

    def forward(self, sequences, sequence_lengths):
        num_sequences, max_seq_length = sequences.shape

        # Initialize alpha tensor
        alpha = torch.zeros((num_sequences, max_seq_length, self.N), device=self.device)

        # Compute emission probabilities for observations
        emission_matrix = torch.where(
            sequences.unsqueeze(2) == 1,
            self.emission_probs.unsqueeze(0).unsqueeze(0),
            1 - self.emission_probs.unsqueeze(0).unsqueeze(0)
        )  # Shape: (num_sequences, max_seq_length, N)

        # Initialization (t = 0)
        alpha[:, 0, :] = 0.0
        alpha[:, 0, self.initial_state] = emission_matrix[:, 0, self.initial_state]

        # Scaling factors to prevent underflow
        scaling_factors = torch.zeros((num_sequences, max_seq_length), device=self.device)
        scaling_factors[:, 0] = alpha[:, 0, :].sum(dim=1) + 1e-10  # Prevent division by zero
        alpha[:, 0, :] /= scaling_factors[:, 0].unsqueeze(1)

        # Recursion
        for t in range(1, max_seq_length):
            mask = (t < sequence_lengths).float().unsqueeze(1)
            alpha_t = torch.bmm(
                alpha[:, t - 1, :].unsqueeze(1),
                self.transition_matrix.unsqueeze(0).expand(num_sequences, self.N, self.N)
            ).squeeze(1) * emission_matrix[:, t, :]
            # Scaling
            scaling_factors[:, t] = alpha_t.sum(dim=1) + 1e-10
            alpha_t /= scaling_factors[:, t].unsqueeze(1)
            alpha[:, t, :] = alpha_t * mask + alpha[:, t, :] * (1 - mask)

        return alpha, scaling_factors

    def backward(self, sequences, sequence_lengths, scaling_factors):
        num_sequences, max_seq_length = sequences.shape

        # Initialize beta tensor
        beta = torch.zeros((num_sequences, max_seq_length, self.N), device=self.device)

        # Compute emission probabilities for observations
        emission_matrix = torch.where(
            sequences.unsqueeze(2) == 1,
            self.emission_probs.unsqueeze(0).unsqueeze(0),
            1 - self.emission_probs.unsqueeze(0).unsqueeze(0)
        )

        # Initialization (t = T - 1)
        beta[:, -1, :] = 1.0 / (scaling_factors[:, -1].unsqueeze(1) + 1e-10)

        # Recursion
        for t in reversed(range(max_seq_length - 1)):
            mask = (t < (sequence_lengths - 1)).float().unsqueeze(1)
            beta_t = torch.bmm(
                self.transition_matrix.unsqueeze(0).expand(num_sequences, self.N, self.N),
                (emission_matrix[:, t + 1, :] * beta[:, t + 1, :]).unsqueeze(2)
            ).squeeze(2)
            beta_t /= scaling_factors[:, t + 1].unsqueeze(1) + 1e-10
            beta[:, t, :] = beta_t * mask + beta[:, t, :] * (1 - mask)

        return beta

    def compute_gamma_xi(self, alpha, beta, sequences, sequence_lengths, scaling_factors):
        num_sequences, max_seq_length = sequences.shape

        # Compute emission probabilities for observations
        emission_matrix = torch.where(
            sequences.unsqueeze(2) == 1,
            self.emission_probs.unsqueeze(0).unsqueeze(0),
            1 - self.emission_probs.unsqueeze(0).unsqueeze(0)
        )

        # Compute gamma
        gamma = alpha * beta
        gamma_sum = gamma.sum(dim=2, keepdim=True) + 1e-10
        gamma = gamma / gamma_sum

        # Compute xi
        xi = torch.zeros((num_sequences, max_seq_length - 1, self.N, self.N), device=self.device)
        for t in range(max_seq_length - 1):
            mask = (t < (sequence_lengths - 1)).float().unsqueeze(1).unsqueeze(1)
            numerator = (
                alpha[:, t, :].unsqueeze(2) *
                self.transition_matrix.unsqueeze(0) *
                emission_matrix[:, t + 1, :].unsqueeze(1) *
                beta[:, t + 1, :].unsqueeze(1)
            )
            numerator /= scaling_factors[:, t + 1].unsqueeze(1).unsqueeze(1) + 1e-10
            xi_t = numerator
            xi[:, t, :, :] = xi_t * mask + xi[:, t, :, :] * (1 - mask)

        # Normalize xi
        xi_sum = xi.sum(dim=(2, 3), keepdim=True) + 1e-10
        xi = xi / xi_sum

        return gamma, xi


    def get_predicted_prob(self, sequences, sequence_lengths):
        """
        Compute P(s_{t+1} = 1 | s_{1:t}) for each time step in each sequence.
    
        Args:
            sequences (torch.Tensor): Tensor of shape (num_sequences, max_seq_length) containing the sequences.
            sequence_lengths (torch.Tensor): Tensor of shape (num_sequences,) containing the lengths of each sequence.
    
        Returns:
            next_obs_probabilities_list (list[np.ndarray]): List of NumPy arrays containing the probabilities for each sequence.
        """
        num_sequences, max_seq_length = sequences.shape
    
        with torch.no_grad():
            # Forward algorithm
            alpha, _ = self.forward(sequences, sequence_lengths)
        
            # Initialize tensor for next observation probabilities
            next_obs_probabilities = torch.zeros((num_sequences, max_seq_length - 1), device=self.device)
        
            # For each time step t, compute P(s_{t+1} = 1 | s_{1:t})
            for t in range(max_seq_length - 1):
                # Compute P(h_t | s_{1:t})
                alpha_t = alpha[:, t, :]  # Shape: (num_sequences, N)
                alpha_sum = alpha_t.sum(dim=1, keepdim=True) + 1e-10
                state_prob_t = alpha_t / alpha_sum  # P(h_t | s_{1:t})
    
                # Predict P(h_{t+1} | s_{1:t}) = state_prob_t @ transition_matrix
                state_prob_next = torch.matmul(state_prob_t, self.transition_matrix)
    
                # Compute P(s_{t+1} = 1 | s_{1:t})
                next_obs_probs_t = torch.matmul(state_prob_next, self.emission_probs)
    
                next_obs_probabilities[:, t] = next_obs_probs_t
    
            # Convert to CPU and NumPy
            next_obs_probabilities_np = next_obs_probabilities.cpu().numpy()
            sequence_lengths_np = sequence_lengths.cpu().numpy()
        
            # Split into a list of arrays corresponding to each sequence
            next_obs_probabilities_list = []
            for i in range(next_obs_probabilities_np.shape[0]):
                length = sequence_lengths_np[i] - 1  # One less than the sequence length
                if length > 0:
                    next_obs_probs_seq = next_obs_probabilities_np[i, :length]  # Truncate to valid length
                    next_obs_probabilities_list.append(next_obs_probs_seq)
                else:
                    next_obs_probabilities_list.append(np.array([]))

        self.predicted_prob = next_obs_probabilities_list
        return next_obs_probabilities_list
    
    def calc_loss(self, sequences, padded_sequences, sequence_lengths):
        """Calculate negative log likelihood loss"""
        if self.predicted_prob is None:
            predicted_prob = self.get_predicted_prob(padded_sequences, sequence_lengths)
        else:
            predicted_prob = self.predicted_prob
            
        # Compute the log-likelihood
        loss = 0
        for i in range(len(predicted_prob)):
            loss += np.sum(sequences[i][1:] * np.log(predicted_prob[i]) + (1 - sequences[i][1:]) * np.log(1 - predicted_prob[i]))
        
        n_total = np.sum([len(seq)-1 for seq in sequences])
        self._loss = -loss / n_total
        print(f"Hidden Markov Model with {self.N} hidden states:\n"
              f"  Loss: {self.loss}\n"
              f"  Transition Matrix: {self.transition_matrix}\n"
              f"  Emission Matrix: {self.emission_probs}"
              f"  Initial States: {self.initial_state} with P0 = {self.p0}.")
        return self._loss
    
    @property
    def loss(self):
        return self._loss
        
    def compute_loss(self, scaling_factors, sequence_lengths):
        # Compute the log-likelihood
        log_likelihood = -torch.sum(torch.log(scaling_factors), dim=1)
        # Sum over sequences
        total_loss = log_likelihood.sum()
        average_loss = total_loss / sequence_lengths.sum()
        return total_loss.item(), average_loss.item()

    def train(self, sequences, sequence_lengths, iterations=10):
        for iteration in range(iterations):
            # Forward algorithm
            alpha, scaling_factors = self.forward(sequences, sequence_lengths)

            # Compute loss
            total_loss, average_loss = self.compute_loss(scaling_factors, sequence_lengths)
            print(f"Iteration {iteration + 1}, Total Loss: {total_loss:.4f}, Average Loss: {average_loss:.6f}")

            # Backward algorithm
            beta = self.backward(sequences, sequence_lengths, scaling_factors)

            # Compute gamma and xi
            gamma, xi = self.compute_gamma_xi(alpha, beta, sequences, sequence_lengths, scaling_factors)

            # Update transition matrix
            xi_sum = xi.sum(dim=1)  # Sum over time steps
            gamma_sum = gamma[:, :-1, :].sum(dim=1)  # Sum over time steps excluding the last

            numerator = xi_sum.sum(dim=0)  # Sum over sequences
            denominator = gamma_sum.sum(dim=0).unsqueeze(1) + 1e-10  # Sum over sequences

            self.transition_matrix = numerator / denominator

            # Ensure rows sum to 1
            self.transition_matrix = self.transition_matrix / self.transition_matrix.sum(dim=1, keepdim=True)

    def get_transition_matrix(self):
        return self.transition_matrix.cpu().numpy()  

if __name__ == "__main__":
    
    import pickle
    # Number of hidden states
    N = 40  # You can change this to 5, 10, 20, etc.
    # Get sequences

    with open(r"E:\Anaconda\envs\maze\Lib\site-packages\mylib\test\demo_seq.pkl", 'rb') as handle:
        sequences = pickle.load(handle)

    # Convert sequences and lengths to tensors
    sequence_lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    max_length = torch.max(sequence_lengths)

    # Pad sequences to the maximum length
    max_seq_length = max(sequence_lengths)
    num_sequences = len(sequences)
    padded_sequences = torch.zeros((num_sequences, max_seq_length), dtype=torch.long)

    for i, seq in enumerate(sequences):
        seq_len = sequence_lengths[i]
        padded_sequences[i, :seq_len] = torch.tensor(seq, dtype=torch.long)

    # Move data to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    padded_sequences = padded_sequences.to(device)
    sequence_lengths = sequence_lengths.to(device)

    # Initialize the HMM model
    hmm_model = HMM(N, device)

    # Train the HMM
    hmm_model.train(padded_sequences, sequence_lengths, iterations=100)

    # Get the trained transition matrix
    trained_transition_matrix = hmm_model.get_transition_matrix()
    
    # Get loss
    loss = hmm_model.calc_loss(
        sequences=sequences, 
        padded_sequences=padded_sequences, 
        sequence_lengths=sequence_lengths
    )