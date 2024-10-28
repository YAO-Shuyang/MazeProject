# Hidden Markov Model.

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

class HMM:
    def __init__(self, N, device=device, target_prob: float = 0.6):
        """
        Initialize the HMM model.

        Args:
            N (int): Number of hidden states.
            device (torch.device): Device to run the computations on.
            target_prob (float): Target emission probability to determine the initial state.
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
        self._loss = None  # Initialize loss

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

    def get_predicted_prob(self, sequences):
        """
        Compute P(s_{t+1} = 1 | s_{1:t}) for each time step in each sequence.

        Args:
            sequences (torch.Tensor): Tensor of shape (num_sequences, max_seq_length) containing the sequences.
            sequence_lengths (torch.Tensor): Tensor of shape (num_sequences,) containing the lengths of each sequence.

        Returns:
            next_obs_probabilities_list (list[np.ndarray]): List of NumPy arrays containing the probabilities for each sequence.
        """
        sequences, sequence_lengths = self.get_padded_sequences(sequences)
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

    def get_padded_sequences(self, sequences: list[np.ndarray]):
        sequence_lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
        # Convert sequences and lengths to tensors
    
        max_length = torch.max(sequence_lengths)

        # Pad sequences to the maximum length
        max_seq_length = max(sequence_lengths)
        num_sequences = len(sequences)
        padded_sequences = torch.zeros((num_sequences, max_seq_length), dtype=torch.long)

        for i, seq in enumerate(sequences):
            seq_len = sequence_lengths[i]
            padded_sequences[i, :seq_len] = torch.tensor(seq, dtype=torch.long)

        # Move data to GPU if available
        padded_sequences = padded_sequences.to(device)
        sequence_lengths = sequence_lengths.to(device)
        return padded_sequences, sequence_lengths

    def calc_loss(self, sequences: list[np.ndarray]):
        """Calculate negative log likelihood loss"""
        predicted_prob = self.get_predicted_prob(sequences)

        # Compute the log-likelihood
        loss = 0
        for i in range(len(predicted_prob)):
            if len(predicted_prob[i]) > 0:
                loss += np.sum(sequences[i][1:] * np.log(predicted_prob[i] + 1e-10) + (1 - sequences[i][1:]) * np.log(1 - predicted_prob[i] + 1e-10))

        n_total = np.sum([len(seq)-1 for seq in sequences if len(seq) > 1])
        self._loss = -loss / n_total
        print(f"Hidden Markov Model with {self.N} hidden states:\n"
              f"  Loss: {self.loss}\n")
        return self._loss

    def calc_loss_along_seq(self, sequences: list[np.ndarray]):
        max_length = max([len(seq) for seq in sequences])
        predicted_p = self.get_predicted_prob(sequences)
        padded_p = np.zeros((len(predicted_p), max_length-1)) * np.nan
        padd_seq = np.zeros((len(predicted_p), max_length-1)) * np.nan
        for i in range(len(predicted_p)):
            padded_p[i, :len(predicted_p[i])] = predicted_p[i]
            padd_seq[i, :len(predicted_p[i])] = sequences[i][1:]
        
        dloss = padd_seq * np.log(padded_p + 1e-10) + (1 - padd_seq) * np.log(1 - padded_p + 1e-10)
        loss = -np.nanmean(dloss, axis=0)
        print(f"Hidden Markov Model with {self.N} hidden states:\n"
              f"  Loss: {loss}\n")
        return loss

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

    def fit(self, data_loader, iterations=10):
        for iteration in tqdm(range(iterations)):
            total_loss = 0.0
            total_sequences = 0
            numerator_accum = torch.zeros_like(self.transition_matrix, device=self.device)
            denominator_accum = torch.zeros_like(self.transition_matrix, device=self.device)
            for batch_idx, (batch_sequences, batch_lengths) in enumerate(data_loader):
                batch_sequences = batch_sequences.to(self.device)
                batch_lengths = batch_lengths.to(self.device)

                # Forward algorithm
                alpha, scaling_factors = self.forward(batch_sequences, batch_lengths)

                # Compute loss
                batch_total_loss, batch_average_loss = self.compute_loss(scaling_factors, batch_lengths)
                total_loss += batch_total_loss
                total_sequences += batch_lengths.sum().item()

                # Backward algorithm
                beta = self.backward(batch_sequences, batch_lengths, scaling_factors)

                # Compute gamma and xi
                gamma, xi = self.compute_gamma_xi(alpha, beta, batch_sequences, batch_lengths, scaling_factors)

                # Accumulate numerators and denominators
                xi_sum = xi.sum(dim=1)  # Sum over time steps
                gamma_sum = gamma[:, :-1, :].sum(dim=1)  # Sum over time steps excluding the last

                numerator_batch = xi_sum.sum(dim=0)  # Sum over sequences
                denominator_batch = gamma_sum.sum(dim=0).unsqueeze(1) + 1e-10  # Sum over sequences

                numerator_accum += numerator_batch
                denominator_accum += denominator_batch

            # Update transition matrix once per iteration
            self.transition_matrix = numerator_accum / denominator_accum

            # Ensure rows sum to 1
            self.transition_matrix = self.transition_matrix / self.transition_matrix.sum(dim=1, keepdim=True)

            average_loss = total_loss / total_sequences
            #print(f"Iteration {iteration + 1}, Total Loss: {total_loss:.4f}, Average Loss: {average_loss:.6f}")

    def get_transition_matrix(self):
        return self.transition_matrix.cpu().numpy()
    
    @staticmethod
    def process_fit(
        N: int, # Number of hidden states
        sequences: list[np.ndarray],
        batch_size: int = 4096,
        n_iterations: int = 1000
    ) -> 'HMM':

        # Initialize the HMM model
        hmm_model = HMM(N, device)
        
        padded_sequences, sequence_lengths = hmm_model.get_padded_sequences(sequences)
        # Create a TensorDataset
        dataset = torch.utils.data.TensorDataset(padded_sequences, sequence_lengths)

        # Create a DataLoader
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Train the HMM using the DataLoader
        hmm_model.fit(data_loader, iterations=n_iterations)
        return hmm_model
        
    def simulate(self, sequences: list[np.ndarray]) -> list[np.ndarray]:
        self.get_predicted_prob(sequences)
        simu_seq = []
        for n, seq in enumerate(sequences):
            simu = [1]
            for i in range(len(seq) - 1):
                curr_p = self.predicted_prob[n][i]
                simu.append(np.random.choice([0, 1], p=[1 - curr_p, curr_p]))
            simu_seq.append(np.array(simu))
        return simu_seq
        
if __name__ == "__main__":
    import pickle
    # Number of hidden states
    N = 20  # You can change this to 5, 10, 20, etc.
    # Get sequences

    with open(r"E:\Anaconda\envs\maze\Lib\site-packages\mylib\test\demo_seq.pkl", 'rb') as handle:
        sequences = pickle.load(handle)

    hmm_model = HMM.process_fit(
        N=N,
        sequences=sequences,
        batch_size=4096,
        n_iterations=100
    )

    # Get loss
    loss = hmm_model.calc_loss(sequences)