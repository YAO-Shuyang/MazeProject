import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import time
import numpy as np
from tqdm import tqdm

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

class SequenceDataset(Dataset):
    def __init__(self, sequences):
        """
        Initializes the dataset by storing the sequences.

        Parameters:
            sequences (list of list or np.ndarray): List of sequences.
        """
        self.sequences = [torch.tensor(seq, dtype=torch.float32, device=device) for seq in sequences]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Retrieves the sequence at the specified index.

        Parameters:
            idx (int): Index of the sequence to retrieve.

        Returns:
            torch.Tensor: The sequence as a float32 tensor.
        """
        return self.sequences[idx]

def collate_fn(batch):
    """
    Pads sequences to the same length within a batch.

    Parameters:
        batch (list of torch.Tensor): List of sequences.

    Returns:
        tuple: Padded sequences tensor and list of original lengths.
    """
    sequences = [item for item in batch]
    lengths = [len(seq) for seq in sequences]
    padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded_sequences, lengths

class ProbabilityRNN(nn.Module):
    def __init__(self, hidden_size, p0: float = 0.55):
        """
        Initializes the ProbabilityRNN model.

        Parameters:
            hidden_size (int): Number of features in the hidden state of the GRU.
        """
        super(ProbabilityRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size=2, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.predicted_prob = None
        self.p0 = p0

    def forward(self, s: torch.Tensor, lengths):
        """
        Forward pass of the model.

        Parameters:
            s (torch.Tensor): Padded input sequences of shape (batch_size, seq_len).
            lengths (list of int): Original lengths of each sequence.

        Returns:
            torch.Tensor: Predicted probabilities of shape (batch_size, seq_len - 1).
        """
        batch_size, seq_len = s.size()
        r_i = torch.full((batch_size, 1), self.p0, device=device)
        r_preds = []

        for t in range(seq_len-1):
            s_i = s[:, t].unsqueeze(-1)  # Shape: (batch_size, 1)
            input_t = torch.cat([s_i, r_i], dim=-1)  # Shape: (batch_size, 2)
            input_t = input_t.unsqueeze(1)  # Shape: (batch_size, 1, 2)
            output_t, _ = self.rnn(input_t)  # Output shape: (batch_size, 1, hidden_size)
            r_i = self.sigmoid(self.fc(output_t.squeeze(1)))  # Shape: (batch_size, 1)
            r_preds.append(r_i)

        r_preds = torch.stack(r_preds, dim=1).squeeze(-1)  # Shape: (batch_size, seq_len - 1)
        return r_preds

    def fit(self, dataloader, num_epochs, optimizer, criterion, val_loader=None, early_stopping_patience=10):
        """
        Train the model using the given dataloader and optimizer.

        Parameters:
            dataloader (torch.utils.data.DataLoader): The dataloader that provides the training data.
            num_epochs (int): The number of epochs to train the model.
            optimizer (torch.optim.Optimizer): The optimizer to use during training.
            criterion (torch.nn.Module): The loss function to use during training.
            val_loader (torch.utils.data.DataLoader, optional): Dataloader for validation data.
            early_stopping_patience (int, optional): Number of epochs with no improvement after which training will be stopped.

        Notes:
            This method trains the model in-place, meaning that it modifies the
            model's parameters. The model is set to training mode before training
            begins and is left in training mode after training is finished. The
            method also prints the total loss for each epoch to the console.

        Example:
            >>> model = ProbabilityRNN(128)
            >>> criterion = nn.BCELoss()
            >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            >>> dataloader = torch.utils.data.DataLoader(
            ...     dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
            >>> model.fit(dataloader, num_epochs=10, optimizer=optimizer, criterion=criterion)
        """
        self.train()
        best_val_loss = float('inf')
        patience = early_stopping_patience
        t0 = time.time()

        for epoch in tqdm(range(num_epochs)):
            self.train()
            total_loss = 0.0
            total_valid_elements = 0
            for s_batch, lengths in dataloader:
                s_batch = s_batch.to(device)
                optimizer.zero_grad()
                # Forward pass
                r_pred = self(s_batch, lengths)
                # Compute loss
                s_target = s_batch[:, 1:]  # Shifted target
                lengths_tensor = torch.tensor(lengths, dtype=torch.long).to(device)
                mask = (torch.arange(s_target.size(1), device=device).expand(len(lengths_tensor), s_target.size(1)) < (lengths_tensor - 1).unsqueeze(1))
                loss = criterion(r_pred[mask], s_target[mask])
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * torch.sum(mask.float())
                total_valid_elements += torch.sum(mask.float()) # Accumulate valid element count
                
            average_loss = total_loss / total_valid_elements

            # Validation
            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                total_valid_elements = 0
                with torch.no_grad():
                    for s_batch, lengths in val_loader:
                        s_batch = s_batch.to(device)
                        r_pred = self(s_batch, lengths)
                        s_target = s_batch[:, 1:]
                        lengths_tensor = torch.tensor(lengths, dtype=torch.long).to(device)
                        mask = (torch.arange(s_target.size(1), device=device).expand(len(lengths_tensor), s_target.size(1)) < (lengths_tensor - 1).unsqueeze(1))
                        loss = criterion(r_pred[mask], s_target[mask])
                        val_loss += loss.item() * torch.sum(mask.float())
                        total_valid_elements += torch.sum(mask.float())
                average_val_loss = val_loss / total_valid_elements
                #print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {average_loss:.6f}, Val Loss: {average_val_loss:.6f} "
                #      f"time: {int((time.time() - t0) // 3600)}:{int(((time.time() - t0) % 3600) // 60)}:{int((time.time() - t0) % 60)}.{int(((time.time() - t0) * 1000) % 1000)}")
                
                # Early Stopping Check
                if average_val_loss < best_val_loss:
                    best_val_loss = average_val_loss
                    patience = early_stopping_patience  # Reset patience
                else:
                    patience -= 1
                    if patience == 0:
                        print("  Early stopping triggered.")
                        break
            else:
                dt = time.time() - t0
                #print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.6f}, "
                #      f"time: {int(dt // 3600)}:{int((dt % 3600) // 60)}:{int(dt % 60)}.{int((dt*1000) % 1000)}")

    def simulate(self, sequences: list[np.ndarray]) -> list[np.ndarray]:
        """
        Generates a list of simulated 0-1 sequences based on the trained model.

        Parameters:
            sequences (list of np.ndarray): List of input sequences to simulate.

        Returns:
            simulated (list of np.ndarray): List of simulated sequences with the same lengths as input.
        """
        self.get_predicted_prob(sequences)

        simu_seq = []
        for n, seq in enumerate(sequences):
            simu = [1]
            for i in range(len(seq) - 1):
                curr_p = self.predicted_prob[n][i]
                simu.append(np.random.choice([0, 1], p=[1 - curr_p, curr_p]))
            simu_seq.append(np.array(simu))
        return simu_seq
    

    def get_predicted_prob(self, sequences: list[np.ndarray]) -> list[np.ndarray]:
        """
        Compute P(s_{t+1} = 1 | s_{1:t}) for each time step in each sequence.

        Args:
            sequences (list of np.ndarray): List of input sequences to predict probabilities for.
            sequence_lengths (list of int): List containing the lengths of each corresponding sequence.

        Returns:
            list of np.ndarray: List containing arrays of predicted probabilities for each sequence.
        """
        self.eval()  # Set model to evaluation mode
        probabilities_list = []

        with torch.no_grad():
            batch_size = len(sequences)
            lengths = [len(seq) for seq in sequences]
            if batch_size == 0:
                return probabilities_list  # Return empty list if no sequences provided

            max_length = max(lengths)
            # Convert sequences to tensor with padding
            sequences_tensor = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]
            padded_sequences = nn.utils.rnn.pad_sequence(sequences_tensor, batch_first=True, padding_value=0)
            padded_sequences = padded_sequences.to(device)

            # Forward pass to get predicted probabilities
            r_pred = self(padded_sequences, lengths)  # Shape: (batch_size, seq_len - 1)

            # Create mask to filter out padded positions
            lengths_tensor = torch.tensor(lengths, dtype=torch.long).to(device)
            mask = (torch.arange(r_pred.size(1), device=device).expand(len(lengths_tensor), r_pred.size(1)) < (lengths_tensor - 1).unsqueeze(1))
            r_pred = r_pred * mask.float()

            # Convert to CPU and NumPy
            r_pred_np = r_pred.cpu().numpy()

            # Split into list of arrays corresponding to each sequence
            for i in range(batch_size):
                seq_length = lengths[i]
                if seq_length > 1:
                    prob_seq = r_pred_np[i, :seq_length - 1]
                else:
                    prob_seq = np.array([])
                probabilities_list.append(prob_seq)

        self.predicted_prob = probabilities_list
        return probabilities_list
    
    def calc_loss(self, sequences: list[np.ndarray]) -> float:
        """Calculate negative log likelihood loss"""
        predicted_prob = self.get_predicted_prob(sequences)

        # Compute the log-likelihood
        loss = 0
        for i in range(len(predicted_prob)):
            if len(predicted_prob[i]) > 0:
                loss += np.sum(sequences[i][1:] * np.log(predicted_prob[i] + 1e-10) + (1 - sequences[i][1:]) * np.log(1 - predicted_prob[i] + 1e-10))

        n_total = np.sum([len(seq)-1 for seq in sequences if len(seq) > 1])
        self._loss = -loss / n_total
        print(f"Recurrent Neural Network Model:\n"
              f"  Loss: {self.loss}\n")
        return self._loss
    
    @property
    def loss(self):
        return self._loss
    
    @staticmethod
    def process_fit(
        sequences: list[np.ndarray], 
        train_index: np.ndarray = None,
        split_ratio: float = 0.8,
        hidden_size: int = 32,
        lr: float = 0.001,
        epochs: int = 1000, 
        batch_size: int = 2048
    ) -> 'ProbabilityRNN':
        if train_index is None:
            train_idx = np.random.choice(len(sequences), size=int(len(sequences) * split_ratio), replace=False)
            train_sequences = [sequences[i] for i in train_idx]
            val_sequences = [sequences[i] for i in np.setdiff1d(np.arange(len(sequences)), train_idx)]
        else:
            train_sequences = [sequences[i] for i in train_index]
            val_sequences = [sequences[i] for i in np.setdiff1d(np.arange(len(sequences)), train_index)]
        
        p0 = np.sum([seq[1] for seq in train_sequences]) / len(sequences)
        train_dataset = SequenceDataset(train_sequences)
        val_dataset = SequenceDataset(val_sequences)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        model = ProbabilityRNN(hidden_size=hidden_size, p0=p0).to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model.fit(
            dataloader=train_loader, 
            num_epochs=epochs, 
            optimizer=optimizer, 
            criterion=criterion, 
            val_loader=val_loader,
            early_stopping_patience=100
        )
        return model

if __name__ == '__main__':
    import pickle
    
    # Load sequences from the pickle file
    with open(r"E:\Anaconda\envs\maze\Lib\site-packages\mylib\test\demo_seq.pkl", 'rb') as handle:
        sequences = pickle.load(handle)

    # Train the model with early stopping
    model = ProbabilityRNN.process_fit(
        sequences=sequences,
        split_ratio=0.8,
        hidden_size=32,
        lr=0.001,
        epochs=1000, 
        batch_size=2048
    )
    
    model.calc_loss(sequences)