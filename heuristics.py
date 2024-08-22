import random

import torch.optim as optim
import numpy as np

import torch
import torch.nn as nn


from topspin import TopSpinState

class BaseHeuristic:
    def __init__(self, n=20, k=5):
        self._n = n
        self._k = k

    def get_h_values(self, states):
        # states_as_list = [states.get_state_as_list()]
        states_as_list = [state.get_state_as_list() for state in states]
        gaps = []

        for state_as_list in states_as_list:
            gap = 0
            if state_as_list[0] != 1:
                gap = 1

            for i in range(len(state_as_list) - 1):
                if abs(state_as_list[i] - state_as_list[i + 1]) != 1:
                    gap += 1

            gaps.append(gap)

        return gaps



class HeuristicModel(nn.Module):
    def __init__(self, input_dim):
        super(HeuristicModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)

        # Multi-Head Attention Layer
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)

        self.fc3 = nn.Linear(128, 128)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(128, 64)
        self.dropout4 = nn.Dropout(0.3)
        self.fc5 = nn.Linear(64, 16)
        self.dropout5 = nn.Dropout(0.3)
        self.fc6 = nn.Linear(16, 1)

        self._initialize_weights()

        # Identity layers to match dimensions for residual connections
        self.identity1 = nn.Linear(256, 128, bias=False)
        self.identity2 = nn.Linear(128, 64, bias=False)
        self.identity3 = nn.Linear(64, 16, bias=False)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # First layer
        x1 = torch.relu(self.fc1(x))
        x1 = self.dropout1(x1)

        # Residual connection from x1 to x2
        residual1 = self.identity1(x1)
        x2 = torch.relu(self.fc2(x1))
        x2 = self.dropout2(x2)
        x2 += residual1

        # Attention layer
        x2 = x2.unsqueeze(1)  # Adding a dimension for attention (sequence length of 1)
        x2, _ = self.attention(x2, x2, x2)
        x2 = x2.squeeze(1)  # Removing the sequence length dimension

        # Residual connection within layer 3
        residual2 = x2
        x3 = torch.relu(self.fc3(x2))
        x3 = self.dropout3(x3)
        x3 += residual2

        # Residual connection from x3 to x4
        residual3 = self.identity2(x3)
        x4 = torch.relu(self.fc4(x3))
        x4 = self.dropout4(x4)
        x4 += residual3

        # Residual connection from x4 to x5
        residual4 = self.identity3(x4)
        x5 = torch.relu(self.fc5(x4))
        x5 = self.dropout5(x5)
        x5 += residual4

        # Final layer without residual connection
        x = self.fc6(x5)

        return x


class LearnedHeuristic:
    def __init__(self, n=20, k=5):
        self._n = n
        self._k = k
        self._model = HeuristicModel(n)
        self._criterion = nn.MSELoss()
        self._optimizer = optim.Adam(self._model.parameters(), lr=0.001)

    def get_h_values(self, states):
        if isinstance(states, TopSpinState):
            states_as_list = [states.get_state_as_list()]
        else:
            states_as_list = [state.get_state_as_list() for state in states]
        states = np.array(states_as_list, dtype=np.float32)
        states_tensor = torch.tensor(states)
        with torch.no_grad():
            predictions = self._model(states_tensor).numpy()
        return predictions.flatten()

    def train_model(self, input_data, output_labels, epochs=100, verbose=True):
        input_as_list = [state.get_state_as_list() for state in input_data]
        inputs = np.array(input_as_list, dtype=np.float32)
        outputs = np.array(output_labels, dtype=np.float32)

        inputs_tensor = torch.tensor(inputs)
        outputs_tensor = torch.tensor(outputs).unsqueeze(1)  # Adding a dimension for the output

        loss = None
        for epoch in range(epochs):
            self._model.train()
            self._optimizer.zero_grad()

            predictions = self._model(inputs_tensor)
            loss = self._criterion(predictions, outputs_tensor)
            if verbose:
                print(f"Epoch: {epoch}, Loss: {loss.item()}")
            loss.backward()
            self._optimizer.step()

        return loss.item()

    def save_model(self, path):
        torch.save(self._model.state_dict(), path)

    def load_model(self, path):
        self._model.load_state_dict(torch.load(path))
        self._model.eval()

class BootstrappingHeuristic(LearnedHeuristic):
    def __init__(self, n=11, k=4, path='bootstrapping_heuristic.pth'):
        super().__init__(n, k)
        self.goal_state = [i for i in range(1, n + 1)]
        self.top_spin_object = TopSpinState(self.goal_state, k)
        self.input_data = []
        self.output_labels = []
        self.iter = 0
        self.n = n
        self.k = k
        self.path = path
        self.generated_states = []

    def generate_random_states(self, num_states_to_generate=1000, min_steps=10, max_steps=200):
        states = []
        current_state = self.top_spin_object.get_neighbors()
        left_rotate, right_rotate, flip = current_state
        while len(states) < num_states_to_generate:
            last_obj = None
            num_steps = random.randint(min_steps, max_steps)
            for step in range(num_steps):
                chosen_neighbor = random.choice([left_rotate[0], right_rotate[0], flip[0]])
                left_rotate, right_rotate, flip = chosen_neighbor.get_neighbors()
                if step == num_steps - 1:
                    last_obj = chosen_neighbor

            states.append(last_obj)

        return states

    def generate_random_states_test(self, num_states=10_000):
        states = []
        while len(states) < num_states:
            current_state = self.top_spin_object.get_neighbors()  # final state
            left_rotate, right_rotate, flip = current_state
            num_steps = random.randint(20, 200)
            for step in range(num_steps):
                chosen_neighbor = random.choice([left_rotate[0], right_rotate[0], flip[0]])
                left_rotate, right_rotate, flip = chosen_neighbor.get_neighbors()
                states.append(chosen_neighbor)
        return states


    def save_model(self):
        path = self.path
        super().save_model(path)
        print("Model saved")

    def load_model(self):
        path = self.path
        super().load_model(path)
        print("Model loaded")






