import torch
import torch.nn as nn
from src.qlstm_cell import QLSTMCell
from src.qnn_circuit import N_QUBITS


class QLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size=N_QUBITS, n_qubits=N_QUBITS, n_layers=2, num_cells=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.cells = nn.ModuleList(
            QLSTMCell(
                input_size=input_size if i == 0 else hidden_size,
                hidden_size=hidden_size,
                n_qubits=n_qubits,
                n_layers=n_layers,
            )
            for i in range(num_cells)
        )

    def forward(self, x, states=None):
        if states is None:
            states = [cell.init_hidden(x.shape[0], x.device) for cell in self.cells]
        new_states = []
        cur = x
        for i, cell in enumerate(self.cells):
            h, c = cell(cur, *states[i])
            new_states.append((h, c))
            cur = h
        return cur, new_states


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim=64,
        patch_size=196,
        n_steps=4,
        hidden_size=N_QUBITS,
        n_qubits=N_QUBITS,
        n_vqc_layers=2,
        num_cells=4,
    ):
        super().__init__()
        self.latent_dim  = latent_dim
        self.patch_size  = patch_size
        self.n_steps     = n_steps
        self.hidden_size = hidden_size

        self.input_embed = nn.Linear(latent_dim, hidden_size)
        self.qlstm1 = QLSTMLayer(hidden_size, hidden_size, n_qubits, n_vqc_layers, num_cells)
        self.qlstm2 = QLSTMLayer(hidden_size, hidden_size, n_qubits, n_vqc_layers, num_cells)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, patch_size),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.input_embed(z)
        s1 = s2 = None
        patches = []
        for _ in range(self.n_steps):
            h1, s1 = self.qlstm1(x, s1)
            h2, s2 = self.qlstm2(h1, s2)
            patches.append(self.output_head(h2))
        return torch.cat(patches, dim=-1)

    def generate_image(self, batch_size=1, device=torch.device("cpu")):
        z = torch.randn(batch_size, self.latent_dim, device=device)
        with torch.no_grad():
            return self.forward(z)
