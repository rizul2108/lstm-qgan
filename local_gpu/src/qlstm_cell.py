import torch
import torch.nn as nn
from src.qnn_circuit import QuantumNeuralNetwork, N_QUBITS


class QLSTMCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = N_QUBITS,
        n_qubits: int = N_QUBITS,
        n_layers: int = 2,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_qubits = n_qubits

        self.input_proj = nn.Linear(hidden_size + input_size, 2 ** n_qubits)
        self.qnn_forget    = QuantumNeuralNetwork(n_qubits, n_layers)
        self.qnn_input     = QuantumNeuralNetwork(n_qubits, n_layers)
        self.qnn_candidate = QuantumNeuralNetwork(n_qubits, n_layers)
        self.qnn_output    = QuantumNeuralNetwork(n_qubits, n_layers)
        self.out_proj = nn.Linear(n_qubits, hidden_size) if n_qubits != hidden_size else nn.Identity()

    def _gate(self, qnn, z):
        out = qnn(z).to(dtype=self.input_proj.weight.dtype)
        return self.out_proj(out).to(dtype=self.input_proj.weight.dtype)

    def forward(self, x_t, h_prev, c_prev):
        dtype = self.input_proj.weight.dtype
        x_t, h_prev, c_prev = x_t.to(dtype), h_prev.to(dtype), c_prev.to(dtype)

        z = self.input_proj(torch.cat([h_prev, x_t], dim=-1))
        f = torch.sigmoid(self._gate(self.qnn_forget,    z))
        i = torch.sigmoid(self._gate(self.qnn_input,     z))
        g = torch.tanh(   self._gate(self.qnn_candidate, z))
        o = torch.sigmoid(self._gate(self.qnn_output,    z))

        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)
        return h_t, c_t

    def init_hidden(self, batch_size, device):
        return (
            torch.zeros(batch_size, self.hidden_size, device=device),
            torch.zeros(batch_size, self.hidden_size, device=device),
        )
