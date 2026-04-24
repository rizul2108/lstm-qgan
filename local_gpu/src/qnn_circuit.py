import pennylane as qml
import torch
import torch.nn as nn
import numpy as np

N_QUBITS = 7
N_LAYERS = 2
_HILBERT = 2 ** N_QUBITS

_dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(_dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs: torch.Tensor, weights: torch.Tensor) -> list:
    qml.AmplitudeEmbedding(features=inputs, wires=range(N_QUBITS), normalize=True, pad_with=0.0)
    for layer in range(N_LAYERS):
        for q in range(N_QUBITS):
            qml.RX(weights[layer, q, 0], wires=q)
            qml.RY(weights[layer, q, 1], wires=q)
            qml.RZ(weights[layer, q, 2], wires=q)
        for q in range(N_QUBITS):
            qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
    return [qml.expval(qml.PauliZ(q)) for q in range(N_QUBITS)]


class QuantumNeuralNetwork(nn.Module):
    def __init__(self, n_qubits: int = N_QUBITS, n_layers: int = N_LAYERS):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.weights = nn.Parameter(torch.rand(n_layers, n_qubits, 3) * 2 * np.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self._pad_batch(x)
        norms = torch.norm(x, dim=-1, keepdim=True).clamp_min(1e-8)
        x = x / norms
        result = quantum_circuit(x, self.weights)
        return torch.stack(result, dim=-1)

    def _pad_batch(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[-1]
        if n < _HILBERT:
            pad = torch.zeros(x.shape[0], _HILBERT - n, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=-1)
        elif n > _HILBERT:
            x = x[:, :_HILBERT]
        return x
