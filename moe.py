# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#
# The code is based on the TensorFlow implementation:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py


import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.init as init
import numpy as np


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""
        self._gates = gates
        self._num_experts = num_experts

        # Find indices where gates are non-zero
        nonzero_indices = gates.nonzero(as_tuple=False)  # shape: [nonzero_elements, 2]
        batch_indices = nonzero_indices[:, 0]  # Indices in batch dimension
        expert_indices = nonzero_indices[:, 1]  # Indices in expert dimension

        # Sort experts and get sorted batch indices
        sorted_expert_indices, sort_order = expert_indices.sort()
        self._expert_index = sorted_expert_indices
        self._batch_index = batch_indices[sort_order]

        # Calculate the number of samples per expert
        self._part_sizes = torch.bincount(self._expert_index, minlength=num_experts).tolist()

        # Get the corresponding gate values
        self._nonzero_gates = gates[self._batch_index, self._expert_index].unsqueeze(1)

    def dispatch(self, inp):
        """Create one input Tensor for each expert."""
        # Extract inputs corresponding to non-zero gates
        dispatched_inputs = inp[self._batch_index]

        # Split inputs for each expert
        expert_inputs = torch.split(dispatched_inputs, self._part_sizes, dim=0)
        return expert_inputs

    def combine(self, expert_outputs, multiply_by_gates=True):
        """Sum together the expert outputs, weighted by the gates."""
        # Concatenate expert outputs
        stitched = torch.cat(expert_outputs, dim=0)

        if multiply_by_gates:
            # Adjust dimensions for broadcasting
            gates = self._nonzero_gates.view(-1, 1, 1)
            stitched = stitched * gates

        # Initialize zeros tensor to accumulate results
        output = torch.zeros(self._gates.size(0), *stitched.size()[1:], device=stitched.device)

        # Accumulate outputs back to their original positions
        output.index_add_(0, self._batch_index, stitched)
        return output

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s."""
        return torch.split(self._nonzero_gates.squeeze(1), self._part_sizes, dim=0)

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.soft(out)
        return out


class RNNExpert(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.01, bidirectional=False):
        super(RNNExpert, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        factor = 2 if bidirectional else 1
        self.fc1 = nn.Linear(hidden_size * factor, 20)
        self.fc2 = nn.Linear(20, output_size)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]
        output, _ = self.rnn(x)
        # Get the last output from the sequence
        last_output = output[:, -1, :]  # Shape: [batch_size, hidden_size * factor]
        output = torch.tanh(self.fc1(last_output))
        output = self.fc2(output)
        return output

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                          nonlinearity='tanh')

        # 初始化 RNN 层权重
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.uniform_(param.data, -0.1, 0.1)
            elif 'weight_hh' in name:
                nn.init.uniform_(param.data, -0.1, 0.1)

        # 添加额外的全连接层
        self.fc1 = nn.Linear(hidden_size, 20)
        self.fc2 = nn.Linear(20, 1)

        # 正则化在优化器中处理
    def forward(self, x):
        x, _ = self.rnn(x)
        x = x[:, -1, :]  # 只使用序列的最后一个输出
        x = torch.tanh(self.fc1(x))  # 第一个全连接层使用 tanh 激活函数
        x = self.fc2(x)  # 输出层没有激活函数
        return x

class MoE(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, input_size, output_size, num_experts, hidden_size,seq_len, noisy_gating=True, k=4):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.k = k
        # instantiate experts
        # self.experts = nn.ModuleList([MLP(self.input_size, self.output_size, self.hidden_size) for i in range(self.num_experts)])
        self.experts = nn.ModuleList([SimpleRNN(input_size, hidden_size,output_size) for _ in range(num_experts)])
        self.num_experts = num_experts
        self.w_gate = nn.Parameter(torch.randn(input_size, num_experts))
        self.w_noise = nn.Parameter(torch.randn(input_size, num_experts))
        # self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        # self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """
        Adjusted to handle inputs with shape [batch_size * seq_len, num_experts]
        Computes the probability that each value in `noisy_values` is in the top k values of its batch element.
        """
        batch_seq_len, num_experts = clean_values.size()
        m = noisy_top_values.size(1)  # Assuming noisy_top_values is [batch_seq_len, m]
        k = self.k

        # Flatten top k values
        top_values_flat = noisy_top_values.view(-1)

        # Compute the indices for threshold values
        k_min = min(k, m - 1)
        threshold_positions_if_in = torch.arange(batch_seq_len, device=clean_values.device) * m + k_min
        threshold_if_in = top_values_flat[threshold_positions_if_in].view(batch_seq_len, 1)

        is_in = noisy_values > threshold_if_in  # shape [batch_seq_len, num_experts]

        # Compute threshold for values not in top k
        threshold_positions_if_out = torch.clamp(threshold_positions_if_in - 1, min=0)
        threshold_if_out = top_values_flat[threshold_positions_if_out].view(batch_seq_len, 1)

        # Compute probabilities
        normal = Normal(0, 1)
        eps = 1e-10
        noise_stddev = noise_stddev + eps  # Avoid division by zero

        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)

        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob  # shape: [batch_seq_len, num_experts]

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """
        Noisy top-k gating.
        """
        batch_size, input_size = x.size()
        clean_logits = x @ self.w_gate  # Shape: [batch_size, num_experts]

        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # Compute top k + 1 logits
        m = min(self.k + 1, self.num_experts)
        top_logits, top_indices = logits.topk(m, dim=1)

        # Normalize top k logits
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        # Create gates tensor
        zeros = torch.zeros_like(logits)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits).sum(0)
        else:
            load = self._gates_to_load(gates)

        return gates, load

    def forward(self, x, loss_coef=1e-2):
        """
        Args:
            x: tensor shape [batch_size, seq_len, input_size]
        """
        batch_size, seq_len, input_size = x.size()
        # Compute gates using the mean over the sequence dimension
        x_mean = x.mean(dim=1)  # Shape: [batch_size, input_size]
        gates, load = self.noisy_top_k_gating(x_mean, self.training)
        # **Compute importance and auxiliary loss**
        importance = gates.sum(0)  # Sum over the batch dimension
        loss = (self.cv_squared(importance) + self.cv_squared(load)) * loss_coef
        # Dispatcher
        dispatcher = SparseDispatcher(self.num_experts, gates)
        # Dispatch inputs without flattening
        expert_inputs = dispatcher.dispatch(x)
        # Pass to experts
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        # Combine outputs
        y = dispatcher.combine(expert_outputs)

        # Return the output and the auxiliary loss
        return y, loss

