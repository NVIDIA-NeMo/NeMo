import torch


@torch.jit.script
def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))


def openai_gelu(x):
    return gelu_impl(x)


@torch.jit.script
def squared_relu(x):
    """Squared ReLU activation function."""
    return torch.pow(torch.nn.functional.relu(x), 2)
