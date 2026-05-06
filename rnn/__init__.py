"""
RNN Module - Recurrent Neural Network implementation and visualization
"""

from .engine import (
    SimpleRNN,
    generate_sine_sequence,
    train_rnn_model,
    get_rnn_architecture_info
)

__all__ = [
    'SimpleRNN',
    'generate_sine_sequence',
    'train_rnn_model',
    'get_rnn_architecture_info'
]
