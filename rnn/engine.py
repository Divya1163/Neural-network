"""
RNN Engine - Simple RNN implementation for training visualization
Exports RNN training and architecture information for the Flask dashboard
"""

import numpy as np
import json
from typing import Dict, List, Tuple


class SimpleRNN:
    """
    Manual implementation of a simple RNN (Vanilla RNN) for sequence prediction.
    """
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        scale = 0.01
        self.W_xh = np.random.randn(hidden_size, input_size) * scale
        self.W_hh = np.random.randn(hidden_size, hidden_size) * scale
        self.W_hy = np.random.randn(output_size, hidden_size) * scale
        
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))
        
        self.h_prev = np.zeros((hidden_size, 1))
        
    def forward(self, X, training=False):
        seq_len = X.shape[0]
        h_states = np.zeros((seq_len + 1, self.hidden_size, 1))
        h_states[0] = self.h_prev.copy()
        
        outputs = []
        
        for t in range(seq_len):
            x_t = X[t].reshape(-1, 1) if X[t].ndim == 1 else X[t]
            
            h_t = np.tanh(
                np.dot(self.W_hh, h_states[t]) +
                np.dot(self.W_xh, x_t) +
                self.b_h
            )
            h_states[t + 1] = h_t
            
            y_t = np.dot(self.W_hy, h_t) + self.b_y
            outputs.append(y_t)
        
        outputs = np.array([o.flatten() for o in outputs])
        
        if training:
            self.cached_h_states = h_states
            self.cached_outputs = outputs
            self.cached_X = X
        
        self.h_prev = h_states[-1].copy()
        
        return outputs, h_states
    
    def backward(self, dL_dy):
        seq_len = dL_dy.shape[0]
        h_states = self.cached_h_states
        X = self.cached_X
        
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        
        dh_next = np.zeros((self.hidden_size, 1))
        
        for t in reversed(range(seq_len)):
            dy_t = dL_dy[t].reshape(-1, 1)
            
            dW_hy += np.dot(dy_t, h_states[t + 1].T)
            db_y += dy_t
            
            dh_t = np.dot(self.W_hy.T, dy_t) + dh_next
            dh_raw = dh_t * (1 - h_states[t + 1] ** 2)
            
            dW_hh += np.dot(dh_raw, h_states[t].T)
            
            x_t = X[t].reshape(-1, 1) if X[t].ndim == 1 else X[t]
            dW_xh += np.dot(dh_raw, x_t.T)
            
            db_h += dh_raw
            dh_next = np.dot(self.W_hh.T, dh_raw)
        
        max_grad = 5.0
        for dparam in [dW_xh, dW_hh, dW_hy, db_h, db_y]:
            np.clip(dparam, -max_grad, max_grad, out=dparam)
        
        return dW_xh, dW_hh, dW_hy, db_h, db_y
    
    def update_weights(self, dW_xh, dW_hh, dW_hy, db_h, db_y):
        self.W_xh -= self.learning_rate * dW_xh
        self.W_hh -= self.learning_rate * dW_hh
        self.W_hy -= self.learning_rate * dW_hy
        self.b_h -= self.learning_rate * db_h
        self.b_y -= self.learning_rate * db_y
    
    def train_step(self, X, y):
        outputs, _ = self.forward(X, training=True)
        loss = np.mean((outputs - y) ** 2)
        dL_dy = 2 * (outputs - y) / y.shape[0]
        dW_xh, dW_hh, dW_hy, db_h, db_y = self.backward(dL_dy)
        self.update_weights(dW_xh, dW_hh, dW_hy, db_h, db_y)
        return loss
    
    def predict(self, X):
        outputs, _ = self.forward(X, training=False)
        return outputs
    
    def reset_hidden_state(self):
        self.h_prev = np.zeros((self.hidden_size, 1))


def generate_sine_sequence(num_sequences=100, seq_length=15, freq=0.5):
    """Generate sine wave training data."""
    sequences = []
    targets = []
    
    for i in range(num_sequences):
        phase = np.random.rand() * 2 * np.pi
        t = np.arange(seq_length + 1)
        y = np.sin(2 * np.pi * freq * t / seq_length + phase)
        
        X_seq = y[:-1].reshape(-1, 1)
        y_seq = y[1:].reshape(-1, 1)
        
        sequences.append(X_seq)
        targets.append(y_seq)
    
    return sequences, targets


def train_rnn_model(num_epochs=50, hidden_size=8, learning_rate=0.01, seq_length=15):
    """
    Train RNN model and return training history for visualization.
    
    Returns:
        Dict with training info, losses, and model parameters
    """
    train_X, train_y = generate_sine_sequence(100, seq_length)
    
    rnn = SimpleRNN(input_size=1, hidden_size=hidden_size, output_size=1, learning_rate=learning_rate)
    
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        
        for seq_idx in range(len(train_X)):
            loss = rnn.train_step(train_X[seq_idx], train_y[seq_idx])
            epoch_loss += loss
            rnn.reset_hidden_state()
        
        epoch_loss /= len(train_X)
        losses.append(float(epoch_loss))
    
    # Test predictions
    test_X, test_y = generate_sine_sequence(5, seq_length)
    test_losses = []
    test_preds = []
    
    for seq_idx in range(len(test_X)):
        pred = rnn.predict(test_X[seq_idx])
        loss = float(np.mean((pred - test_y[seq_idx]) ** 2))
        test_losses.append(loss)
        test_preds.append(pred.tolist())
        rnn.reset_hidden_state()
    
    return {
        'losses': losses,
        'avg_test_loss': float(np.mean(test_losses)),
        'test_predictions': test_preds,
        'num_epochs': num_epochs,
        'hidden_size': hidden_size,
        'learning_rate': learning_rate,
        'final_loss': losses[-1] if losses else 0,
        'initial_loss': losses[0] if losses else 0
    }


def get_rnn_architecture_info() -> Dict:
    """Get RNN architecture information for visualization."""
    return {
        'layers': [
            {
                'name': 'Input Layer',
                'units': 1,
                'type': 'input',
                'description': 'Single time step input'
            },
            {
                'name': 'Hidden Layer (Recurrent)',
                'units': 8,
                'type': 'hidden',
                'description': 'Processes input + previous hidden state',
                'activation': 'tanh'
            },
            {
                'name': 'Output Layer',
                'units': 1,
                'type': 'output',
                'description': 'Prediction for current timestep'
            }
        ],
        'connections': [
            {
                'from': 'Input',
                'to': 'Hidden',
                'weight_name': 'W_xh',
                'description': 'Input-to-Hidden weights'
            },
            {
                'from': 'Hidden (t-1)',
                'to': 'Hidden (t)',
                'weight_name': 'W_hh',
                'description': 'Recurrent weights (temporal dependency)'
            },
            {
                'from': 'Hidden',
                'to': 'Output',
                'weight_name': 'W_hy',
                'description': 'Hidden-to-Output weights'
            }
        ],
        'equations': {
            'hidden_state': 'h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b_h)',
            'output': 'y_t = W_hy @ h_t + b_y',
            'loss': 'L = (1/T) * sum(||y_t - target_t||^2)'
        },
        'total_parameters': '8 + 64 + 8 + 8 + 8 + 1 + 1 = 98 parameters'
    }
