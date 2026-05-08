import numpy as np
from collections import deque

# =================================================================
# 1. SPIKING BEAD (LIF Model with RLS Learning)
# =================================================================
class SpikingBead:
    """
    Basic unit of neuromorphic processing.
    A bead is a miniature reservoir of LIF (Leaky Integrate-and-Fire) neurons 
    that specializes in a specific task: ethics, logic, emotions, etc.
    
    Now includes a Linear Readout layer trained via Recursive Least Squares (RLS).
    """
    def __init__(self, name, n_neurons=128, n_columns=4, threshold=1.0, leak=0.9, seed=42):
        self.name = name
        self.n = n_neurons
        self.n_cols = n_columns
        self.threshold = threshold
        self.leak = leak

        rng = np.random.default_rng(seed)

        # Excitatory weights (W_exc): sparse recurrent connections (~10%)
        self.W_exc = rng.normal(0, 0.3, (n_neurons, n_neurons))
        mask = rng.random((n_neurons, n_neurons)) < 0.1
        self.W_exc *= mask
        np.fill_diagonal(self.W_exc, 0)

        # Inhibitory weights (W_inh): lateral inhibition between neural columns
        # Implements a winner-take-all competitive dynamic
        self.W_inh = np.zeros((n_neurons, n_neurons))
        col_size = n_neurons // n_columns
        for c in range(n_columns):
            col_mask = (np.arange(n_neurons) // col_size) == c
            self.W_inh[np.ix_(~col_mask, col_mask)] = -0.5

        # Neuronal states
        self.v = np.zeros(n_neurons)      # Membrane potential
        self.spikes = np.zeros(n_neurons, dtype=bool) # Spike output

        # RLS Readout Parameters (Initialized on demand)
        self.readout_weights = None
        self.P = None # Inverse correlation matrix
        self.n_classes = None

    def forward(self, input_spikes, feedback=None):
        """
        Single step of neural dynamics.
        Integrates input, recurrent excitation, and lateral inhibition.
        """
        current = input_spikes.astype(float)
        if feedback is not None:
            current += feedback.astype(float) * 0.5

        # LIF Dynamics: Leak -> Recurrent Input -> External Input
        self.v = self.leak * self.v + self.W_exc @ self.spikes + current
        self.v = np.clip(self.v, -1.5, 2.0)

        # Apply lateral inhibition
        self.v += self.W_inh @ self.spikes

        # Thresholding: Fire if potential > threshold
        self.spikes = self.v >= self.threshold

        # Winner-Take-All (WTA) within each column
        col_size = self.n // self.n_cols
        for c in range(self.n_cols):
            col_slice = slice(c * col_size, (c + 1) * col_size)
            if np.any(self.spikes[col_slice]):
                winner = c * col_size + np.argmax(self.v[col_slice])
                self.spikes[col_slice] = False
                self.spikes[winner] = True

        output = self.spikes.copy()
        self.v[self.spikes] = 0.0 # Reset potential after spike
        return output

    # --- Learning Block (RLS) ---
    def init_readout(self, n_classes=2):
        """Initializes the linear readout for online learning."""
        self.n_classes = n_classes
        self.readout_weights = np.random.randn(n_classes, self.n) * 0.01
        self.P = np.eye(self.n) * 10.0 # High initial uncertainty

    def predict(self, spikes):
        """Converts spike vector to class probabilities using Softmax."""
        if self.readout_weights is None:
            raise ValueError("Readout not initialized. Call init_readout() first.")
        
        logits = self.readout_weights @ spikes.astype(float)
        # Numerical stability for Softmax
        exp_l = np.exp(logits - np.max(logits))
        return exp_l / exp_l.sum()

    def apply_rls(self, spikes, target_idx, forget_factor=0.99):
        """
        One-shot training step using Recursive Least Squares.
        Rapidly adapts weights to associate spike patterns with labels.
        """
        x = spikes.astype(float).reshape(-1, 1)
        
        # Build target vector (one-hot)
        y_target = np.zeros((self.n_classes, 1))
        y_target[target_idx] = 1.0
        
        # Calculate error
        y_pred = (self.readout_weights @ x)
        error = y_target - y_pred
        
        # Update P matrix (Memory/Correlation)
        Px = self.P @ x
        gain = Px / (forget_factor + x.T @ Px)
        self.P = (self.P - gain @ (x.T @ self.P)) / forget_factor
        
        # Adjust Readout weights
        self.readout_weights += error @ gain.T
        return np.abs(error).mean()

# =================================================================
# 2. CROSS-FILTER
# =================================================================
class CrossFilter:
    """
    Compares two independent processing threads (A and B).
    Identifies zones of agreement and provides feedback on discrepancies.
    """
    def __init__(self, name, agreement_threshold=0.9):
        self.name = name
        self.agreement_threshold = agreement_threshold
        self.last_disagreement_A = None
        self.last_disagreement_B = None

    def cross_correct(self, out_A, out_B):
        agreement = out_A & out_B
        union = out_A | out_B
        
        total_active = np.sum(union)
        match_score = np.sum(agreement) / max(1, total_active)
        
        # Track differences for potential feedback loops
        self.last_disagreement_A = out_A & ~out_B
        self.last_disagreement_B = out_B & ~out_A
        
        is_stable = match_score >= self.agreement_threshold
        return agreement, agreement, is_stable

# =================================================================
# 3. TRIPLE MERGE (The "000" Node)
# =================================================================
class TripleMerge:
    """
    Fusion node that integrates Thread A, Thread B, and Grounding input.
    Ensures the final consensus is tethered to the original input.
    """
    def __init__(self, name="TripleMerge"):
        self.name = name

    def merge(self, out_A, out_B, original_input=None):
        # Find majority agreement
        consensus = (out_A & out_B)
        
        # Grounding: Add bits from original input if validated by any thread
        if original_input is not None:
            consensus |= (original_input & (out_A | out_B))
        
        total_ab = np.sum(out_A | out_B)
        match_score = np.sum(out_A & out_B) / max(1, total_ab)
        
        return out_A, out_B, consensus, match_score

# =================================================================
# 4. ARBITER
# =================================================================
class Arbiter:
    """
    Final decision maker. Evaluates consensus stability to determine:
    - Return answer (High confidence)
    - Continue iterations (Deepen processing)
    - Escalate to Heavy LLM (Ambiguity detected)
    """
    def __init__(self, confidence_threshold=0.95, max_iterations=12):
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.history = []

    def decide(self, consensus, match_score, iteration):
        self.history.append((consensus.copy(), match_score))
        
        # Calculate stability over time
        if iteration >= 1:
            prev_consensus = self.history[-2][0]
            stability = np.sum(consensus == prev_consensus) / len(consensus)
        else:
            stability = 1.0

        if match_score >= self.confidence_threshold and iteration >= 2:
            return "answer", stability
        elif iteration >= self.max_iterations:
            return "llm", stability # Escalation to cloud
        else:
            return "continue", stability