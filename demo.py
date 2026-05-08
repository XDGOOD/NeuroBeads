"""
NeuroBeads Architecture Demonstration
=====================================
Scenario: Training a bead to recognize Sentiment (Positive/Negative)
and then using the Cross-Verification pipeline for inference.

Author: Alexander Farsiev (15 years old)
License: MIT
"""

import numpy as np
from neurobead_core import SpikingBead, CrossFilter, TripleMerge, Arbiter

# 1. Simple N-gram Encoder
def text_to_spikes(text, dim=128):
    """Converts text into a binary spike vector using word hashing."""
    vec = np.zeros(dim, dtype=bool)
    words = text.lower().split()
    for word in words:
        vec[hash(word) % dim] = True
    return vec

# 2. Setup System Components
print("--- Initializing NeuroBeads System ---")
# We use two beads to represent the "Dual-Thread" brain-like processing
bead_A = SpikingBead("Primary_Thread", n_neurons=128, seed=42)
bead_B = SpikingBead("Control_Thread", n_neurons=128, seed=99)

# Prepare beads for binary classification (0: Positive, 1: Negative)
bead_A.init_readout(n_classes=2)
bead_B.init_readout(n_classes=2)

cf = CrossFilter("Verification_Gate", agreement_threshold=0.85)
tm = TripleMerge("Consensus_Well")
arb = Arbiter(confidence_threshold=0.92, max_iterations=8)

# 3. Fast Training Phase (RLS)
print("\n--- Phase 1: Fast Online Training (RLS) ---")
training_set = [
    ("this is great and amazing", 0),
    ("i love this system", 0),
    ("terrible failure and bad", 1),
    ("i hate this error", 1)
]

for epoch in range(5):
    err = 0
    for text, label in training_set:
        spikes = text_to_spikes(text)
        # Train both threads to ensure they learn the same concepts
        proc_A = bead_A.forward(spikes)
        proc_B = bead_B.forward(spikes)
        err += bead_A.apply_rls(proc_A, label)
        err += bead_B.apply_rls(proc_B, label)
    if epoch % 2 == 0:
        print(f"Training Epoch {epoch} | System Error: {err/len(training_set):.4f}")

print("Training Complete. System is now specialized.")

# 4. Inference Phase with Cross-Verification
print("\n--- Phase 2: Live Inference with Verification ---")
user_request = "this is amazing"
input_spikes = text_to_spikes(user_request)

# Initial states
state_A, state_B = input_spikes.copy(), input_spikes.copy()

for i in range(arb.max_iterations):
    # Step 1: Spiking Dynamics
    state_A = bead_A.forward(state_A)
    state_B = bead_B.forward(state_B)
    
    # Step 2: Cross-Filtering (Finding Discrepancies)
    corr_A, corr_B, is_stable = cf.cross_correct(state_A, state_B)
    
    # Step 3: Triple Merge (Grounding with original input)
    new_A, new_B, consensus, score = tm.merge(state_A, state_B, input_spikes)
    
    # Step 4: Arbiter Decision
    decision, stability = arb.decide(consensus, score, i)
    
    print(f"Iteration {i+1}: Match Score: {score:.2f} | Stability: {stability:.2f} | Action: {decision}")
    
    if decision == "answer":
        # Final classification from the consensus spikes
        probs = bead_A.predict(consensus)
        label = "POSITIVE" if np.argmax(probs) == 0 else "NEGATIVE"
        print(f"\n✅ SUCCESS: Autonomous Answer Ready!")
        print(f"Result: {label} (Confidence: {np.max(probs)*100:.1f}%)")
        break
    elif decision == "llm":
        print("\n⚠️ ESCALATION: Consensus failed. Sending to Heavy LLM (Gemini).")
        break
    
    # Update states for next iteration
    state_A, state_B = new_A, new_B