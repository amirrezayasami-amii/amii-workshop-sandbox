"""Central hyperparameters — the single source of truth for training.

Keeping every knob in one place makes experiments reproducible and gives the
workshop a clean, high-traffic file to practise on. It is deliberately the
file two feature branches both want to edit, so it doubles as *conflict bait*
for the Git & GitHub module.
"""

# --- Optimisation -----------------------------------------------------------
EPOCHS = 50                    # feature/faster-epochs: shorter runs
BATCH_SIZE = 256
LEARNING_RATE = 5e-4          # staging: lowered for stability

# --- Data -------------------------------------------------------------------
TEST_SIZE = 0.2               # fraction of rows held out for evaluation
SEED = 42

# --- Architecture -----------------------------------------------------------
HIDDEN_LAYERS = (64, 32)      # widths of the MLP hidden layers
