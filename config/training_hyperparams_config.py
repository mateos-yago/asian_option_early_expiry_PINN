
# Training hyperparameters
n_pde_samples_per_batch = 2048*4
n_boundary_samples      = 512*4
n_stopping_samples      = 512*4
n_terminal_samples      = 1024*4
n_far_samples           = 512*4

max_epochs      = 20000
learning_rate = 1e-3

# ============================================================
# Stopping / monitoring hyperparameters
# ============================================================
target_total_loss = 1.0e-4  # stop early if total loss goes below this
target_pde_loss = 1.0e-4  # and PDE loss is also below this

patience = 5000  # number of epochs with no sufficient improvement before stopping
min_delta = 1.0e-5  # minimum improvement in loss to reset patience


# loss function weights
w_pde = 20.0
w_vm = 10.0
w_sp = 5.0
w_stop = 5.0
w_term = 7.0
w_far = 1.0