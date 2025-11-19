
# Training hyperparameters
n_pde_samples_per_batch = 2048*4
n_boundary_samples      = 512*4
n_stopping_samples      = 512*4
n_terminal_samples      = 1024*4
n_far_samples           = 512*4

max_epochs      = 20000
learning_rate = 1e-3


# loss function weights
w_pde = 1.0
w_vm = 10.0
w_sp = 5.0  # slightly reduced, since smooth pasting is stiff at small t
w_stop = 5.0  # new stopping region weight
w_term = 5.0
w_far = 1.0