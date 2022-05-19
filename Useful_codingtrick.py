volumes = sum_tensor(y_onehot, axes) + 1e-6 # add some eps to prevent div by zero
