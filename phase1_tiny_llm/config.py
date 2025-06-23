class Config:
    vocab_size = 50257
    d_model = 128
    n_heads = 4
    d_ff = 4 * d_model
    n_layers = 2
    context_length = 256
    dropout = 0.1

    batch_size = 16
    learning_rate = 3e-4
    num_epochs = 10
