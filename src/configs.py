class TMConfigs:
    def __init__(self):
        self.gene_size = 1000
        self.act = 'relu'
        self.enc_drop = 0.5
        self.device = 'cuda:0'
        self.batch_size = 32
        self.learning_rate = 1e-3
        self.weight_decay = 1e-6
        self.clip = 0
        self.verbose = 1