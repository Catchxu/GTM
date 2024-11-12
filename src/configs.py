class GeneformerConfigs:
    def __init__(self):
        self.custom_attr_name_dict = {'cell_type': 'cell_type', 'organism': 'organ'}
        self.nproc = 16
        self.chunk_size = 512
        self.model_input_size = 2048
        self.special_token = False
        self.collapse_gene_ids = True
        self.max_ncells = 1000
        self.forward_batch_size = 16


class TopicConfigs:
    def __init__(self):
        self.device = 'cuda'
        self.batch_size = 32
        self.learning_rate = 5e-4
        self.weight_decay = 1e-6
        self.clip = 0
        self.verbose = 1

        self.Q_hidden_dim = [1024, 512, 256]
        self.Q_dropout = 0.3

        self.num_topics = 20
        self.topk = 4
        self.noisy_gating = True
        self.GEGLU_mult_bias = True
        self.RMSNorm = True