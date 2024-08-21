class GeneformerTokenizer:
    def __init__(self):
        self.custom_attr_name_dict = {'cell_type': 'cell_type', 'organism': 'organ'}
        self.nproc = 16
        self.chunk_size = 512
        self.model_input_size = 2048
        self.special_token = False
        self.collapse_gene_ids = True