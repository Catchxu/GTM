import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import argparse
from pathlib import Path
from typing import Optional
from geneformer import TranscriptomeTokenizer, EmbExtractor

from utils import update_configs_with_args
from configs import GeneformerConfigs


class GeneProcessor:
    def __init__(
        self,
        custom_attr_name_dict=None,
        nproc=16,
        chunk_size=512,
        model_input_size=2048,
        special_token=False,
        collapse_gene_ids=True,
        max_ncells=1000,
        forward_batch_size=16
    ):
        self.tk = TranscriptomeTokenizer(
            custom_attr_name_dict, 
            nproc,
            chunk_size,
            model_input_size,
            special_token,
            collapse_gene_ids
        )

        self.embex = EmbExtractor(
            model_type='Pretrained', 
            emb_mode='gene',
            max_ncells=max_ncells,
            forward_batch_size=forward_batch_size
        )
    
    def vocabularize(self, data_directory, output_directory, file_format, model_directory):
        self.tk.tokenize_data(
            data_directory, 
            output_directory, 
            file_format
        )

        input_directory = str(Path(output_directory).joinpath('data.dataset'))
        embs = self.embex.extract_embs(
            model_directory,
            input_directory,
            output_directory,
            'gene'
        )
        return embs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processing gene vocabulary with geneformer.')
    configs = GeneformerConfigs()

    parser.add_argument('--data_directory', type=Path, default='../data/',
                        help='Path to directory containing loom files or anndata files.')
    parser.add_argument('--output_directory', type=Path, default='../data/',
                        help='Path to directory where tokenized data and gene vocabulary will be saved.')
    parser.add_argument('--file_format', type=str, choices=['loom', 'h5ad'], default='h5ad',
                        help='Format of input files. Can be "loom" or "h5ad".')
    parser.add_argument('--model_directory', type=Path, default='../save_model/gf-20L-95M-i4096',
                        help='Path to directory containing model.')

    parser.add_argument('--custom_attr_name_dict', type=Optional[dict], default=configs.custom_attr_name_dict, 
                        help='Dictionary of custom attributes to be added to the dataset.')
    parser.add_argument('--nproc', type=int, default=configs.nproc, 
                        help='Number of processes to use for dataset mapping.')
    parser.add_argument('--chunk_size', type=int, default=configs.chunk_size, 
                        help='Chunk size for anndata tokenizer.')
    parser.add_argument('--model_input_size', type=int, default=configs.model_input_size, 
                        help='Max input size of model to truncate input to.')
    parser.add_argument('--special_token', type=bool, default=configs.special_token, 
                        help='Add CLS token before and EOS token after rank value encoding.')
    parser.add_argument('--collapse_gene_ids', type=bool, default=configs.collapse_gene_ids, 
                        help='Collapse gene IDs based on gene mapping dictionary.')
    parser.add_argument('--max_ncells', type=int, default=configs.max_ncells, 
                        help='Maximum number of cells to extract embeddings from.')    
    parser.add_argument('--forward_batch_size', type=int, default=configs.forward_batch_size, 
                        help='Batch size for forward pass.')

    args = parser.parse_args()
    args_dict = vars(args)
    update_configs_with_args(configs, args_dict, None)

    model = GeneProcessor(**configs.__dict__)
    embds = model.vocabularize(
        args.data_directory,
        args.output_directory,
        args.file_format,
        args.model_directory
    )
