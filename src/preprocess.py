import argparse
from pathlib import Path
from typing import Optional
from geneformer import TranscriptomeTokenizer

from utils import update_configs_with_args
from configs import GeneformerTokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing for geneformer.')
    configs = GeneformerTokenizer()

    parser.add_argument('--custom_attr_name_dict', type=Optional[dict], default=configs.custom_attr_name_dict, 
                        help='Dictionary of custom attributes to be added to the dataset.')
    parser.add_argument('--nproc', type=int, default=configs.nproc, 
                        help='Number of processes to use for dataset mapping.')
    parser.add_argument('--chunk_size', type=int, default=configs.chunk_size, 
                        help='Chunk size for anndata tokenizer.')
    parser.add_argument('--model_input_size', type=int, default=configs.model_input_size, 
                        help='Max input size of model to truncate input to.')
    parser.add_argument('--special_token', action='store_true', default=configs.special_token, 
                        help='Add CLS token before and EOS token after rank value encoding.')
    parser.add_argument('--collapse_gene_ids', action='store_true', default=configs.collapse_gene_ids, 
                        help='Collapse gene IDs based on gene mapping dictionary.')
    
    parser.add_argument('--data_directory', type=Path, default='../data/',
                        help='Path to directory containing loom files or anndata files.')
    parser.add_argument('--output_directory', type=Path, default='../data/',
                        help='Path to directory where tokenized data will be saved as .dataset.')
    parser.add_argument('--file_format', type=str, choices=['loom', 'h5ad'], default='h5ad',
                        help='Format of input files. Can be "loom" or "h5ad".')

    args = parser.parse_args()
    args_dict = vars(args)
    update_configs_with_args(configs, args_dict, None)

    tk = TranscriptomeTokenizer(**configs.__dict__)

    # Pass arguments to tk.tokenize_data
    tk.tokenize_data(
        data_directory=args.data_directory,
        output_directory=args.output_directory,
        file_format=args.file_format
    )

