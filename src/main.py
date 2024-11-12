import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import pickle
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt

from model import TopicModel
from configs import TopicConfigs


class MoTM(object):
    def __init__(self, vocabulary: np.ndarray, args: TopicConfigs):
        """
        Mixture Of Topic Models in Gene Embedding Space.

        Parameters
        ----------
        vocabulary: np.ndarray
            Pretrained gene vocabulary embeddings with shape `num_genes x gene_dim`.
        args: 
            Model arguments.
        """

        vocabulary = torch.tensor(vocabulary, dtype=torch.float32)
        self.model = TopicModel(vocabulary, args=args)
        self.num_epochs = args.num_epochs
    
    def train(self, adata: ad.AnnData):
        # train topic_model
        for idx in range(self.num_epochs):
            self.model.train_one_epoch(idx, adata)
        
        # save model
        torch.save(self.model.state_dict(), './save_model/MoTM.pth')
    
    def C2T_heatmap(self, query: ad.AnnData):
        C2T = self.model.call_C2T(query)
        num_topics = C2T.shape[1]
        topic_df = pd.DataFrame(C2T, columns=[f"Topic_{i+1}" for i in range(num_topics)])
        topic_df['cell_type'] = query.obs['cell_type'].values
        avg_topic_df = topic_df.groupby('cell_type', observed=False).mean()
        adata_topics = sc.AnnData(X=avg_topic_df.values)
        adata_topics.obs['cell_type'] = avg_topic_df.index

        sc.pl.heatmap(
            adata_topics,
            var_names=adata_topics.var_names,
            groupby='cell_type',
            cmap='RdBu_r',
            standard_scale='var',
            swap_axes=True,
            show=False)
        plt.savefig('./results/PBMC_C2T_heatmap_avg.png', dpi=300, bbox_inches='tight')

        topic_df = pd.DataFrame(C2T, columns=[f"Topic_{i+1}" for i in range(num_topics)])
        adata_topics = sc.AnnData(topic_df)
        adata_topics.obs['cell_type'] = query.obs['cell_type'].values

        sc.pl.heatmap(
            adata_topics,
            var_names=adata_topics.var_names,
            groupby='cell_type',
            cmap='RdBu_r',
            standard_scale='var',
            swap_axes=True,
            show=False)
        plt.savefig('./results/PBMC_C2T_heatmap.png', dpi=300, bbox_inches='tight')

    def G2T_heatmap(self):
        G2T = self.model.call_G2T(self.model.vocabulary)
        num_topics = G2T.shape[1]
        topic_df = pd.DataFrame(G2T, columns=[f"Topic_{i+1}" for i in range(num_topics)])
        adata_topics = sc.AnnData(topic_df)
        adata_topics.obs['dummy_group'] = 'gene'

        sc.pl.heatmap(
            adata_topics,
            var_names=adata_topics.var_names,
            groupby='dummy_group',
            cmap='RdBu_r',
            standard_scale='var',
            swap_axes=True,
            show=False)
        plt.savefig('./results/PBMC_G2T_heatmap.png', dpi=300, bbox_inches='tight')




if __name__ == '__main__':
    args = TopicConfigs()

    # load data
    with open('./data/gene.pkl', 'rb') as f:
        gene_embeddings = pickle.load(f)

    adata = sc.read_h5ad('./data/PBMC.h5ad')
    gene_embeddings_genes = gene_embeddings.index
    gene_data_genes = adata.var_names

    common_genes = gene_embeddings_genes.intersection(gene_data_genes)
    adata_common = adata[:, common_genes]
    gene_embeddings_common = gene_embeddings.loc[common_genes].to_numpy()

    Model = MoTM(gene_embeddings_common, args)
    Model.train(adata_common)

    # visualization
    Model.C2T_heatmap(adata_common)
    Model.G2T_heatmap()
