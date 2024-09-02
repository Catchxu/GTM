import pandas as pd
import torch
import anndata as ad
import numpy as np
import sys
from topic import TopicModel
from configs import TopicConfigs
from topic import TopicModel
from torch.utils.data import DataLoader
import torch.nn.functional as F

args = TopicConfigs()


#load data 
gene_embeddings = pd.read_csv('../../data/gene.csv', index_col=0)
#print(gen_embeddings.dtypes)

adata=ad.read_h5ad('../..//data/PBMC.h5ad')
gene_expression_total = np.array(adata.X.sum(axis=0)).flatten()
gene_embeddings_genes = gene_embeddings.index  
gene_data_genes = adata.var_names 
common_genes = gene_embeddings_genes.intersection(gene_data_genes)
gene_counts_common = adata[:, common_genes]
gene_embeddings_common = gene_embeddings.loc[common_genes].to_numpy()
gene_embeddings = torch.tensor(gene_embeddings_common, dtype=torch.float64)


#train topic_model
model = TopicModel(embeddings=gene_embeddings, num_topics=50, args=args)

model.train_one_epoch(epoch=0, adata=gene_counts_common)

# save model
torch.save(model.state_dict(), './saved_model/solution2.pth')
