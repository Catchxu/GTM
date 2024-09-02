import pandas as pd
import torch
import anndata as ad
import sys
from .topic import TopicModel
from .configs import TopicConfigs
from torch.utils.data import DataLoader
import torch.nn.functional as F

args = TopicConfigs()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load data
gene_embeddings = pd.read_csv('../../data/gene.csv', index_col=0)
#print(gen_embeddings.dtypes)

adata=ad.read_h5ad('../../data/PBMC.h5ad')
gene_embeddings_genes = gene_embeddings.index  
gene_data_genes = adata.var_names 

common_genes = gene_embeddings_genes.intersection(gene_data_genes)
gene_counts_common = adata[:, common_genes]
gene_embeddings_common = gene_embeddings.loc[common_genes].to_numpy()
gene_embeddings = torch.tensor(gene_embeddings_common, dtype=torch.float64).to(device)


#train topic_model
model = TopicModel(embeddings=gene_embeddings, num_topics=50, args=args)
num_epochs = 10
for epoch in range(num_epochs):
    model.train_one_epoch(epoch, gene_counts_common)

# save model
torch.save(model.state_dict(), './saved_model/solution1.pth')


beta = model.get_beta().to(device)

# conut topic embedding
topic_embeddings = torch.matmul(beta.to(device), gene_embeddings.to(device))

# save topic embedding
torch.save(topic_embeddings, './topic_embedding/solution1_topic_embeddings.pth')
