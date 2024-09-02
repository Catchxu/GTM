import pandas as pd
import torch
import anndata as ad
import sys
from topic import TopicModel
from configs import TopicConfigs
from topic import TopicModel
from torch.utils.data import DataLoader
import torch.nn.functional as F

args = TopicConfigs()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#数据导入与处理
gene_embeddings = pd.read_csv('../../data/gene.csv', index_col=0)
#print(gen_embeddings.dtypes)

adata=ad.read_h5ad('../../data/PBMC.h5ad')
gene_embeddings_genes = gene_embeddings.index  
gene_data_genes = adata.var_names 

common_genes = gene_embeddings_genes.intersection(gene_data_genes)
gene_counts_common = adata[:, common_genes]
gene_embeddings_common = gene_embeddings.loc[common_genes].to_numpy()
gene_embeddings = torch.tensor(gene_embeddings_common, dtype=torch.float64).to(device)
model = TopicModel(embeddings=gene_embeddings, num_topics=50, args=args)  
model.to(device)

# load model
model.load_state_dict(torch.load('./saved_model/solution2.pth'))
model.eval()  

beta = model.get_beta()
beta = beta.to(torch.float64)
topic_embeddings = torch.matmul(beta, gene_embeddings)

torch.save(topic_embeddings, './topic embeddings/solution2.pth')