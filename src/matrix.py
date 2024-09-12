import torch
import pandas as pd
import scanpy as sc
import pickle
from model import TopicModel
from configs import TopicConfigs
from scipy.sparse import issparse


args = TopicConfigs()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

adata = sc.read_h5ad('../data/PBMC.h5ad')
with open('../data/gene.pkl', 'rb') as f:
    gene_embeddings = pickle.load(f)
    gene_embeddings.set_index(gene_embeddings.columns[0], inplace=True)

gene_embeddings_genes = gene_embeddings.index
gene_data_genes = adata.var_names
common_genes = gene_embeddings_genes.intersection(gene_data_genes)
gene_counts_common = adata[:, common_genes]
gene_embeddings_common = gene_embeddings.loc[common_genes].to_numpy()
gene_embeddings = torch.tensor(gene_embeddings_common, dtype=torch.float32).to(device)

# load model
model = TopicModel(embeddings=gene_embeddings, num_topics=200, args=args)
model.load_state_dict(torch.load('../save_model/GTM.pth'))
model.eval() 

# Single Cell-Topics Matrix
sc.pp.normalize_total(gene_counts_common, target_sum=1) 
if issparse(gene_counts_common.X):
    gene_counts_tensor = torch.tensor(gene_counts_common.X.toarray(), dtype=torch.float32).to(device)
else:
    gene_counts_tensor = torch.tensor(gene_counts_common.X, dtype=torch.float32).to(device)

theta, _ = model.get_theta(gene_counts_tensor)  
single_cell_topics_matrix = theta.detach().cpu().numpy() 

# Topic-Gene Similarity Matrix
beta = model.get_beta().detach().cpu().numpy()  


pd.DataFrame(single_cell_topics_matrix, index=gene_counts_common.obs_names).to_csv('../matrix/single_cell x topics.csv')
pd.DataFrame(beta, columns=common_genes).to_csv('../matrix/topics x gene.csv')


