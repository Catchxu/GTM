import pickle
import torch
import pandas as pd
import scanpy as sc

from model import TopicModel
from configs import TopicConfigs


if __name__ == '__main__':
    args = TopicConfigs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    with open('../data/gene.pkl', 'rb') as f:
        gene_embeddings = pickle.load(f)

    adata = sc.read_h5ad('../data/PBMC.h5ad')
    gene_embeddings_genes = gene_embeddings.index
    gene_data_genes = adata.var_names

    common_genes = gene_embeddings_genes.intersection(gene_data_genes)
    gene_counts_common = adata[:, common_genes]
    gene_embeddings_common = gene_embeddings.loc[common_genes].to_numpy()
    gene_embeddings = torch.tensor(gene_embeddings_common, dtype=torch.float32).to(device)

    #train topic_model
    model = TopicModel(embeddings=gene_embeddings, num_topics=200, args=args)
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train_one_epoch(epoch, gene_counts_common)

    # save model
    torch.save(model.state_dict(), '../save_model/GTM.pth')

    beta = model.get_beta().to(device)

    # conut topic embedding
    topic_embeddings = torch.matmul(beta.to(device), gene_embeddings.to(device))

    # save topic embedding
    # torch.save(topic_embeddings, )
    with open('../data/topic.pkl', 'wb') as f:
        pickle.dump(topic_embeddings, f)