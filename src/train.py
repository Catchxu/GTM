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
    with open('./data/gene.pkl', 'rb') as f:
        gene_embeddings = pickle.load(f)

    adata = sc.read_h5ad('./data/PBMC.h5ad')
    gene_embeddings_genes = gene_embeddings.index
    gene_data_genes = adata.var_names

    common_genes = gene_embeddings_genes.intersection(gene_data_genes)
    adata_common = adata[:, common_genes]
    gene_embeddings_common = gene_embeddings.loc[common_genes].to_numpy()
    vocabulary = torch.tensor(gene_embeddings_common, dtype=torch.float32).to(device)

    # train topic_model
    model = TopicModel(vocabulary, args=args)
    num_epochs = 0
    for epoch in range(num_epochs):
        model.train_one_epoch(epoch, adata_common)

    # save model
    torch.save(model.state_dict(), './save_model/GTM.pth')

    # query topic proportions for each cell
    C2T = model.call_C2T(adata_common)
    # adata_common.obsm['topic_proportion'] = C2T
    
    num_topics = C2T.shape[1]
    topic_df = pd.DataFrame(C2T, columns=[f"Topic_{i+1}" for i in range(num_topics)])
    adata_topics = sc.AnnData(X=topic_df.values)
    adata_topics.obs = adata_common.obs.copy()

    sc.pl.heatmap(
        adata_topics,
        var_names=adata_topics.var_names,
        groupby='cell_type',
        cmap='RdBu',
        standard_scale='var',
        save='./result/PBMC_C2T_heatmap.png'
    )
