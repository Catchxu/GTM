import anndata as ad
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.sparse import issparse

from utils import get_device
from configs import TopicConfigs
from .moe import MixtureOfExperts


class QEncoder(nn.Module):
    def __init__(self, num_genes: int, args: TopicConfigs):
        super().__init__()

        hidden_dim = args.Q_hidden_dim
        dim_1 = num_genes

        modules = []
        for dim_2 in hidden_dim:
            modules.append(
                nn.Sequential(
                    nn.Linear(dim_1, dim_2),
                    nn.BatchNorm1d(dim_2),
                    nn.LeakyReLU()
                )    
            )
            dim_1 = dim_2
        
        if args.Q_dropout > 0:
            modules.append(nn.Dropout(args.Q_dropout))
        self.encoder = nn.Sequential(*modules)

        self.mu = nn.Linear(dim_2, args.num_topics)
        self.logsigma = nn.Linear(dim_2, args.num_topics)

    def forward(self, gene_counts: torch.Tensor):
        """
        Get paramters of the variational distribution.

        Parameters
        ----------
        gene_counts: torch.Tensor
            Read counts with shape `batch_size x num_genes`.
        """
        gene_counts = gene_counts
        q = self.encoder(gene_counts)

        mu = self.mu(q)  # batch_size x num_topics
        logsigma = self.logsigma(q)
        kl = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp(), dim=-1).mean()
        return mu, logsigma, kl


class TopicModel(nn.Module):
    def __init__(self, vocabulary: torch.Tensor, args: TopicConfigs):
        """
        Topic Model in Gene Embedding Space.

        Parameters
        ----------
        vocabulary: torch.Tensor
            Pretrained gene vocabulary embeddings with shape `num_genes x gene_dim`.
        args: 
            Model arguments.
        """
        super().__init__()

        num_genes, gene_dim = vocabulary.size()
        self.num_genes = num_genes
        self.gene_dim = gene_dim
        self.num_topics = args.num_topics

        # model
        self.device = get_device(args.device)
        self.vocabulary = vocabulary.clone().float().to(self.device)
        self.encoder = QEncoder(num_genes, args).to(self.device)
        self.moe = MixtureOfExperts(gene_dim, args).to(self.device)

        # training
        self.batch_size = args.batch_size
        self.optim = torch.optim.Adam(
            self.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
            )
        self.clip = args.clip
        self.verbose = args.verbose

    def reparameterize(self, mu, logvar):
        """
        Draw a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * logvar) 
            eps = torch.randn_like(std)
            delta = eps.mul_(std).add_(mu)
            return delta
        else:
            return mu

    def call_beta(self):
        """
        Get the topic distribution (`beta`) for the genes.
        """
        _, scores, _ = self.moe(self.vocabulary)
        beta = scores.transpose(1, 0)
        return beta

    def call_theta(self, gene_counts):
        """
        Get the topic poportion (`theta`) for the cells.
        """
        mu_theta, logsigma_theta, kl_theta = self.encoder(gene_counts)
        delta = self.reparameterize(mu_theta, logsigma_theta)
        theta = F.softmax(delta, dim=-1)
        return theta, kl_theta

    def decode(self, theta, beta):
        """
        Compute the probability of topic given the read counts.
        """
        res = torch.mm(theta, beta)  # (BS, topics) x (topics, genes)
        almost_zeros = torch.full_like(res, 1e-6)
        prob = res.add(almost_zeros)
        return prob

    def forward(self, gene_counts, aggregate=True):
        theta, kl_loss = self.call_theta(gene_counts)
        beta = self.call_beta()

        logp = torch.log(self.decode(theta, beta))
        recon_loss = - (logp * gene_counts).sum(1)
        if aggregate:
            recon_loss = recon_loss.mean()
        return recon_loss, kl_loss
    
    def train_one_batch(self, batch):
        batch = batch.to(self.device)

        self.optim.zero_grad()
        self.zero_grad()

        recon_loss, kl_theta = self.forward(batch)
        loss = recon_loss + kl_theta
        loss.backward()
        if self.clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip)
        self.optim.step()
        return loss

    def preprocess(self, adata: ad.AnnData):
        sc.pp.normalize_total(adata, target_sum=1)
        
        if issparse(adata.X):
            dataset = torch.Tensor(adata.X.toarray())
        else:
            dataset = torch.Tensor(adata.X)

        return dataset    

    def train_one_epoch(self, epoch: int, adata: ad.AnnData):
        """
        Train topic model with one epoch.

        Parameters
        ----------
        epoch: int
            Training epoch ID.
        adata: AnnData
            Training dataset.
        """
        self.train()
        dataset = self.preprocess(adata)
        loader = DataLoader(dataset, self.batch_size, shuffle=True)

        if self.verbose:
            with tqdm(total=len(loader)) as t:
                t.set_description(f'Training Epochs {epoch}')
                for _, batch in enumerate(loader):
                    loss = self.train_one_batch(batch)
                    t.set_postfix(Loss=loss.item())
                    t.update(1)
        else:
            for _, batch in enumerate(loader):
                _ = self.train_one_batch(batch)
    
    @torch.no_grad()
    def call_C2T(self, adata: ad.AnnData):
        """
        Call the assignment matrix with shape `cells x topics`

        Parameters
        ----------
        adata: AnnData
            Query dataset.
        """
        dataset = self.preprocess(adata)
        loader = DataLoader(dataset, self.batch_size*5, shuffle=False)

        cells_topics = []
        for _, batch in enumerate(loader):
            batch = batch.to(self.device)
            theta, _ = self.call_theta(batch)
            cells_topics.append(theta)
        
        return torch.cat(cells_topics).detach().cpu().numpy()


    @torch.no_grad()
    def call_G2T(self, gene_sets):
        """
        Call the assignment matrix with shape `genes x topics`

        Parameters
        ----------
        gene_sets: torch.Tensor
            Query genes with shape `num_genes x gene_dim`.
        """

        gene_sets = gene_sets.to(self.device)
        _, scores, _ = self.moe(gene_sets)
        
        return scores.detach().cpu().numpy()
        


