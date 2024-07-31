import torch
import torch.nn as nn

from ..utils import get_activation, get_device


class TopicModel(nn.Module):
    def __init__(self, embeddings: torch.Tensor, num_topics: int,
                 act: str = 'relu', enc_drop: float = 0.5, device: str = 'cuda:0'):
        """
        Topic Model in Gene Embedding Space.

        Parameters
        ----------
        embeddings: torch.Tensor
            Pretrained gene embeddings with shape `num_genes` x `embedding_size`.
        num_topics: int
            Number of topics.
        act: str
            Activity function used in neural networks.
        enc_drop: float
            Dropout rate used in neural networks.
        device: str
            Device for training model.
        """
        super().__init__()

        num_genes, embedding_size = embeddings.size()
        self.num_genes = num_genes
        self.embedding_size = embedding_size
        self.act = get_activation(act)
        self.enc_drop = enc_drop
        self.dropout = nn.Dropout(enc_drop)
        self.device = get_device(device)

        self.rho = embeddings.clone().float().to(self.device)

        # Model
        self.alphas = nn.Linear(num_genes, num_topics, bias=False)
        self.q = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            self.act,
            nn.Linear(embedding_size, embedding_size),
            self.act,          
        )
        self.mu = nn.Linear(embedding_size, num_topics, bias=True)
        self.logsigma = nn.Linear(embedding_size, num_topics, bias=True)

    def reparameterize(self, mu, logvar):
        """
        Draw a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * logvar) 
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def encode(self, gene_counts: torch.Tensor):
        """
        Get paramters of the variational distribution.

        Parameters
        ----------
        gene_counts: torch.Tensor
                     The read counts matrix with shape `num_cells` x `num_genes`.
        """
        q = self.q(gene_counts)

        if self.enc_drop > 0:
            q = self.dropout(q)

        mu = self.mu(q)
        logsigma = self.logsigma(q)
        kl = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp(), dim=-1).mean()
        return mu, logsigma, kl
