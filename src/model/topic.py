import torch
import torch.nn as nn
import torch.nn.functional as F 

from ..utils import get_activation, get_device


class TopicModel(nn.Module):
    def __init__(self, embeddings: torch.Tensor, num_topics: int,
                 gene_size: int, act: str = 'relu', enc_drop: float = 0.5, 
                 device: str = 'cuda:0'):
        """
        Topic Model in Gene Embedding Space.

        Parameters
        ----------
        embeddings: torch.Tensor
            Pretrained gene embeddings with shape `num_genes` x `embedding_size`.
        num_topics: int
            Number of topics.
        gene_size: int
            Dimension of represented gene counts.
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
        self.num_topics = num_topics
        self.gene_size = gene_size

        self.act = get_activation(act)
        self.enc_drop = enc_drop
        self.dropout = nn.Dropout(enc_drop)
        self.device = get_device(device)

        self.rho = embeddings.clone().float().to(self.device)

        # Model
        self.alpha = nn.Linear(embedding_size, num_topics, bias=False)  # topic embedding
        self.q = nn.Sequential(
            nn.Linear(num_genes, gene_size),
            self.act,
            nn.Linear(gene_size, gene_size),
            self.act,          
        )
        self.mu = nn.Linear(gene_size, num_topics, bias=True)
        self.logsigma = nn.Linear(gene_size, num_topics, bias=True)

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

    def encode(self, gene_counts: torch.Tensor):
        """
        Get paramters of the variational distribution.

        Parameters
        ----------
        gene_counts: torch.Tensor
                     The read counts matrix with shape `batch_size` x `num_genes`.
        """
        q = self.q(gene_counts)  # batch_size x num_genes -> batch_size x gene_size

        if self.enc_drop > 0:
            q = self.dropout(q)

        mu = self.mu(q)  # batch_size x gene_size -> batch_size x num_topics
        logsigma = self.logsigma(q)
        kl = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp(), dim=-1).mean()
        return mu, logsigma, kl

    def get_beta(self):
        logit = self.alpha(self.rho)  # num_genes x embedding_size -> num_genes x num_topics
        beta = F.softmax(logit, dim=0).transpose(1, 0)
        return beta

    def get_theta(self, gene_counts):
        """
        Get the topic poportion for the document passed in the normalixe bow or TF-IDF.
        """
        mu_theta, logsigma_theta, kl_theta = self.encode(gene_counts)
        delta = self.reparameterize(mu_theta, logsigma_theta)
        theta = F.softmax(delta, dim=-1)
        return theta, kl_theta

    def decode(self, theta, beta):
        """
        Compute the probability of topic given the read counts matrix.
        """
        res = torch.mm(theta, beta)
        almost_zeros = torch.full_like(res, 1e-6)
        results_without_zeros = res.add(almost_zeros)
        prob = torch.log(results_without_zeros)
        return prob

    def forward(self, gene_counts, aggregate=True):
        theta, kl_theta = self.get_theta(gene_counts)
        beta = self.get_beta()

        preds = self.decode(theta, beta)
        recon_loss = - (preds * gene_counts).sum(1)
        if aggregate:
            recon_loss = recon_loss.mean()
        return recon_loss, kl_theta