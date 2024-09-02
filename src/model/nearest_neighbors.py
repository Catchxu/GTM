import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load embeddings
topic_embeddings = torch.load('./topic embeddings/solution1').to(device)
