"""
Graph autoencoder module (ported from DiffGAD). Exposes GraphAE (historically also referenced as Graph_AE).
"""
import math
import torch
import torch.nn as nn
from torch_geometric.nn import GCN
from pygod.nn.decoder import DotProductDecoder
from pygod.nn.functional import double_recon_loss


class GraphAE(nn.Module):
    """
    Graph AutoEncoder.

    Learns a low-dimensional node embedding by jointly reconstructing node attributes and
    graph structure (via a dot-product structure decoder).
    """

    def __init__(
        self,
        in_dim,
        hid_dim=64,
        num_layers=4,
        dropout=0.0,
        act=torch.nn.functional.relu,
        sigmoid_s=False,
        backbone=GCN,
        **kwargs,
    ):
        super().__init__()

        # Split layers between encoder and decoder stacks
        assert num_layers >= 2, "Number of layers must be greater than or equal to 2."
        encoder_layers = math.floor(num_layers / 2)
        decoder_layers = math.ceil(num_layers / 2)

        # Shared encoder: node features + edges -> embedding
        self.shared_encoder = backbone(
            in_channels=in_dim,
            hidden_channels=hid_dim,
            num_layers=encoder_layers,
            out_channels=hid_dim,
            dropout=dropout,
            act=act,
            **kwargs,
        )

        # Attribute decoder: embedding -> reconstructed x
        self.attr_decoder = backbone(
            in_channels=hid_dim,
            hidden_channels=hid_dim,
            num_layers=decoder_layers,
            out_channels=in_dim,
            dropout=dropout,
            act=act,
            **kwargs,
        )

        # Structure decoder: embedding -> adjacency logits
        self.struct_decoder = DotProductDecoder(
            in_dim=hid_dim,
            hid_dim=hid_dim,
            num_layers=decoder_layers - 1,
            dropout=dropout,
            act=act,
            sigmoid_s=sigmoid_s,
            backbone=backbone,
            **kwargs,
        )

        self.loss_func = double_recon_loss
        self.emb = None  # last encoded embedding

    def forward(self, x, edge_index):
        """
        Args:
            x: node features [num_nodes, in_dim]
            edge_index: edges [2, num_edges]

        Returns:
            x_: reconstructed features
            s_: reconstructed adjacency / structure logits
            self.emb: node embeddings
        """
        self.emb = self.encode(x, edge_index)
        x_, s_ = self.decode(self.emb, edge_index)
        return x_, s_, self.emb

    def encode(self, x, edge_index):
        """Encode nodes to embeddings [num_nodes, hid_dim]."""
        self.emb = self.shared_encoder(x, edge_index)
        return self.emb

    def decode(self, emb, edge_index):
        """Decode embeddings to node features and structure."""
        x_ = self.attr_decoder(emb, edge_index)
        s_ = self.struct_decoder(emb, edge_index)
        return x_, s_
