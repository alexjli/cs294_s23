import dgl
import torch
from torch import nn

from se3_transformer.model import SE3Transformer
from se3_transformer.model.fiber import Fiber

# pylint: disable=no-member

class SE3Denoiser(nn.Module):
    """ Denoise coordinates using SE3 transformer """
    def __init__(self,
                 knn_k=20,
                 inv_cube_k=40,
                 num_layers=4,
                 l1_in_features=3,
                 num_degrees=4,
                 num_channels=32,
                 num_heads=4,
                 div=4,
                 l1_out_features=3):
        super().__init__()
        self.knn_k = knn_k
        self.inv_cube_k = inv_cube_k

        fiber_in = Fiber({
            0: 1,
            1: l1_in_features
        })
        fiber_hidden = Fiber.create(num_degrees, num_channels)
        fiber_out = Fiber({1: l1_out_features})


        self.se3 = SE3Transformer(num_layers=num_layers,
                                  fiber_in=fiber_in,
                                  fiber_hidden=fiber_hidden,
                                  fiber_out=fiber_out,
                                  num_heads=num_heads,
                                  channels_div=div,
                                  use_layer_norm=True)

    def _coords_to_vecs(self, coords):
        is_nan_node = coords.isnan().any(dim=-1).any(dim=-1)  # B x N

        N, CA, C, O = coords[..., 0:1, :], coords[..., 1:2, :], coords[..., 2:3, :], coords[..., 3:, :]
        N[is_nan_node] = 0
        CA[is_nan_node] = 0
        C[is_nan_node] = 0
        O[is_nan_node] = 0

        CA_N = N - CA
        CA_C = C - CA
        CA_O = O - CA
        return torch.cat([CA, CA_N, CA_C, CA_O], dim=-2)

    def _coords_to_graph(self, coords, inf=1e8):
        CA = coords[..., 1, :]  # B x N x 3
        is_nan_node = coords.isnan().any(dim=-1).any(dim=-1)  # B x N
        CA[is_nan_node] = 0
        dist_mask = is_nan_node.unsqueeze(-2) & is_nan_node.unsqueeze(-1)  # B x N x N

        num_batch, num_res, _ = CA.shape
        rel_pos_CA = CA.unsqueeze(-2) - CA.unsqueeze(-3)  # B x N x N x 3
        dist_CA = torch.linalg.vector_norm(rel_pos_CA, dim=-1)  # B x N x N
        dist_CA[dist_mask] = inf
        sorted_dist, sorted_edges = torch.sort(dist_CA, dim=-1)  # B x N x N
        knn_edges = sorted_edges[..., :self.knn_k]

        # remove knn edges
        remaining_dist = sorted_dist[..., self.knn_k:]  # B x N x (N - knn_k)
        remaining_edges = sorted_edges[..., self.knn_k:]  # B x N x (N - knn_k)

        ## inv cube
        uniform = torch.distributions.Uniform(0,1)
        dist_noise = uniform.sample(remaining_dist.shape).to(coords.device)  # B x N x (N - knn_k)

        logprobs = -3 * torch.log(remaining_dist)  # B x N x (N - knn_k)
        perturbed_logprobs = logprobs - torch.log(-torch.log(dist_noise))  # B x N x (N - knn_k)
        flat_probs = torch.exp(perturbed_logprobs).view([num_batch * num_res, -1])  # (B x N) x inv_cube_k
        flat_sampled_edges_relative_idx = torch.multinomial(flat_probs, num_samples=self.inv_cube_k)  # (B x N) x inv_cube_k
        sampled_edges_relative_idx = flat_sampled_edges_relative_idx.view([num_batch, num_res, -1])  # B x N x inv_cube_k
        sampled_edges = torch.gather(remaining_edges, -1, sampled_edges_relative_idx)  # B x N x inv_cube_k

        edge_sinks = torch.cat([knn_edges, sampled_edges], dim=-1)  # B x N x (knn_k + inv_cube_k)
        flat_edge_sinks = edge_sinks.view(num_batch, -1)  # B x N * (knn_k + inv_cube_k)
        flat_edge_sources = torch.arange(num_res).repeat_interleave(self.knn_k + self.inv_cube_k)  # N * (knn_k + inv_cube_k)
        flat_edge_sources = flat_edge_sources.unsqueeze(0).expand(num_batch, -1)  # B x N * (knn_k + inv_cube_k)
        rel_pos = torch.gather(rel_pos_CA, -2, edge_sinks.unsqueeze(-1).expand(list(edge_sinks.shape) + [3]))  # B x N * (knn_k + inv_cube_k) x 3

        flat_edge_sinks_list, flat_edge_sources_list, rel_pos_list = map(torch.unbind,
                                                                         (flat_edge_sinks, flat_edge_sources, rel_pos))
        graphs = []
        for src, tgt, rel_pos in zip(flat_edge_sources_list, flat_edge_sinks_list, rel_pos_list):
            src = src.to(coords.device)
            tgt = tgt.to(coords.device)
            g = dgl.graph((src, tgt))
            g.edata['rel_pos'] = rel_pos.to(coords.device).view([-1, 3]).detach()
            graphs.append(g)

        G = dgl.batch(graphs)

        return G, is_nan_node

    def forward(self, coords, ts):
        num_batch, num_nodes = coords.shape[:2]
        G, is_nan_node = self._coords_to_graph(coords)

        # ts = (ts - (-7)) / (13.5 - (-7))

        vecs = self._coords_to_vecs(coords)
        node_features = {
            '0': ts.repeat_interleave(vecs.shape[1], dim=0).unsqueeze(-1),
            '1': vecs.view(-1, 3, 3)
        }
        update_G = self.se3(G, node_features)
        update_coords = update_G['1']
        update_coords = update_coords.view([num_batch, num_nodes, 3, 3])
        return update_coords, is_nan_node
