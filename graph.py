import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

class NeighborFinder:
    def __init__(self, adj_list, n_user, n_item, uniform=False, seed=None, device='cpu'):
        self.node_to_neighbors = []
        self.node_to_edge_idxs = []
        self.node_to_edge_timestamps = []
        self.device = device

        adj_list_new = [[] for _ in range(n_user + n_item + 1)] # TODO: Not us +1
        for u in adj_list:
            adj_list_new[u] = [x for x in adj_list[u]]

        for neighbors in adj_list_new:
            # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
            # We sort the list based on timestamp
            sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
            self.node_to_neighbors.append(
                torch.tensor([x[0] for x in sorted_neighhbors]).to(self.device))
            self.node_to_edge_idxs.append(
                torch.tensor([x[1] for x in sorted_neighhbors]).to(self.device))
            self.node_to_edge_timestamps.append(
                torch.tensor([x[2] for x in sorted_neighhbors]).to(self.device))

        self.uniform = uniform

        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)

    def find_before(self, src_idx, cut_time):
        """
        Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

        Returns 3 lists: neighbors, edge_idxs, timestamps

        """
        i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

        return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]

    def find_before_batch(self, batch_src_idx, batch_cut_time):
        """
        Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

        Returns 3 lists: neighbors, edge_idxs, timestamps

        """
        batch_time_stamps = pad_sequence([self.node_to_edge_timestamps[src_idx] for src_idx in batch_src_idx], batch_first=True, padding_value=9999999999)
        batch_cut_time = batch_cut_time.reshape(-1, 1)
        batch_cut_idx = torch.searchsorted(batch_time_stamps, batch_cut_time)

        return batch_cut_idx

    def get_temporal_neighbor(self, batch_source_node, batch_timestamp, n_neighbors=20):
        """
        Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        num_neighbors: int
        """
        assert (len(batch_source_node) == len(batch_timestamp))

        tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
        # NB! All interactions described in these matrices are sorted in each row by time
        out_neighbors = torch.zeros((len(batch_source_node), tmp_n_neighbors)).to(torch.int32).to(self.device) # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
        out_timestamps = torch.zeros((len(batch_source_node), tmp_n_neighbors)).to(torch.float32).to(self.device) # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
        baout_edges = torch.zeros((len(batch_source_node), tmp_n_neighbors)).to(torch.int32).to(self.device) # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

        batch_cut_idx = self.find_before_batch(batch_source_node, batch_timestamp)
        for i, (source_node, cut_idx) in enumerate(zip(batch_source_node, batch_cut_idx)):
            cut_idx = int(cut_idx.item())
            source_neighbors = self.node_to_neighbors[source_node][:cut_idx]
            source_edge_idxs = self.node_to_edge_idxs[source_node][:cut_idx]
            source_edge_times = self.node_to_edge_timestamps[source_node][:cut_idx]

            if len(source_neighbors) > 0 and n_neighbors > 0:
                if self.uniform: # if we are applying uniform sampling, shuffles the data above before sampling
                    sampled_idx = torch.randint(0, len(source_neighbors), (n_neighbors,))

                    out_neighbors[i, :] = source_neighbors[sampled_idx]
                    out_timestamps[i, :] = source_edge_times[sampled_idx]
                    baout_edges[i, :] = source_edge_idxs[sampled_idx]

                    # re-sort based on time
                    pos = out_timestamps[i, :].argsort()
                    out_neighbors[i, :] = out_neighbors[i, :][pos]
                    out_timestamps[i, :] = out_timestamps[i, :][pos]
                    baout_edges[i, :] = baout_edges[i, :][pos]
                else:
                    # Take most recent interactions
                    source_edge_times = source_edge_times[-n_neighbors:]
                    source_neighbors = source_neighbors[-n_neighbors:]
                    source_edge_idxs = source_edge_idxs[-n_neighbors:]

                    assert (len(source_neighbors) <= n_neighbors)
                    assert (len(source_edge_times) <= n_neighbors)
                    assert (len(source_edge_idxs) <= n_neighbors)

                    out_neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
                    out_timestamps[i, n_neighbors - len(source_edge_times):] = source_edge_times
                    baout_edges[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs

        return (out_neighbors, baout_edges, out_timestamps)
