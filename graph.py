import numpy as np
import torch

class NeighborFinder:
    def __init__(self, adj_list, n_user, n_item, uniform=False, seed=None, use_mem=False):
        self.node_to_neighbors = []
        self.node_to_edge_idxs = []
        self.node_to_edge_timestamps = []

        adj_list_new = [[] for _ in range(n_user + n_item + 1)] # TODO: Not us +1
        for u in adj_list:
            assert u != 0
            adj_list_new[u] = [x for x in adj_list[u]]

        for neighbors in adj_list_new:
            # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
            # We sort the list based on timestamp
            sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
            self.node_to_neighbors.append(
                np.array([x[0] for x in sorted_neighhbors]))
            self.node_to_edge_idxs.append(
                np.array([x[1] for x in sorted_neighhbors]))
            self.node_to_edge_timestamps.append(
                np.array([x[2] for x in sorted_neighhbors]))

        self.uniform = uniform

        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)

        self.ngh_memory = {}
        self.use_mem = use_mem

    def find_before(self, src_idx, cut_time):
        # if src_idx == 0:
        #     return np.array([]), np.array([]), np.array([])
        """
        Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

        Returns 3 lists: neighbors, edge_idxs, timestamps

        """
        i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

        return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]

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
        out_neighbors = np.zeros((len(batch_source_node), tmp_n_neighbors)).astype(np.int32) # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
        out_timestamps = np.zeros((len(batch_source_node), tmp_n_neighbors)).astype(np.float32) # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
        baout_edges = np.zeros((len(batch_source_node), tmp_n_neighbors)).astype(np.int32) # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

        for i, (source_node, timestamp) in enumerate(zip(batch_source_node, batch_timestamp)):
            # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time
            if self.use_mem:
                if (source_node, timestamp) in self.ngh_memory:
                    source_neighbors, source_edge_idxs, source_edge_times = self.ngh_memory[source_node, timestamp]
                else:
                    source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node, timestamp)
                    self.ngh_memory[source_node, timestamp] = (source_neighbors, source_edge_idxs, source_edge_times)
            else:
                source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node, timestamp)

            if len(source_neighbors) > 0 and n_neighbors > 0:
                if self.uniform: # if we are applying uniform sampling, shuffles the data above before sampling
                    sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)
                    # if n_neighbors <= len(source_neighbors):
                    #     sampled_idx = np.random.choice(len(source_neighbors), n_neighbors)
                    # else:
                    #     sampled_idx_1 = np.arange(len(source_neighbors))
                    #     sampled_idx_2 = np.random.randint(0, len(source_neighbors), n_neighbors - len(source_neighbors))
                    #     sampled_idx = np.concatenate((sampled_idx_1, sampled_idx_2))

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

                    # assert (len(source_neighbors) <= n_neighbors)
                    # assert (len(source_edge_times) <= n_neighbors)
                    # assert (len(source_edge_idxs) <= n_neighbors)

                    out_neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
                    out_timestamps[i, n_neighbors - len(source_edge_times):] = source_edge_times
                    baout_edges[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs

        return (out_neighbors, baout_edges, out_timestamps)
