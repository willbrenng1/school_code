import numpy as np
from os.path import abspath, exists
from scipy import sparse, stats
from sklearn.cluster import KMeans


class PoliticalBlogsClustering:
    def __init__(self):
        pass

    def find_majority_labels(self, num_clusters=2):
        """
        This method loads the data, performs spectral clustering  and reports the majority labels

        Inputs:
            num_clusters (int): The number of clusters to be created

        Output:
            A map with following attributes
            1. overall_mismatch_rate: <2 decimal places>
            2. mismatch_rates: [{"majority_index": <int>, "mismatch_rate": <2 decimal places>}]
        """

        map = {"overall_mismatch_rate": None, "mismatch_rates": []}

        # TODO - start your implementation.
        ## It is suggested to break your code into smaller methods but not nessasary.

        def import_nodes_and_edges(node_path, edge_path):

            nodes = []
            with open(node_path) as f:
                for line in f:
                    n = line.split("\t")
                    n[0] = int(n[0])
                    n[1] = n[1].strip('"')
                    n[2] = int(n[2])
                    n[3] = n[3].strip('"')
                    nodes.append(n)
            edges = np.loadtxt(edge_path, dtype=int)

            return nodes, edges

        def clean_nodes_and_edges(nodes, edges):
            node_assignment_dict = {e[0]: e[2] for e in nodes}
            edges_no_self_loop = edges[edges[:, 0] != edges[:, 1]]
            edges_de_dupe = np.unique(np.sort(edges_no_self_loop, axis=1), axis=0)

            nodes_used = np.unique(edges_de_dupe)

            new_node_map = {old: new for new, old in enumerate(nodes_used)}
            new_node_map_inv = {v: k for k, v in new_node_map.items()}

            return (
                node_assignment_dict,
                edges_de_dupe,
                nodes_used,
                new_node_map,
                new_node_map_inv,
            )

        def lap_matrix_construction(node_map, edges, nodes_used):

            i = np.array([node_map[n1] for n1 in edges[:, 0]])
            j = np.array([node_map[n2] for n2 in edges[:, 1]])
            v = np.ones(len(i))

            A = sparse.coo_matrix((v, (i, j)), shape=(len(nodes_used), len(nodes_used)))
            A = A + A.T
            A = sparse.csc_matrix.todense(A)

            D = np.diag(np.sum(A, axis=1).A1)

            L = D - A

            return L

        def construct_Z_matrix(L):

            v, x = np.linalg.eig(L)

            idx_sorted = np.argsort(v)
            Z = x[:, idx_sorted[:num_clusters]]
            row_vec_norm = np.linalg.norm(
                Z, axis=1, keepdims=True
            )  # https://stackoverflow.com/questions/68227583/np-linalg-norm-axiserror-axis-1-is-out-of-bounds-for-array-of-dimension-1

            Z_normalized = Z / row_vec_norm
            Z_normalized = np.array(Z_normalized)

            return Z_normalized

        def k_means_and_accuracy(Z, node_assignment_dict, new_node_map_inv, nodes_used):
            kmeans = KMeans(n_clusters=num_clusters).fit(Z.real)

            labels = kmeans.labels_

            label_assignment_dict = {
                ks + 1: np.where(labels == ks)[0] for ks in range(num_clusters)
            }

            cluster_lable_dict = {}
            for k, v in label_assignment_dict.items():
                k_list = []
                for e in v:
                    k_list.append(node_assignment_dict[new_node_map_inv[e]])

                cluster_lable_dict[k] = k_list

            cluster_lable_and_mismatch_dict = []
            running_mismath_total = 0

            for k, v in cluster_lable_dict.items():
                cluster_lable_and_mismatch_dict.append(
                    {
                        "majority_index": k,
                        "mismatch_rate": round(
                            (
                                len(cluster_lable_dict[k])
                                - stats.mode(cluster_lable_dict[k])[1]
                            )
                            / len(cluster_lable_dict[k]),
                            2,
                        ),
                    }
                )
                running_mismath_total += (
                    len(cluster_lable_dict[k]) - stats.mode(cluster_lable_dict[k])[1]
                )

            round(running_mismath_total / len(nodes_used), 2)

            return cluster_lable_and_mismatch_dict, round(
                running_mismath_total / len(nodes_used), 2
            )

        nodes, edges = import_nodes_and_edges("data/nodes.txt", "data/edges.txt")

        (
            node_assignment_dict,
            edges_de_dupe,
            nodes_used,
            new_node_map,
            new_node_map_inv,
        ) = clean_nodes_and_edges(nodes, edges)

        L = lap_matrix_construction(new_node_map, edges_de_dupe, nodes_used)

        Z = construct_Z_matrix(L)

        cluster_assignments, mismatch_rate = k_means_and_accuracy(
            Z, node_assignment_dict, new_node_map_inv, nodes_used
        )

        map["overall_mismatch_rate"] = mismatch_rate
        map["mismatch_rates"] = cluster_assignments

        return map
