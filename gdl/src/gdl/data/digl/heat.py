import numpy as np
from scipy.linalg import expm

from gdl.data.base import BaseDataset
from gdl.data.utils import get_adj_matrix, get_clipped_matrix, get_top_k_matrix


def get_heat_matrix(adj_matrix: np.ndarray, t: float = 5.0) -> np.ndarray:
    num_nodes = adj_matrix.shape[0]
    A_tilde = adj_matrix + np.eye(num_nodes)
    D_tilde = np.diag(1 / np.sqrt(A_tilde.sum(axis=1)))
    H = D_tilde @ A_tilde @ D_tilde
    return expm(-t * (np.eye(num_nodes) - H))


class HeatDataset(BaseDataset):
    """
    Dataset preprocessed with GDC using heat kernel diffusion.
    Note that this implementations is not scalable
    since we directly calculate the matrix exponential
    of the adjacency matrix.
    """

    def __init__(
        self,
        name: str = "Cora",
        use_lcc: bool = True,
        t: float = 5.0,
        k: int = 16,
        eps: float = None,
        undirected: bool = False,
        data_dir: str = None,
    ):
        self.name = name
        self.use_lcc = use_lcc
        self.t = t
        self.k = k
        self.eps = eps
        self.undirected = undirected
        super(HeatDataset, self).init(data_dir)

    def process(self):
        base = self.get_dataset()
        adj_matrix = get_adj_matrix(base)
        # get heat matrix as described in Berberidis et al., 2019
        heat_matrix = get_heat_matrix(adj_matrix, t=self.t)
        if self.k:
            print(f"Selecting top {self.k} edges per node.")
            heat_matrix = get_top_k_matrix(heat_matrix, k=self.k)
        elif self.eps:
            print(f"Selecting edges with weight greater than {self.eps}.")
            heat_matrix = get_clipped_matrix(heat_matrix, eps=self.eps)
        else:
            raise ValueError

        if self.undirected:
            # Undirected modification suggested in DIGL paper
            heat_matrix = (heat_matrix + heat_matrix.T) / 2

        # create PyG Data object
        edges_i = []
        edges_j = []
        edge_attr = []
        for i, row in enumerate(heat_matrix):
            for j in np.where(row > 0)[0]:
                edges_i.append(i)
                edges_j.append(j)
                edge_attr.append(heat_matrix[i, j])
        edge_index = [edges_i, edges_j]
        self.to_dataset(base, edge_index, edge_attr)

    def __str__(self) -> str:
        return (
            f"{self.name}_heat_t={self.t}_k={self.k}_eps={self.eps}_lcc={self.use_lcc}"
            + ("_undirected" if self.undirected else "")
        )
