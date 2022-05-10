import numpy as np

from gdl.data.base import BaseDataset
from gdl.data.utils import get_adj_matrix, get_clipped_matrix, get_top_k_matrix


def get_ppr_matrix(adj_matrix: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    num_nodes = adj_matrix.shape[0]
    A_tilde = adj_matrix + np.eye(num_nodes)
    D_tilde = np.diag(1 / np.sqrt(A_tilde.sum(axis=1)))
    H = D_tilde @ A_tilde @ D_tilde
    return alpha * np.linalg.inv(np.eye(num_nodes) - (1 - alpha) * H)


class PPRDataset(BaseDataset):
    """
    Dataset preprocessed with GDC using PPR diffusion.
    Note that this implementations is not scalable
    since we directly invert the adjacency matrix.
    """

    def __init__(
        self,
        name: str = "Cora",
        use_lcc: bool = True,
        alpha: float = 0.1,
        k: int = 16,
        eps: float = None,
        undirected: bool = False,
        data_dir: str = None,
    ):
        self.name = name
        self.use_lcc = use_lcc
        self.alpha = alpha
        self.k = k
        self.eps = eps
        self.undirected = undirected
        super(PPRDataset, self).init(data_dir)

    def process(self):
        base = self.get_dataset()
        # generate adjacency matrix from sparse representation
        adj_matrix = get_adj_matrix(base)
        # obtain exact PPR matrix
        ppr_matrix = get_ppr_matrix(adj_matrix, alpha=self.alpha)

        if self.k:
            print(f"Selecting top {self.k} edges per node.")
            ppr_matrix = get_top_k_matrix(ppr_matrix, k=self.k)
        elif self.eps:
            print(f"Selecting edges with weight greater than {self.eps}.")
            ppr_matrix = get_clipped_matrix(ppr_matrix, eps=self.eps)
        else:
            raise ValueError

        if self.undirected:
            # Undirected modification suggested in DIGL paper
            ppr_matrix = (ppr_matrix + ppr_matrix.T) / 2

        # create PyG Data object
        edges_i = []
        edges_j = []
        edge_attr = []
        for i, row in enumerate(ppr_matrix):
            for j in np.where(row > 0)[0]:
                edges_i.append(i)
                edges_j.append(j)
                edge_attr.append(ppr_matrix[i, j])
        edge_index = [edges_i, edges_j]

        self.to_dataset(base, edge_index, edge_attr)

    def __str__(self) -> str:
        return (
            f"{self.name}_ppr_alpha={self.alpha}_k={self.k}_eps={self.eps}_lcc={self.use_lcc}"
            + ("_undirected" if self.undirected else "")
        )
