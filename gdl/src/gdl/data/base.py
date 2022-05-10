import numpy as np
import os
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import (
    Planetoid,
    Amazon,
    Coauthor,
    WebKB,
    WikipediaNetwork,
    Actor,
)

from gdl.data.utils import get_undirected_adj_matrix

DEFAULT_DATA_PATH = "~/data"


def get_node_mapper(lcc: np.ndarray) -> dict:
    mapper = {}
    counter = 0
    for node in lcc:
        mapper[node] = counter
        counter += 1
    return mapper


def remap_edges(edges: list, mapper: dict) -> list:
    row = [e[0] for e in edges]
    col = [e[1] for e in edges]
    row = list(map(lambda x: mapper[x], row))
    col = list(map(lambda x: mapper[x], col))
    return [row, col]


def get_component(dataset: InMemoryDataset, start: int = 0) -> set:
    visited_nodes = set()
    queued_nodes = set([start])
    row, col = dataset.data.edge_index.numpy()
    while queued_nodes:
        current_node = queued_nodes.pop()
        visited_nodes.update([current_node])
        neighbors = col[np.where(row == current_node)[0]]
        neighbors = [
            n for n in neighbors if n not in visited_nodes and n not in queued_nodes
        ]
        queued_nodes.update(neighbors)
    return visited_nodes


def get_largest_connected_component(dataset: InMemoryDataset) -> np.ndarray:
    remaining_nodes = set(range(dataset.data.x.shape[0]))
    comps = []
    while remaining_nodes:
        start = min(remaining_nodes)
        comp = get_component(dataset, start)
        comps.append(comp)
        remaining_nodes = remaining_nodes.difference(comp)
    return np.array(list(comps[np.argmax(list(map(len, comps)))]))


def get_dataset(
    name: str, use_lcc: bool = True, data_dir=DEFAULT_DATA_PATH
) -> InMemoryDataset:
    path = os.path.join(data_dir, name)
    if name in ["Cora", "Citeseer", "Pubmed"]:
        dataset = Planetoid(path, name)
    elif name in ["Computers", "Photo"]:
        dataset = Amazon(path, name)
    elif name == "CoauthorCS":
        dataset = Coauthor(path, "CS")
    elif name in ["Cornell", "Texas", "Wisconsin"]:
        dataset = WebKB(path, name)
    elif name in ["Chameleon", "Squirrel"]:
        dataset = WikipediaNetwork(path, name, geom_gcn_preprocess=True)
    elif name == "Actor":
        dataset = Actor(path, "Actor")
    else:
        raise Exception(f"Unknown dataset: {name}")

    if use_lcc:
        lcc = get_largest_connected_component(dataset)

        x_new = dataset.data.x[lcc]
        y_new = dataset.data.y[lcc]

        row, col = dataset.data.edge_index.numpy()
        edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
        edges = remap_edges(edges, get_node_mapper(lcc))

        data = Data(
            x=x_new,
            edge_index=torch.LongTensor(edges),
            y=y_new,
            train_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            test_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            val_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
        )
        dataset.data = data

    mapping = dict(
        zip(np.unique(dataset.data.y), range(len(np.unique(dataset.data.y))))
    )
    dataset.data.y = torch.LongTensor([mapping[u] for u in np.array(dataset.data.y)])

    return dataset


class BaseDataset(InMemoryDataset):
    """
    Data preprocessed by being made undirected.
    """

    def __init__(
        self,
        name: str = "Cora",
        use_lcc: bool = True,
        undirected: bool = False,
        data_dir: str = None,
    ):
        self.name = name
        self.use_lcc = use_lcc
        self.undirected = undirected
        self.init(data_dir)

    def init(self, data_dir):
        if data_dir is None:
            data_dir = DEFAULT_DATA_PATH
        self.data_dir = data_dir

        super(BaseDataset, self).__init__(self.data_dir)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> list:
        return []

    @property
    def processed_file_names(self) -> list:
        return [str(self) + ".pt"]

    def download(self):
        pass

    def get_dataset(self):
        return get_dataset(name=self.name, use_lcc=self.use_lcc, data_dir=self.data_dir)

    def to_dataset(self, base, edge_index, edge_attr):
        data = Data(
            x=base.data.x,
            edge_index=torch.LongTensor(edge_index),
            edge_attr=torch.FloatTensor(edge_attr) if edge_attr is not None else None,
            y=base.data.y,
            train_mask=torch.zeros(base.data.train_mask.size()[0], dtype=torch.bool),
            test_mask=torch.zeros(base.data.test_mask.size()[0], dtype=torch.bool),
            val_mask=torch.zeros(base.data.val_mask.size()[0], dtype=torch.bool),
        )

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    def process(self):
        """
        Overwrites of this method must start with self.get_dataset() and
        end with self.to_dataset(base, edge_index, edge_attr).
        """
        base = self.get_dataset()

        if not self.undirected:
            adj_matrix = get_undirected_adj_matrix(base)
            edges_i = []
            edges_j = []
            edge_attr = []
            for i, row in enumerate(adj_matrix):
                for j in np.where(row > 0)[0]:
                    edges_i.append(i)
                    edges_j.append(j)
                    edge_attr.append(adj_matrix[i, j])
            edge_index = [edges_i, edges_j]
        else:
            edge_index = base.data.edge_index
            edge_attr = (
                base.data.edge_attr if "edge_attr" in base.data.__dir__() else None
            )

        self.to_dataset(base, edge_index, edge_attr)

    def __str__(self) -> str:
        return f"{self.name}_{'base' if not self.undirected else 'undirected'}_lcc={self.use_lcc}"
