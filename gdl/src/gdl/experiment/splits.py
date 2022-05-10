from math import ceil
import numpy as np
import random
import torch
from torch_geometric.data import Data

from gdl.seeds import development_seed


def set_train_val_test_split(
        seed: int,
        data: Data,
        development_frac: float = 0.5,
        num_per_class: int = 20) -> Data:
    rnd_state = np.random.RandomState(development_seed)
    num_nodes = data.y.shape[0]

    num_development = ceil(development_frac * num_nodes)

    development_idx = rnd_state.choice(num_nodes, num_development, replace=False)
    test_idx = [i for i in np.arange(num_nodes) if i not in development_idx]

    train_idx = []
    rnd_state = np.random.RandomState(seed)
    for c in range(data.y.max() + 1):
        class_idx = development_idx[np.where(data.y[development_idx].cpu() == c)[0]]
        train_idx.extend(rnd_state.choice(class_idx, min(num_per_class, ceil(len(class_idx) * 0.5)), replace=False))

    val_idx = [i for i in development_idx if i not in train_idx]

    def get_mask(idx):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[idx] = 1
        return mask

    data.train_mask = get_mask(train_idx)
    data.val_mask = get_mask(val_idx)
    data.test_mask = get_mask(test_idx)

    return data
    

def set_train_val_test_split_frac(seed: int, data: Data, val_frac: float, test_frac: float):
    num_nodes = data.y.shape[0]

    val_size = ceil(val_frac * num_nodes)
    test_size = ceil(test_frac * num_nodes)
    train_size = num_nodes - val_size - test_size

    nodes = list(range(num_nodes))

    # Take same test set every time using development seed for robustness
    random.seed(development_seed)
    random.shuffle(nodes)
    test_idx = sorted(nodes[:test_size])
    nodes = [x for x in nodes if x not in test_idx]

    # Take train / val split according to seed
    random.seed(seed)
    random.shuffle(nodes)
    train_idx = sorted(nodes[:train_size])
    val_idx = sorted(nodes[train_size:])
    
    assert len(train_idx) + len(val_idx) + len(test_idx) == num_nodes

    def get_mask(idx):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[idx] = 1
        return mask

    data.train_mask = get_mask(train_idx)
    data.val_mask = get_mask(val_idx)
    data.test_mask = get_mask(test_idx)

    return data


# def set_train_val_test_split_classic(seed: int, data: Data, val_frac: float, test_frac: float):
#     random.seed(seed)
#     num_nodes = data.y.shape[0]

#     val_size = ceil(val_frac * num_nodes)
#     test_size = ceil(test_frac * num_nodes)
#     train_size = num_nodes - val_size - test_size

#     nodes = list(range(num_nodes))
#     random.shuffle(nodes)
#     train_idx = sorted(nodes[:train_size])
#     val_idx = sorted(nodes[train_size:train_size+val_size])
#     test_idx = sorted(nodes[train_size+val_size:])

#     def get_mask(idx):
#         mask = torch.zeros(num_nodes, dtype=torch.bool)
#         mask[idx] = 1
#         return mask

#     data.train_mask = get_mask(train_idx)
#     data.val_mask = get_mask(val_idx)
#     data.test_mask = get_mask(test_idx)

#     return data
    

# def set_train_val_test_split_robust(seed: int, data: Data, val_frac: float, test_frac: float):
#     num_nodes = data.y.shape[0]

#     val_size = ceil(val_frac * num_nodes)
#     test_size = ceil(test_frac * num_nodes)
#     train_size = num_nodes - val_size - test_size

#     nodes = list(range(num_nodes))

#     # Take same test set every time using development seed for robustness
#     random.seed(development_seed)
#     random.shuffle(nodes)
#     test_idx = sorted(nodes[:test_size])
#     nodes = [x for x in nodes if x not in test_idx]

#     # Take train / val split according to seed
#     random.seed(seed)
#     random.shuffle(nodes)
#     train_idx = sorted(nodes[:train_size])
#     val_idx = sorted(nodes[train_size:])
    
#     assert len(train_idx) + len(val_idx) + len(test_idx) == num_nodes

#     def get_mask(idx):
#         mask = torch.zeros(num_nodes, dtype=torch.bool)
#         mask[idx] = 1
#         return mask

#     data.train_mask = get_mask(train_idx)
#     data.val_mask = get_mask(val_idx)
#     data.test_mask = get_mask(test_idx)

#     return data


# def set_train_val_test_split(
#     seed: int, data: Data, num_development: int = 1500, num_per_class: int = 20
# ) -> Data:
#     rnd_state = np.random.RandomState(development_seed)
#     num_nodes = data.y.shape[0]
#     development_idx = rnd_state.choice(num_nodes, num_development, replace=False)
#     test_idx = [i for i in np.arange(num_nodes) if i not in development_idx]

#     train_idx = []
#     rnd_state = np.random.RandomState(seed)
#     for c in range(data.y.max() + 1):
#         class_idx = development_idx[np.where(data.y[development_idx].cpu() == c)[0]]
#         train_idx.extend(
#             rnd_state.choice(
#                 class_idx, min(num_per_class, int(len(class_idx) * 0.7)), replace=False
#             )
#         )

#     val_idx = [i for i in development_idx if i not in train_idx]

#     def get_mask(idx):
#         mask = torch.zeros(num_nodes, dtype=torch.bool)
#         mask[idx] = 1
#         return mask

#     data.train_mask = get_mask(train_idx)
#     data.val_mask = get_mask(val_idx)
#     data.test_mask = get_mask(test_idx)

#     return data


# def set_train_val_test_split_webkb(
#     seed: int,
#     data: Data,
#     num_development: int = 1500,
#     num_per_class: int = 20,
#     train_proportion: float = None,
# ) -> Data:
#     rnd_state = np.random.RandomState(development_seed)
#     num_nodes = data.y.shape[0]
#     development_idx = rnd_state.choice(num_nodes, num_development, replace=False)
#     test_idx = [i for i in np.arange(num_nodes) if i not in development_idx]

#     rnd_state = np.random.RandomState(seed)
#     if train_proportion:
#         train_idx = rnd_state.choice(
#             development_idx, int(train_proportion * len(development_idx)), replace=False
#         )
#     else:
#         train_idx = []
#         for c in range(data.y.max() + 1):
#             class_idx = development_idx[np.where(data.y[development_idx].cpu() == c)[0]]
#             train_idx.extend(rnd_state.choice(class_idx, num_per_class, replace=False))

#     val_idx = [i for i in development_idx if i not in train_idx]

#     def get_mask(idx):
#         mask = torch.zeros(num_nodes, dtype=torch.bool)
#         mask[idx] = 1
#         return mask

#     data.train_mask = get_mask(train_idx)
#     data.val_mask = get_mask(val_idx)
#     data.test_mask = get_mask(test_idx)

#     return data
