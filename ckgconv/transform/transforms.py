import logging
import time

import torch
from torch_geometric.utils import subgraph
from tqdm import tqdm
import torch.multiprocessing as mp
from functools import partial
# from multiprocessing import Pool
# from torch.multiprocessing import Pool
from torch_geometric.data import Batch
import torch_geometric as pyg

def pre_transform_in_memory(dataset, transform_func, show_progress=False, cfg=dict(), posenc_mode=False):
    """Pre-transform already loaded PyG dataset object.

    Apply transform function to a loaded PyG dataset object so that
    the transformed result is persistent for the lifespan of the object.
    This means the result is not saved to disk, as what PyG's `pre_transform`
    would do, but also the transform is applied only once and not at each
    data access as what PyG's `transform` hook does.

    Implementation is based on torch_geometric.data.in_memory_dataset.copy

    Args:
        dataset: PyG dataset object to modify
        transform_func: transformation function to apply to each data example
        show_progress: show tqdm progress bar
    """
    if transform_func is None:
        return dataset

    posenc_cfg = cfg.get("compute_posenc", dict())
    if not posenc_mode:
        posenc_cfg["multiprocess"] = False

    if posenc_cfg.get("multiprocess", False):
        torch.multiprocessing.set_sharing_strategy('file_system')
        # notice: might be dangerous, please read torch.multiprocessing before using
        st = time.time()
        num_processes = min(mp.cpu_count(), posenc_cfg.get('num_processes'))
        progress_bar = tqdm(total=len(dataset))
        with mp.Pool(num_processes) as p:
            futures = tqdm(p.imap(transform_func, iter(dataset)), total=len(dataset),
                           disable=not show_progress,
                           mininterval=10,
                           miniters=len(dataset)//20
                           )
            data_list = tuple(futures)
    elif posenc_cfg.get("batch_process", False):
        data_list = []
        sub_list = []
        def batch_transform(sub_list):
            batch = Batch.from_data_list(sub_list)
            batch = transform_func(batch)
            abs_pe = dict()
            rel_pe = dict()
            for pe_name in posenc_cfg.get('abs_pe_ls', []):
                if pe_name not in batch:
                    continue
                pe_ls = pyg.utils.unbatch(batch[pe_name], batch.batch, dim=0)
                abs_pe[pe_name] = pe_ls
            for pe_name in posenc_cfg.get('rel_pe_ls', []):
                idx_name = pe_name + "_index"
                val_name = pe_name + "_val"
                if idx_name not in batch:
                    continue
                idx_ls = pyg.utils.unbatch_edge_index(batch[idx_name], batch.batch)
                rel_pe[idx_name] = idx_ls
                if val_name not in batch:
                    continue
                edge_batch = [torch.ones(idx_ls[i].size(1), dtype=torch.long) * i for i in range(len(sub_list))]
                edge_batch = torch.concat(edge_batch, dim=0)
                val_ls = pyg.utils.unbatch(batch[val_name], edge_batch)
                rel_pe[val_name] = val_ls

            for i in range(len(sub_list)):
                for k in abs_pe:
                    sub_list[i][k] = abs_pe[k][i]
                for k in rel_pe:
                    sub_list[i][k] = rel_pe[k][i]
            return sub_list

        for i in tqdm(range(len(dataset)),
                      disable=not show_progress,
                      mininterval=10,
                      miniters=len(dataset)//20):
            sub_list.append(dataset.get(i))
            if len(sub_list) == posenc_cfg.get("batch_size", 16):
                sub_list = batch_transform(sub_list)
                data_list = data_list + sub_list
                sub_list = []

        if len(sub_list) > 0:
            sub_list = batch_transform(sub_list)
            data_list = data_list + sub_list
            sub_list = []

        print("done")
    else:
        data_list = [transform_func(dataset.get(i))
                     for i in tqdm(range(len(dataset)),
                                   disable=not show_progress,
                                   mininterval=10,
                                   miniters=len(dataset)//20)]


    data_list = list(filter(None, data_list))

    dataset._indices = None
    dataset.data_list = data_list
    dataset.data, dataset.slices = dataset.collate(data_list)


def typecast_x(data, type_str):
    if type_str == 'float':
        data.x = data.x.float()
    elif type_str == 'long':
        data.x = data.x.long()
    else:
        raise ValueError(f"Unexpected type '{type_str}'.")
    return data


def concat_x_and_pos(data):
    data.x = torch.cat((data.x, data.pos), 1)
    return data


def clip_graphs_to_size(data, size_limit=5000):
    if hasattr(data, 'num_nodes'):
        N = data.num_nodes  # Explicitly given number of nodes, e.g. ogbg-ppa
    else:
        N = data.x.shape[0]  # Number of nodes, including disconnected nodes.
    if N <= size_limit:
        return data
    else:
        logging.info(f'  ...clip to {size_limit} a graph of size: {N}')
        if hasattr(data, 'edge_attr'):
            edge_attr = data.edge_attr
        else:
            edge_attr = None
        edge_index, edge_attr = subgraph(list(range(size_limit)),
                                         data.edge_index, edge_attr)
        if hasattr(data, 'x'):
            data.x = data.x[:size_limit]
            data.num_nodes = size_limit
        else:
            data.num_nodes = size_limit
        if hasattr(data, 'node_is_attributed'):  # for ogbg-code2 dataset
            data.node_is_attributed = data.node_is_attributed[:size_limit]
            data.node_dfs_order = data.node_dfs_order[:size_limit]
            data.node_depth = data.node_depth[:size_limit]
        data.edge_index = edge_index
        if hasattr(data, 'edge_attr'):
            data.edge_attr = edge_attr
        return data
