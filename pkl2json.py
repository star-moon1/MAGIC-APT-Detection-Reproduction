import pickle
import networkx as nx
import json
import dgl
import os
import torch
import numpy as np

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class StreamspotDataset(dgl.data.DGLDataset):
    def process(self):
        pass
    def __init__(self, name):
        super().__init__(name=name)

def clean_nx_graph(nx_graph):
    # 清洗节点属性
    for n, attr in nx_graph.nodes.items():
        for k, v in attr.items():
            if isinstance(v, np.integer):
                attr[k] = int(v)
            elif isinstance(v, np.floating):
                attr[k] = float(v)
            elif isinstance(v, np.ndarray):
                attr[k] = v.tolist()
            elif isinstance(v, torch.Tensor):
                attr[k] = v.tolist()
    # 清洗边属性
    for u, v, attr in nx_graph.edges.items():
        for k, val in attr.items():
            if isinstance(val, np.integer):
                attr[k] = int(val)
            elif isinstance(val, np.floating):
                attr[k] = float(val)
            elif isinstance(val, np.ndarray):
                attr[k] = val.tolist()
            elif isinstance(val, torch.Tensor):
                attr[k] = val.tolist()
    return nx_graph


pkl_path = "./data/streamspot/graphs.pkl"
save_dir = "./data/streamspot/final"
os.makedirs(save_dir, exist_ok=True)

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

for idx, (g, label) in enumerate(data):
    nx_g = g.to_networkx(node_attrs=["type"], edge_attrs=["type"])
    nx_g = clean_nx_graph(nx_g)
    json_data = nx.node_link_data(nx_g)
    out_file = f"{save_dir}/{idx+1}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, cls=CustomEncoder)
print(f"转换完成！一共生成 {len(data)} 个json文件")
