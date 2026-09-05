import pickle
import networkx as nx
import json
import dgl

# 重点！在当前脚本里提前定义这个类，pickle加载时直接在这里找到，不去读loaddata.py
class StreamspotDataset(dgl.data.DGLDataset):
    def process(self):
        pass
    def __init__(self, name):
        super().__init__(name=name)

pkl_path = "./data/streamspot/graphs.pkl"
save_dir = "./data/streamspot"

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

for idx, (g, label) in enumerate(data):
    nx_g = g.to_networkx(node_attrs=["type"], edge_attrs=["type"])
    json_data = nx.node_link_data(nx_g)
    out_file = f"{save_dir}/{idx+1}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f)
print(f"转换完成！一共生成 {len(data)} 个json文件")
