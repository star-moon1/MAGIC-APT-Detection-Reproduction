import pickle
import json
import networkx as nx

# 读取已有的graphs.pkl
with open("./data/streamspot/graphs.pkl", "rb") as f:
    graph_list = pickle.load(f)

# 批量导出1.json ~ 600.json
for idx, g in enumerate(graph_list):
    json_data = nx.node_link_data(g)
    out_file = f"./data/streamspot/{idx+1}.json"
    with open(out_file, "w", encoding="utf-8") as f_out:
        json.dump(json_data, f_out)

print(f"成功导出 {len(graph_list)} 个json文件！")
