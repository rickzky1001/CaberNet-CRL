import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import matplotlib.pyplot as plt
import networkx as nx

import os
import matplotlib.pyplot as plt
import networkx as nx

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def plot_SCM(B, var_names, target=None, threshold=0.05, layout="circular",
             node_size=1500, show_weights=False, directory=None,
             colors=None, return_sets=False):
    """
    画因果图 (SCM)，并高亮 target 的 Pa/Ch/Sp。
    """
    n = B.shape[0]
    assert len(var_names) == n, f"len(var_names)={len(var_names)} 与 B.shape[0]={n} 不一致"

    # 默认颜色
    if colors is None:
        colors = {
            "target": "#bdbdbd",   # gray
            "pa":     "#a9d18e",   # light green
            "ch":     "#ffe699",   # light yellow
            "sp":     "#d86869", # light red
            "other":  "#87ceeb"    # skyblue
        }

    # 构图
    G = nx.DiGraph()
    G.add_nodes_from(var_names)

    # 根据阈值添加边
    edge_labels = {}
    for i in range(n):
        for j in range(n):
            if i != j and abs(B[i, j]) > threshold:
                G.add_edge(var_names[i], var_names[j])
                if show_weights:
                    edge_labels[(var_names[i], var_names[j])] = round(B[i, j], 2)

    # 计算 Pa/Ch/Sp
    Pa_names, Ch_names, Sp_names = set(), set(), set()
    if target is not None:
        if target not in var_names:
            raise ValueError(f"target '{target}' 不在 var_names 中。")
        y_idx = var_names.index(target)

        # Parents: i -> Y
        Pa_idx = [i for i in range(n) if i != y_idx and abs(B[i, y_idx]) > threshold]
        Pa_names = {var_names[i] for i in Pa_idx}

        # Children: Y -> j
        Ch_idx = [j for j in range(n) if j != y_idx and abs(B[y_idx, j]) > threshold]
        Ch_names = {var_names[j] for j in Ch_idx}

        # Spouses: 其他共同父 -> 任一 child
        Sp_tmp = set()
        for j in Ch_idx:
            for k in range(n):
                if k not in (y_idx, j) and abs(B[k, j]) > threshold:
                    Sp_tmp.add(var_names[k])
        # 仅保留不在 Pa/Ch 的作为 Sp
        Sp_names = Sp_tmp - Pa_names - Ch_names - {target}

    # 节点着色
    node_colors = []
    for v in var_names:
        if v == target:
            node_colors.append(colors["target"])
        elif v in Pa_names:
            node_colors.append(colors["pa"])
        elif v in Ch_names:
            node_colors.append(colors["ch"])
        elif v in Sp_names:
            node_colors.append(colors["sp"])
        else:
            node_colors.append(colors["other"])

    # 布局
    if layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "shell":
        shells = []
        if Pa_names: shells.append(sorted(Pa_names))
        shells.append([target] if target is not None else [])
        if Ch_names: shells.append(sorted(Ch_names))
        if Sp_names: shells.append(sorted(Sp_names))
        other = [v for v in var_names if v not in set().union(*shells)]
        if other: shells.append(other)
        pos = nx.shell_layout(G, nlist=[s for s in shells if s])
    elif layout == "kamada":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "planar":
        try:
            pos = nx.planar_layout(G)
        except Exception:
            pos = nx.spring_layout(G, seed=42)
    else:
        pos = nx.spring_layout(G, seed=42)

    # 字体大小
    font_size = max(8, int(np.mean(node_size) // 200))

    # -------- LaTeX 格式标签 --------
    labels = {}
    for i, v in enumerate(var_names):
        if v == target:
            labels[v] = "$Y$"
        else:
            labels[v] = f"$Z_{{{i}}}$"

    # 画图
    plt.figure(figsize=(9, 7))
    nx.draw(G, pos,
            labels=labels,
            with_labels=True,
            node_size=node_size,
            node_color=node_colors,
            edgecolors="black",
            linewidths=2.0,
            font_size=font_size,
            font_family="serif",
            arrowsize=20,
            edge_color="black",
            width=2.0)

    if show_weights and edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=max(6, font_size-2))

    if directory:
        os.makedirs(os.path.dirname(directory), exist_ok=True)
        plt.savefig(directory, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"[Saved] Graph saved to {directory}")
    else:
        plt.show()

    if return_sets:
        return Pa_names, Ch_names, Sp_names


def analyze_latent(J, feature_names, topk=3):
    H, d = J.shape
    for i in range(H):
        contribs = J[i]
        top_idx = contribs.argsort()[::-1][:topk]
        print(f"r{i}: ", [(feature_names[j], contribs[j]) for j in top_idx])

def rank_features_with_gate(gate, feature_names, topk=None):
    # softmax
    exp_gate = np.exp(gate - np.max(gate))  # 数值稳定
    gate_soft = exp_gate / exp_gate.sum()

    # 排序
    idx_sorted = np.argsort(-gate_soft)  # 从大到小
    if topk is not None:
        idx_sorted = idx_sorted[:topk]

    sorted_features = [(feature_names[i], float(gate_soft[i])) for i in idx_sorted]
    for name, w in sorted_features:
        print(f"{name}: {w:.4f}")
    return None

if __name__=='__main__':
    lingam_threshold=0.005
    B = np.load("result/SCM/B_matrix.npy")
    feature_names=['temperature','humidity','light','co2','tvoc','pressure','air_temperature','is_work']
    gate = np.load("result/SCM/gates.npy")
    rank_features_with_gate(gate, feature_names, topk=None)

    hidden_size=B.shape[0]-1
    var_names = [f'Z{i}' for i in range(hidden_size)] + ["Y"]
    plot_SCM(B, var_names, target="Y",layout='circular', node_size=5600,threshold=lingam_threshold,directory='result/SCM/SCMs.png',show_weights=False)

    J = np.load("result/SCM/J_matrix.npy")
    J_debiased = np.load("result/SCM/J_debiased_matrix.npy")
    print('original')
    analyze_latent(J, feature_names, topk=8)
    print('debiased')
    analyze_latent(J_debiased, feature_names, topk=8)