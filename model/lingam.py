import numpy as np 

def get_markov_blanket(B, var_names, target="Y", threshold=0.05):
    """
    提取 target 的 Markov Blanket，并返回索引。
    B: 邻接矩阵 (numpy array, p x p)
    var_names: 变量名 list
    target: 目标变量名
    threshold: 边权重阈值
    """

    p = len(var_names)
    idx = var_names.index(target)

    # 阈值化
    B = (np.abs(B) > threshold) * B

    # 父节点: X_j -> target
    parents = [var_names[j] for j in range(p) if B[idx, j] != 0]

    # 子节点: target -> X_i
    children = [var_names[i] for i in range(p) if B[i, idx] != 0]

    # 配偶节点: 共同子节点的另一个父母
    spouses = set()
    for child in children:
        c_idx = var_names.index(child)
        for j in range(p):
            if B[c_idx, j] != 0 and var_names[j] != target:
                spouses.add(var_names[j])

    # Markov Blanket
    MB = set(parents) | set(children) | spouses

    # 输出字典
    result = {
        "parents": parents,
        "children": children,
        "spouses": list(spouses),
        "MB": list(MB)
    }
    print(result)

    # 返回索引
    MB_indices = [var_names.index(v) for v in MB]

    return  MB_indices

