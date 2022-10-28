from typing import List
import numpy as np

def to_adj_matrix(string: str):
    lines = string.split('\n')
    adj_mat = []
    for line in lines:
        line = line.strip()
        line = [val.strip() for val in line.split(",")]
        clear_line = []
        for v in line:
            if len(v) > 0:
                clear_line.append(int(v))
        adj_mat.append(clear_line)
    return np.matrix(adj_mat)

def is_list_of_unique_objects(l: List) -> bool:
    test = []
    for jj, _ in enumerate(l):
        for j, _ in enumerate(l):
            if jj != j:
                test.append(l[jj] is l[j])
    return any(test)
