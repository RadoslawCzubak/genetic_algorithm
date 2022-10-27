from typing import List


def is_list_of_unique_objects(l: List) -> bool:
    test = []
    for jj, _ in enumerate(l):
        for j, _ in enumerate(l):
            if jj != j:
                test.append(l[jj] is l[j])
    return any(test)
