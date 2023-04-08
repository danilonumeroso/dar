def constraints_accuracy(y, w, adj):
    from numpy import expand_dims, transpose

    y = expand_dims(y, axis=-1)
    cons_value = transpose(y, (0, 2, 1)) - y
    cons = ((cons_value * adj <= w))

    return (cons * adj).sum() / adj.sum()


def objective_node_accuracy(path_preds, truth):
    accuracy = 0.0
    for path_a, path_b in zip(path_preds, truth):
        if path_a[-1] == path_b[-1]:
            accuracy += 1

    return accuracy / len(truth)


def overall_accuracy(path_preds, truth):
    accuracy = 0.0
    for path_a, path_b in zip(path_preds, truth):
        if len(path_a) == len(path_b) and path_a == path_b:
            accuracy += 1

    return accuracy / len(truth)
