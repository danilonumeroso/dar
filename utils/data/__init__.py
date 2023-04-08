from ._loader import load_dataset, Loader  # noqa: F401


def adj_mat(features):
    for inp in features.inputs:
        if inp.name == "adj":
            return inp.data


def edge_attr_mat(features):
    for inp in features.inputs:
        if inp.name == "A":
            return inp.data
