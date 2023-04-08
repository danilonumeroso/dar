def set_seed(seed):
    import numpy as np
    import torch
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_date():
    from datetime import datetime
    return datetime.utcnow().isoformat()[:-7]


def is_not_done_broadcast(lengths, i, tensor):
    import torch
    is_not_done = torch.as_tensor((lengths > i + 1) * 1.0, dtype=torch.float32).to(tensor.device)
    while len(is_not_done.shape) < len(tensor.shape):
        is_not_done = is_not_done.unsqueeze(-1)
    return is_not_done
