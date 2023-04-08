from functools import partial
from norm.experiments.samplers import integer, uniform


_LARGE = {
    'model': dict(num_hidden=partial(integer, a=16, b=512),
                  alpha=partial(uniform, a=0, b=0)),
    'optim': dict(
        lr=partial(uniform, a=1e-5, b=1e-1),
        weight_decay=partial(uniform, a=1e-5, b=1e-1)
    )
}

_SMALL_FF = {
    'model': dict(num_hidden=partial(integer, a=64, b=72),  # 114-138
                  alpha=partial(uniform, a=0, b=1)),
    'optim': dict(
        lr=partial(uniform, 1e-3, b=1e-2),
        weight_decay=partial(uniform, a=1e-3, b=4e-3)
    )
}

_ONE_FF = {
    'model': dict(num_hidden=lambda: 68,  # 114-138
                  alpha=partial(uniform, a=0, b=0)),
    'optim': dict(
        lr=lambda: 0.009341493994646139,
        weight_decay=lambda: 0.003420373065077989,
    )
}

_ONE_FF_MC = {
    'model': dict(num_hidden=lambda: 65,
                  alpha=partial(uniform, a=0, b=0)),
    'optim': dict(
        lr=lambda: 0.009868199084919982,
        weight_decay=lambda: 0.0017345516681916279
    )
}

HP_SPACE = {
    'dual_sp_large': _LARGE,
    'ff_large': _LARGE,
    'ff_mc_large': _LARGE,
    'ff_small': _SMALL_FF,
    'ff_mc_small': _SMALL_FF,
    'ff_one': _ONE_FF,
    'ff_mc_one': _ONE_FF_MC
}
