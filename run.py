import clrs
import os
import torch
import typer

from config.hyperparameters import HP_SPACE
from functools import partial
from nn.models import EncodeProcessDecode, MF_Net, MF_NetPipeline
from pathlib import Path
from statistics import mean, stdev
from utils.data import load_dataset
from utils.experiments import evaluate
from utils.types import Algorithm
from norm import set_seed
from norm.experiments import Experiment, init_runs, run_exp
from norm.io import dump, load

app = typer.Typer(add_completion=False)


def choose_default_type(name: str):
    assert name in ['float16', 'float32']

    if name == 'float16':
        return torch.HalfTensor

    return torch.FloatTensor


def choose_model(name: str):
    assert name in ["epd", "mf_net", "mf_net_pipe", "mf_net_res"]

    if name == "epd":
        model_class = EncodeProcessDecode
    elif name == "mf_net":
        model_class = MF_Net
    else:
        model_class = MF_NetPipeline

    return model_class


def choose_hint_mode(mode: str):

    assert mode in ["io", "o", "none"]

    if mode == "io":
        encode_hints, decode_hints = True, True
    elif mode == "o":
        encode_hints, decode_hints = False, True
    elif mode == "none":
        encode_hints, decode_hints = False, False

    return encode_hints, decode_hints


def split_probes(feedback):
    _, source, tail, weights, adj = feedback.features.inputs
    return (
        source.data.squeeze().numpy().argmax(-1),
        tail.data.squeeze().numpy().argmax(-1),
        weights.data.squeeze().numpy(),
        adj.data.squeeze().numpy()
    )


def _preprocess_yaml(config):
    assert 'algorithm' in config.keys()
    assert 'runs' in config.keys()
    assert 'experiment' in config.keys()

    for key in config['runs'].keys():
        if key == 'hp_space':
            config['runs'][key] = HP_SPACE[config['runs'][key]]
            continue

    return config


@app.command()
def valid(exp_path: Path,
          data_path: Path,
          model: str = "epd",
          hint_mode: str = "io",
          max_steps: int = None,
          num_cpus: int = None,
          num_gpus: int = 1,
          nw: int = 5,
          no_feats: str = 'adj',
          noise: bool = False,
          processor: str = 'pgn',
          aggregator: str = 'max',
          save_path: Path = './runs',
          dtype: str = 'float32',
          num_test_trials: int = 5,
          seed: int = None,):

    torch.set_default_tensor_type(choose_default_type(dtype))

    assert aggregator in ['max', 'sum', 'mean']
    assert processor in ['mpnn', 'pgn']

    if seed is None:
        seed = int.from_bytes(os.urandom(2), byteorder="big")

    encode_hints, decode_hints = choose_hint_mode(hint_mode)
    model_class = choose_model(model)

    configs = _preprocess_yaml(load(exp_path))

    alg = configs['algorithm']
    set_seed(seed)

    print("loading val...")
    vl_sampler, _ = load_dataset('val', alg, folder=data_path)
    print("loading test...")
    ts_sampler, _ = load_dataset('test', alg, folder=data_path)
    print("loading tr...")
    tr_sampler, spec = load_dataset('train', alg, folder=data_path)

    print("loading done")

    model_fn = partial(model_class,
                       spec=spec,
                       dummy_trajectory=tr_sampler.next(1),
                       decode_hints=decode_hints,
                       encode_hints=encode_hints,
                       add_noise=noise,
                       no_feats=no_feats.split(','),
                       max_steps=max_steps,
                       processor=processor,
                       aggregator=aggregator)

    runs = init_runs(seed=seed,
                     model_fn=model_fn,
                     optim_fn=torch.optim.SGD,
                     **configs['runs'])

    experiment = Experiment(runs=runs,
                            evaluate_fn=evaluate,
                            save_path=save_path,
                            num_cpus=num_cpus if num_cpus else num_gpus * nw,
                            num_gpus=num_gpus,
                            nw=nw,
                            num_test_trials=num_test_trials,
                            **configs['experiment'])

    dump(dict(
        alg=alg,
        data_path=str(data_path),
        hint_mode=hint_mode,
        model=model,
        aggregator=aggregator,
        processor=processor,
        no_feats=no_feats.split(','),
        seed=seed,
    ), save_path / experiment.name / 'config.json')

    print(f"Experiment name: {experiment.name}")

    run_exp(experiment=experiment,
            tr_set=tr_sampler,
            vl_set=vl_sampler,
            ts_set=ts_sampler,
            save_path=save_path)


@app.command()
def test(alg: Algorithm,
         test_path: Path,
         data_path: Path,
         max_steps: int = None,
         test_set: str = 'test',):

    from utils.metrics import eval_categorical, masked_mae
    ts_sampler, spec = load_dataset(test_set, alg.value, folder=data_path)
    best_run = load(test_path / 'best_run.json')['config']
    config = load(test_path / 'config.json')

    hint_mode = config['hint_mode']

    encode_hints, decode_hints = choose_hint_mode(hint_mode)
    model_class = choose_model(config['model'])

    feedback = ts_sampler.next()
    runs = []

    adj = feedback.features.inputs[-2].data.numpy()

    def predict(features, outputs, i):

        model = model_class(spec=spec,
                            dummy_trajectory=ts_sampler.next(1),
                            num_hidden=best_run['num_hidden'],
                            alpha=best_run['alpha'],
                            aggregator=config['aggregator'],
                            processor=config['processor'],
                            max_steps=max_steps,
                            no_feats=config['no_feats'],
                            decode_hints=decode_hints,
                            encode_hints=encode_hints,
                            optim_fn=torch.optim.Adam)
        model.restore_model(test_path / f'trial_{i}' / 'model_0.pth', 'cuda')

        preds, aux = model.predict(features)
        for key in preds:
            preds[key].data = preds[key].data.cpu()

        metrics = {}
        for truth in feedback.outputs:
            type_ = preds[truth.name].type_
            y_pred = preds[truth.name].data.numpy()
            y_true = truth.data.numpy()

            if type_ == clrs.Type.SCALAR:
                metrics[truth.name] = masked_mae(y_pred, y_true * adj).item()

            elif type_ == clrs.Type.CATEGORICAL:
                metrics[truth.name] = eval_categorical(y_pred, y_true).item()

        dump(preds, test_path / f'trial_{i}' / f'preds_{i}.{test_set}.pkl')
        dump(model.net_.flow_net.h_t.cpu(), test_path / f'trial_{i}' / f'H{i}.{test_set}.pth')
        dump(model.net_.flow_net.edge_attr.cpu(), test_path / f'trial_{i}' / f'E{i}.{test_set}.pth')
        return metrics

    for i in range(5):
        if not (test_path / f'trial_{i}' / 'model_0.pth').exists():
            continue

        runs.append(predict(feedback.features, feedback.outputs, i))
        torch.cuda.empty_cache()
    dump(runs, test_path / f'scores.{test_set}.json')

    for key in runs[0]:
        out = [evals[key] for evals in runs]
        print(key, mean(out), "pm", stdev(out) if len(out) > 1 else 0)


if __name__ == '__main__':
    app()
