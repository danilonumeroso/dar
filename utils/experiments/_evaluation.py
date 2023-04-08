import clrs
from utils.metrics import accuracy, eval_one, mse, mask_fn, eval_categorical, masked_mae, masked_mse

_Type = clrs.Type


def evaluate(model, feedback, extras=None, verbose=False):
    out = {}
    predictions, (raw_preds, aux) = model.predict(feedback.features)

    if extras:
        out.update(extras)

    if verbose and aux:
        losses, total_loss = model.verbose_loss(feedback, raw_preds, aux)
        out.update(losses)
        out.update({'val_loss': total_loss})

    out.update(_eval_preds(predictions, feedback, verbose))

    return out


def _eval_preds(preds, feedback, verbose=False):
    evals = {}
    extras = {}

    for truth in feedback.outputs:
        assert truth.name in preds
        pred = preds[truth.name]
        assert pred.name == truth.name
        assert pred.location == truth.location
        assert pred.type_ == truth.type_

        y_hat = pred.data.cpu().numpy()
        y = truth.data.numpy()

        if truth.type_ == clrs.Type.SCALAR:
            evals[truth.name + "_mae"] = masked_mae(y_hat, y).item()
            evals[truth.name + "_mse"] = masked_mse(y_hat, y).item()
        else:
            evals[truth.name] = EVAL_FN[truth.type_](y_hat, y).item()

    evals['score'] = evals['f_mse']  # sum([v for v in evals.values()]) / len(evals)

    if verbose:
        evals = {
            **evals,
            **extras,
        }

    return evals


EVAL_FN = {
    clrs.Type.SCALAR: mse,
    clrs.Type.MASK: mask_fn,
    clrs.Type.MASK_ONE: eval_one,
    clrs.Type.CATEGORICAL: eval_categorical,
    clrs.Type.POINTER: accuracy
}
