import torch


def strip_confidence_grads(state):
    if state['current_line_batch'].grad is not None:
        state['current_line_batch'].grad.data[:, :, -1] = 0.


def compose(*fns):
    def composition(state):
        result = None
        for fn in fns:
            result = fn(state)
        return result

    return composition


def coords_only_grads(n_steps=200):
    def fn(state):
        if state['current_step'] < n_steps:
            state['current_line_batch'].grad.data[:, :, -2] = 0.

    return fn


def reduced_width_lr(multiplier=0.2):
    def fn(state):
        state['current_line_batch'].grad.data[:, :, -2] *= multiplier

    return fn


def not_too_thin(state):
    return torch.sum(torch.relu(1. - state['current_line_batch'][:, :, -2]))


def perceptual_bce(model, n_convolutions=2, weight=1.0):
    bce = torch.nn.BCEWithLogitsLoss(reduction='none')

    def fn(state):
        render = model.apply_convolutions(state['render'].unsqueeze(1), n_convolutions)
        raster = model.apply_convolutions(state['raster'].unsqueeze(1), n_convolutions)
        result = weight * bce(render, raster).mean(dim=(1, 2, 3)).sum()
        return result

    return fn


def accumulate_renders(idx=0):
    def fn(state):
        if 'renders' not in state:
            state['renders'] = []
        state['renders'].append(state['render'][idx].detach().cpu().numpy())

    return fn


class IntermediateOutputAligner:
    def __init__(self, renderer=None, ):
        pass

    def __call__(self, sample):
        pass
