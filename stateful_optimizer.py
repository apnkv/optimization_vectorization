from typing import Callable, List, Optional, Any

import numpy as np
import torch
from geomloss import SamplesLoss
from torch import optim

from contrib.differentiable_rendering.sigmoids_renderer.renderer import Renderer
from utils import get_pixel_coords_and_density

DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class StatefulAligner:
    callbacks: List[Callable[[dict], None]]
    device: str
    step_function: Callable[[dict], Any]

    def __init__(self,
                 line_batch: torch.Tensor,
                 image: np.ndarray,
                 init_callback: Callable[[dict], None] = None,
                 device: str = DEFAULT_DEVICE):

        self.callbacks = []
        self.device = device
        self.step_function = lambda x: x

        raster_coords, raster_masses = get_pixel_coords_and_density(image, device)
        self.state = {
            'raster': torch.from_numpy(image).to(self.device),
            'raster_coords': raster_coords,
            'raster_masses': raster_masses,

            'initial_line_batch': line_batch.detach().clone(),
            'current_line_batch': line_batch.detach().clone(),

            'loss_value': None,
            'current_step': -1
        }
        self.state['initial_line_batch'].requires_grad_()
        self.state['current_line_batch'].requires_grad_()

        self._necessary_state_keys = set(self.state.keys())

        if init_callback is not None:
            init_callback(self.state)

    def cleanup_custom_state(self):
        current_keys = list(self.state.keys())
        for key in current_keys:
            if key not in self._necessary_state_keys:
                del self.state[key]

    def get_line_batch(self) -> torch.Tensor:
        return self.state['current_line_batch']

    def step(self):
        self.state['current_step'] += 1
        result = self.step_function(self.state)
        for fn in self.callbacks:
            fn(self.state)
        return result

    def set_step_function(self, fn):
        self.step_function = fn

    def add_after_step_callback(self, fn):
        self.callbacks.append(fn)


class LossComposition:
    def __init__(self):
        self.loss_fns = []

    def add(self, fn):
        self.loss_fns.append(fn)

    def __call__(self, state):
        assert len(self.loss_fns) > 0

        value = self.loss_fns[0](state)
        for loss_fn in self.loss_fns[1:]:
            value += loss_fn(state)

        return value


def store_render_difference(state):
    state['difference'] = state['render'] - state['raster']


def store_grads(state):
    state['grads'] = state['current_line_batch'].grad.clone().detach().cpu().numpy()


def save_best_batch(state):
    if 'min_loss' not in state or state['loss_value'].item() < state['min_loss']:
        state['min_loss'] = state['loss_value'].item()
        state['best_batch'] = state['current_line_batch'].clone().detach()


def make_default_loss_fn(ot_schedule=None, bce_schedule=None):
    ot_loss = SamplesLoss("sinkhorn", p=2, blur=.01)
    bce_loss = torch.nn.BCEWithLogitsLoss()

    ot_schedule = ot_schedule or (lambda state: 1.0)
    bce_schedule = bce_schedule or (lambda state: 1.0)

    def compute_loss(state):
        vector_masses = state['vector_masses']
        vector_coords = state['vector_coords']
        raster_masses = state['raster_masses']
        raster_coords = state['raster_coords']

        return bce_schedule(state) * bce_loss(state['render'], state['raster']) + \
               ot_schedule(state) * ot_loss(vector_masses, vector_coords, raster_masses, raster_coords)

    return compute_loss


def make_default_optimize_fn(aligner):
    optimizer = optim.Adam((aligner.get_line_batch(),), lr=0.5)

    def optimize_fn(state):
        optimizer.zero_grad()
        state['loss_value'].backward()
        if state['current_line_batch'].grad is not None:
            state['current_line_batch'].grad.data[:, :, 5] = 0.
        optimizer.step()

    return optimize_fn


def make_simple_aligner(line_batch: torch.Tensor,
                        image: np.ndarray,
                        loss_fn: Callable[[dict], torch.Tensor] = None,
                        optimize_fn: Callable[[dict], None] = None,
                        callbacks: Optional[List[Callable]] = None,
                        device: str = DEFAULT_DEVICE) -> StatefulAligner:
    aligner = StatefulAligner(line_batch, image)

    loss_fn = loss_fn or make_default_loss_fn()
    optimize_fn = optimize_fn or make_default_optimize_fn(aligner)

    renderer = Renderer((64, 64), linecaps='butt', device=device, dtype=torch.float32)

    def step_fn(state):
        """
        Computes absolutely needed values for OT based alignment, saves them into state,
        calls loss and optimizer callbacks.
        """
        vector_masses = renderer.render(state['current_line_batch'])[0]
        state['render'] = vector_masses.clone().detach()
        vector_masses /= vector_masses.sum()

        vector_coords = vector_masses.nonzero().float()
        vector_masses = vector_masses.reshape(-1)
        vector_masses = vector_masses[vector_masses.nonzero()]
        vector_masses = vector_masses.flatten()
        state['vector_coords'] = vector_coords
        state['vector_masses'] = vector_masses

        state['loss_value'] = loss_fn(state)
        optimize_fn(state)

        return state['current_line_batch']

    callbacks = callbacks or []

    aligner.set_step_function(step_fn)
    for callback in callbacks:
        aligner.add_after_step_callback(callback)

    return aligner
