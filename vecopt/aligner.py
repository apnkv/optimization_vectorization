from typing import Callable, List, Any

import torch
from geomloss import SamplesLoss
from torch import optim

from vecopt.contrib.differentiable_rendering.sigmoids_renderer.renderer import Renderer
from vecopt.data_utils import get_pixel_coords_and_density_batch
from vecopt.aligner_utils import strip_confidence_grads

DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class StatefulBatchAligner:
    prestep_callbacks: List[Callable[[dict], None]]
    callbacks: List[Callable[[dict], None]]
    device: str
    step_function: Callable[[dict], Any]

    def __init__(self,
                 lines_batch: torch.Tensor,
                 images_batch: torch.Tensor,
                 init_callback: Callable[[dict], None] = None,
                 device: str = DEFAULT_DEVICE):

        self.prestep_callbacks = []
        self.callbacks = []
        self.device = device
        self.step_function = lambda x: x

        raster_ot = get_pixel_coords_and_density_batch(images_batch)
        raster_coords, raster_masses = raster_ot['coords'], raster_ot['density']
        self.state = {
            'raster': images_batch,
            'raster_coords': raster_coords,
            'raster_masses': raster_masses,

            'initial_line_batch': lines_batch.detach().clone(),
            'current_line_batch': lines_batch.detach().clone(),

            'loss_value': None,
            'current_step': -1
        }
        self.state['initial_line_batch'].requires_grad_(False)
        self.state['current_line_batch'].requires_grad_()

        self._necessary_state_keys = set(self.state.keys())

        if init_callback is not None:
            init_callback(self.state)

    def clear_custom_state(self):
        current_keys = list(self.state.keys())
        for key in current_keys:
            if key not in self._necessary_state_keys:
                del self.state[key]

    def get_line_batch(self) -> torch.Tensor:
        return self.state['current_line_batch']

    def step(self):
        self.state['current_step'] += 1
        for fn in self.prestep_callbacks:
            fn(self.state)
        result = self.step_function(self.state)
        for fn in self.callbacks:
            fn(self.state)
        return result

    def set_step_function(self, fn):
        self.step_function = fn

    def add_prestep_callback(self, fn):
        self.prestep_callbacks.append(fn)

    def add_callback(self, fn):
        self.callbacks.append(fn)


# def save_best_batch(state):
#     if 'min_loss' not in state or state['loss_value'].item() < state['min_loss']:
#         state['min_loss'] = state['loss_value'].item()
#         state['best_batch'] = state['current_line_batch'].clone().detach()


def make_default_loss_fn(ot_schedule=None, bce_schedule=None, ot_loss=None):
    ot_loss = ot_loss or SamplesLoss("sinkhorn", p=2, blur=.05, scaling=.6, reach=6.)
    bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    ot_schedule = ot_schedule or (lambda state: 1.0)
    bce_schedule = bce_schedule or (lambda state: 1.0)

    def compute_loss(state):
        vector_masses = state['vector_masses']
        vector_coords = state['vector_coords']
        raster_masses = state['raster_masses']
        raster_coords = state['raster_coords']

        bce_per_sample = bce_loss(state['render'], state['raster']).mean(dim=(1, 2))
        ot_loss_per_sample = ot_loss(vector_masses, vector_coords, raster_masses, raster_coords)

        total_loss_per_sample = bce_schedule(state) * bce_per_sample + ot_schedule(state) * ot_loss_per_sample

        state['loss_per_sample'] = total_loss_per_sample

        return total_loss_per_sample.sum()

    return compute_loss


def make_default_optimize_fn(aligner, lr=0.5, transform_grads=None, base_optimizer=optim.Adam):
    optimizer = base_optimizer((aligner.get_line_batch(),), lr=lr)
    transform_grads = transform_grads or strip_confidence_grads

    def optimize_fn(state):
        optimizer.zero_grad()
        state['loss_value'].backward()
        transform_grads(state)
        optimizer.step()

    return optimize_fn


def make_default_step_fn(loss_fn, optimize_fn, device=DEFAULT_DEVICE):
    renderer = Renderer((64, 64), linecaps='butt', device=device, dtype=torch.float32)

    def step_fn(state):
        """
        Computes absolutely needed values for OT based alignment, saves them into state,
        calls loss and optimizer callbacks.
        """
        vector_masses = renderer.render(state['current_line_batch'])

        vector_masses[:, 0, 0][vector_masses[:, 0, 0] < 1e-6] = 0.

        state['render'] = vector_masses.clone()

        render_ot = get_pixel_coords_and_density_batch(vector_masses)
        state['vector_coords'] = render_ot['coords']
        state['vector_masses'] = render_ot['density']
        state['vector_nonzero_coords'] = render_ot['nonzero_coords']

        state['loss_value'] = loss_fn(state)
        optimize_fn(state)

        return state['current_line_batch']

    return step_fn


def init_ot_aligner(aligner: StatefulBatchAligner,
                    loss_fn: Callable[[dict], torch.Tensor] = None,
                    optimize_fn: Callable[[dict], None] = None,
                    callbacks: List[Callable] = None,
                    device: str = DEFAULT_DEVICE) -> StatefulBatchAligner:
    loss_fn = loss_fn or make_default_loss_fn()
    optimize_fn = optimize_fn or make_default_optimize_fn(aligner)
    step_fn = make_default_step_fn(loss_fn, optimize_fn, device)

    callbacks = callbacks or []

    aligner.set_step_function(step_fn)
    for callback in callbacks:
        aligner.add_callback(callback)

    return aligner
