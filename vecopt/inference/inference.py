import torch
import pickle
from geomloss import SamplesLoss
from torch import optim

from vecopt.aligner import (
    StatefulBatchAligner,
    init_ot_aligner,
    make_default_loss_fn,
    make_default_optimize_fn,
)
from vecopt.aligner_utils import (
    LossComposition, perceptual_bce, strip_confidence_grads,
    compose, coords_only_grads, save_best_batch, not_too_thin
)
from vecopt.contrib.differentiable_rendering.sigmoids_renderer.renderer import Renderer
from vecopt.crossing_model import CrossingRefinerFull
from vecopt.data_utils import qbezier_to_cbezier


def zero_primitives_outside_of_render(vectors, renderer):
    vectors.requires_grad_()
    img_mean = renderer.render(vectors).mean()
    img_mean.backward()
    grad = vectors.grad
    vectors.requires_grad_(False)
    vectors[(torch.abs(grad) > 1e-5).sum(dim=2) == 0] = 0.
    return vectors


class IntermediateOutputAligner:
    def __init__(self, aligner: StatefulBatchAligner,
                 batch_size=64,
                 n_steps=500,
                 renderer=None,
                 crossing_model=None,
                 infer_crossings=True,
                 use_best_batch=False):

        self.aligner = aligner
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.device = aligner.device
        self.renderer = renderer or Renderer((64, 64), linecaps='butt', device=self.device, dtype=torch.float32)

        self.infer_crossings = infer_crossings
        self.crossing_model = crossing_model or CrossingRefinerFull().to(self.device) if infer_crossings else None
        self.verbose = True
        self.use_best_batch = use_best_batch

    def __call__(self, sample):
        n_params = sample['patches_vector'].shape[-1]
        assert n_params in (6, 8, 10)
        if n_params == 8:
            sample['patches_vector'] = qbezier_to_cbezier(sample['patches_vector'])

        all_rasters = torch.from_numpy(sample['patches_rgb'])
        nonempty_patches = (all_rasters.mean(dim=(1, 2)) != 255.).view(-1)
        all_rasters = all_rasters[nonempty_patches]
        all_rasters = torch.scalar_tensor(1.) - all_rasters / 255.
        patches_vector = sample['patches_vector'].clone()
        all_vectors = patches_vector[nonempty_patches]

        for batch_start in range(0, len(all_rasters), self.batch_size):
            raster_batch = all_rasters[batch_start:batch_start + self.batch_size]
            raster = raster_batch.squeeze(3).to(self.device).type(torch.float32)

            if self.infer_crossings:
                raster = (0.5 * self.crossing_model.forward(raster.unsqueeze(1)).squeeze(1).detach() + 0.5 * raster) \
                         * (raster > 0.5)
            else:
                raster /= 2 * raster.max(dim=2, keepdim=True).values.max(dim=1, keepdim=True).values

            vector = all_vectors[batch_start:batch_start + self.batch_size].to(self.device).type(torch.float32)
            # important: set half-brightness on all primitives
            vector[:, :, -1] = 0.5

            vector = zero_primitives_outside_of_render(vector, self.renderer)
            # TODO: random inits should go here

            self.aligner.clear_custom_state()
            self.aligner.load_batches(vector, raster)

            for _ in range(self.n_steps):
                self.aligner.step()

            if not self.use_best_batch:
                all_vectors[batch_start:batch_start + self.batch_size] = \
                    self.aligner.state['current_line_batch'].clone().detach().cpu()
            else:
                all_vectors[batch_start:batch_start + self.batch_size] = \
                    self.aligner.state['best_line_batch'].clone().detach().cpu()

            patches_vector[nonempty_patches] = all_vectors

        return patches_vector


def load_crossing_model(path, device):
    crossing_model = CrossingRefinerFull().to(device)
    crossing_model.load_state_dict(torch.load(path))
    _ = crossing_model.train(False)
    return crossing_model


def make_parameterized_aligner(device, config, crossing_model=None):
    if crossing_model is None:
        crossing_model = load_crossing_model(config['crossing_model_weights'], device)

    loss = LossComposition()
    ot_loss = SamplesLoss("sinkhorn", **config['ot_loss'])
    loss.add(make_default_loss_fn(
        bce_schedule=(lambda state: 1.0) if config.get('bce_loss_enabled', False) else (lambda state: 0.0),
        ot_schedule=(lambda state: 1.0) if config.get('ot_loss_enabled', True) else (lambda state: 0.0),
        ot_loss=ot_loss
    ))
    if 'perceptual_bce' in config:
        for layer in config['perceptual_bce']:
            loss_component = perceptual_bce(crossing_model, layer)
            if 'perceptual_bce_from_step' not in config:
                loss.add(loss_component)
            else:
                def perceptual_loss(state):
                    step = state['current_step']
                    if step >= config['perceptual_bce_from_step']:
                        return loss_component(state)
                    else:
                        return torch.scalar_tensor(0.0)
                loss.add(perceptual_loss)

    if not config.get('ot_loss_enabled', True):
        loss.add(not_too_thin)

    grad_transformer = compose(strip_confidence_grads, coords_only_grads(config['coord_only_grads']))

    aligner = StatefulBatchAligner(device=device)
    init_ot_aligner(aligner, loss_fn=loss, device=device,
                    optimize_fn=make_default_optimize_fn(
                        aligner,
                        lr=config['lr'],
                        transform_grads=grad_transformer,
                        base_optimizer=optim.Adam,
                    ))

    if config.get('use_best_batch', False):
        aligner.add_callback(save_best_batch)

    return aligner


def make_intermediate_output_aligner(device, config):
    crossing_model = load_crossing_model(config['crossing_model_weights'], device)
    aligner = make_parameterized_aligner(device, config, crossing_model=crossing_model)
    return IntermediateOutputAligner(aligner,
                                     batch_size=config['batch_size'],
                                     n_steps=config['n_steps'],
                                     crossing_model=crossing_model,
                                     infer_crossings=config['infer_crossings'],
                                     use_best_batch=config.get('use_best_batch', False)
                                     )


def load_intermediate_result(path):
    with open(path, 'rb') as handle:
        sample = pickle.load(handle)
    return sample
