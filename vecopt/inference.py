import torch
from tqdm import tqdm

from vecopt.aligner import StatefulBatchAligner
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
                 verbose=True):

        self.aligner = aligner
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.device = aligner.device
        self.renderer = renderer or Renderer((64, 64), linecaps='butt', device=self.device, dtype=torch.float32)

        self.infer_crossings = infer_crossings
        self.crossing_model = crossing_model or CrossingRefinerFull().to(self.device) if infer_crossings else None

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

            self.aligner.clear_custom_state()
            self.aligner.load_batches(vector, raster)

            for _ in tqdm(range(self.n_steps)):
                self.aligner.step()

            all_vectors[batch_start:batch_start + self.batch_size] = \
                self.aligner.state['current_line_batch'].clone().detach().cpu()

            patches_vector[nonempty_patches] = all_vectors

        return patches_vector
