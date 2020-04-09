import torch

from geomloss import SamplesLoss
from contrib.differentiable_rendering.sigmoids_renderer.renderer import Renderer


DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_LOSS = SamplesLoss("sinkhorn", p=2, blur=.01)
DEFAULT_RENDERER = Renderer((64, 64),
                            linecaps='butt',
                            device=DEFAULT_DEVICE,
                            dtype=torch.float32)


def optimize_line_batch(line_batch, raster_coords, raster_masses, n_iters=50, sinkhorn_steps=50,
                        optimize_width=True, lr=0.2, loss=DEFAULT_LOSS, coord_only_steps=None, bce_image=None,
                        renderer=DEFAULT_RENDERER, length_loss=0., width_loss=0., width_lr=0.1, pos_weight=1.,
                        image=None, mse=0., bce=0., ot=1., return_batches_by_step=False, grads=None):

    mse_loss = torch.nn.MSELoss()
    bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]).to(line_batch.get_device()))

    batches_by_step = [line_batch.detach().cpu()] if return_batches_by_step else []
    if grads is not None:
        grads.append(torch.zeros_like(line_batch).detach().cpu())

    if image is not None:
        image = torch.from_numpy(image).to(line_batch.get_device())

    line_batch.requires_grad_(True)

    initial_length = torch.sqrt((line_batch[:, :, 0] - line_batch[:, :, 2]) ** 2
                                + (line_batch[:, :, 1] - line_batch[:, :, 3]) ** 2).detach()
    initial_width = line_batch[:, :, 4].detach()

    for step in range(n_iters):

        vector_masses = renderer.render(line_batch)[0]

        if image is not None and mse > 0.:
            mse_part = mse * mse_loss(vector_masses, image)
        else:
            mse_part = 0.
        if image is not None and bce > 0.:
            bce_part = bce * bce_loss(vector_masses, image)
        else:
            bce_part = 0.

        vector_coords = vector_masses.nonzero().float()
        vector_masses = vector_masses.reshape(-1)
        vector_masses = vector_masses[vector_masses.nonzero()]
        vector_masses = (vector_masses / vector_masses.sum()).flatten()

        if line_batch.grad is not None:
            line_batch.grad.data.zero_()

        sample_loss = ot * loss(vector_masses, vector_coords, raster_masses, raster_coords)
        if step >= sinkhorn_steps:
            sample_loss *= 0.

        if length_loss > 0.:
            sample_loss += length_loss * torch.mean(
                initial_length - torch.sqrt((line_batch[:, :, 0] - line_batch[:, :, 2]) ** 2
                                            + (line_batch[:, :, 1] - line_batch[:, :, 3]) ** 2))

        if width_loss > 0.:
            sample_loss += width_loss * torch.mean((line_batch[:, :, 4] - initial_width) ** 2)

        sample_loss += mse_part
        sample_loss += bce_part

        sample_loss.backward()

        if grads is not None:
            grads.append(line_batch.grad.data.detach().cpu())

        g_line_batch = line_batch.grad.data
        if not optimize_width:
            g_line_batch[:, :, 4] = 0.
        g_line_batch[:, :, 4] *= width_lr

        g_line_batch[:, :, 5] = 0.

        if coord_only_steps is not None and step < coord_only_steps:
            g_line_batch[:, :, 4] = 0.
            g_line_batch[:, :, 5] = 0.

        line_batch.data -= lr * g_line_batch
        if return_batches_by_step:
            batches_by_step.append(line_batch.detach().cpu())

    if return_batches_by_step:
        return line_batch, batches_by_step
    else:
        return line_batch

