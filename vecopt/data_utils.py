import abc
import os
import tempfile

import PIL
import cairo
import numpy as np
import torch
from PIL import Image, ImageOps


class SyntheticPatch:
    def __init__(self, max_x=64, max_y=64, border=4, default_line_rgba=(1, 1, 1, 1)):
        self._max_x = max_x
        self._max_y = max_y
        self._border = border
        self._default_line_rgba = default_line_rgba
        self._setup_context()

    @staticmethod
    def from_lines(lines, **init_kwargs):
        synthetic = SyntheticPatch(**init_kwargs)
        for line in lines:
            synthetic.line((line[0], line[2]), (line[1], line[3]), line[4])
        return synthetic

    def line(self, xs, ys, width=1, source_rgba=None,
             line_join=cairo.LINE_JOIN_MITER, line_cap=cairo.LINE_CAP_BUTT):

        source_rgba = source_rgba or self._default_line_rgba
        self._ctx.set_source_rgba(*source_rgba)
        self._ctx.set_line_width(width)

        self._ctx.move_to(xs[0], ys[0])
        self._ctx.line_to(xs[1], ys[1])

        self._ctx.set_line_join(line_join)
        self._ctx.set_line_cap(line_cap)

        self._ctx.stroke()
        self._ctx.close_path()
        if (xs[0], ys[0]) < (xs[1], ys[1]):
            line_coords = xs[0], ys[0], xs[1], ys[1]
        else:
            line_coords = xs[1], ys[1], xs[0], ys[0]

        self._lines.append((*line_coords, width))

    def random_line(self, **line_kwargs):
        xs = []
        ys = []
        for _ in range(2):
            xs.append(np.random.randint(self._border, self._max_x - self._border))
            ys.append(np.random.randint(self._border, self._max_y - self._border))
        width = np.random.randint(1, 6)

        return self.line(xs, ys, width, **line_kwargs)

    def random_lines(self, n_lines, random_seed=None):
        np.random.seed(random_seed)

        for _ in range(n_lines):
            self.random_line()

        self._lines.sort()

    def _setup_context(self) -> None:
        self._surface = cairo.SVGSurface(None, self._max_x, self._max_y)
        self._ctx = cairo.Context(self._surface)
        self._lines = []

        self._ctx.save()
        self._ctx.set_source_rgb(0, 0, 0)
        self._ctx.paint()
        self._ctx.restore()
        self._ctx.move_to(0, 0)

    def get_image(self, invert=False, compositing_operator=cairo.OPERATOR_MULTIPLY):
        self._ctx.set_operator(compositing_operator)

        fd, temp_filename = tempfile.mkstemp()

        self._surface.write_to_png(temp_filename)
        self._surface.flush()

        temp_image = Image.open(temp_filename).convert('L')

        image = temp_image.copy() if not invert else PIL.ImageOps.invert(temp_image)

        os.close(fd)
        temp_image.close()
        os.remove(temp_filename)

        return image

    def get_lines(self, return_width=True):
        if return_width:
            return self._lines.copy()
        else:
            return [line[:4] for line in self._lines]

    def detach_surface_and_context(self):
        surface = self._surface
        ctx = self._ctx

        self.reset()

        return surface, ctx

    def reset(self):
        self._setup_context()

    def write_to_png(self, filename):
        self._surface.write_to_png(filename)


class LinePerturbation(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def transform(self, lines):
        raise NotImplementedError('Base class LinePerturbation cannot be used')


class LineRandomShift(LinePerturbation):
    def __init__(self, endpoint_shift_range, width_shift_range):
        super(LineRandomShift, self).__init__()
        self._endpoint_shift_range = endpoint_shift_range
        self._width_shift_range = width_shift_range

    def transform(self, lines):
        new_lines = [list(line) for line in lines]
        for line in new_lines:
            for coord in range(4):
                line[coord] += np.random.randint(*self._endpoint_shift_range)

            line[4] = max(line[4] + np.random.randint(*self._width_shift_range), 1)
        return new_lines


class LineBatchRandomShift(LinePerturbation):
    def __init__(self, endpoint_shift_range, width_shift_range):
        super(LineBatchRandomShift, self).__init__()
        self._endpoint_shift_range = endpoint_shift_range
        self._width_shift_range = width_shift_range

    def transform(self, lines):
        new_lines = lines.clone()
        nondummy_mask = new_lines[:, :, 4] > 1e-6

        endpoints = new_lines[:, :, :4][nondummy_mask]
        new_lines[:, :, :4][nondummy_mask] += torch.randint(*self._endpoint_shift_range, endpoints.shape)

        width = new_lines[:, :, 4][nondummy_mask]
        new_lines[:, :, 4][nondummy_mask] = torch.clamp_min(
            width + torch.randint(*self._width_shift_range, width.shape), 1.)
        return new_lines


class LineClip(LinePerturbation):
    def __init__(self, x_lo, x_hi, y_lo, y_hi):
        super(LineClip, self).__init__()
        self._x_lo = x_lo
        self._x_hi = x_hi
        self._y_lo = y_lo
        self._y_hi = y_hi

    def transform(self, lines):
        new_lines = [list(line) for line in lines]
        for line in new_lines:
            line[0] = min(max(line[0], self._x_lo), self._x_hi)
            line[2] = min(max(line[2], self._x_lo), self._x_hi)

            line[1] = min(max(line[1], self._y_lo), self._y_hi)
            line[3] = min(max(line[3], self._y_lo), self._y_hi)

        return new_lines


class LinePerturbationPipe(LinePerturbation):
    def __init__(self, *steps):
        super(LinePerturbationPipe, self).__init__()
        self._steps = steps

    def transform(self, lines):
        new_lines = lines.clone() if isinstance(lines, torch.Tensor) else lines.copy()
        for step in self._steps:
            new_lines = step.transform(new_lines)
        return new_lines


def compute_pixel_coords(image):
    return np.argwhere(image)


def compute_pixel_density(image):
    pixel_density = image[np.nonzero(image)]
    pixel_density /= np.sum(pixel_density)
    return pixel_density


def get_pixel_coords_and_density(image, device='cuda'):
    pixel_coords = compute_pixel_coords(image)
    pixel_density = compute_pixel_density(image)

    torch_pixel_coords = torch.from_numpy(pixel_coords.astype(np.float32)).to(device)
    torch_pixel_density = torch.from_numpy(pixel_density.astype(np.float32)).to(device)

    return torch_pixel_coords, torch_pixel_density


def get_pixel_coords_and_density_batch(images, eps=1e-8):
    batch_size = images.shape[0]
    device = images.device

    nonzero_mask = images > eps
    nonzero_coords = torch.nonzero(nonzero_mask)
    render_n = nonzero_coords[:, 0]
    rc = nonzero_coords[:, 1:].type(torch.float32)

    n_nonzero_coords = nonzero_mask.sum(axis=(1, 2))
    n_coords = n_nonzero_coords.max().item()

    indices = torch.zeros(n_nonzero_coords.sum().item(), dtype=torch.int64, device=device)
    last = 0
    for elem in n_nonzero_coords:
        idx_range = torch.arange(elem.item(), device=device)
        length = idx_range.shape[0]
        indices[last:(last + length)] = idx_range
        last += length

    coords = torch.zeros(batch_size, n_coords, 2, dtype=torch.float32, device=device)
    coords[render_n, indices] = rc

    density = torch.zeros(batch_size, n_coords, dtype=torch.float32, device=device)
    density[render_n, indices] = images[nonzero_mask]
    density /= density.sum(dim=1, keepdim=True)

    return {
        'nonzero_coords': n_nonzero_coords,
        'coords': coords.to(device),
        'density': density.to(device)
    }


def wrap_lines_into_numpy_batch(lines):
    for line in lines:
        line.append(0.5)
    lines = np.array([lines], dtype=np.float32)
    return lines


def torch_to_synthetic_lines(patch_lines):
    return patch_lines.numpy()[:, :5]
