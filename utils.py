import numpy as np
import tempfile
import cairo
import PIL
import os
from PIL import Image, ImageOps
from typing import Tuple


class Synthetic:
    def __init__(self, max_x=64, max_y=64, border=4, default_line_rgba=(1, 1, 1, 1)):
        self._max_x = max_x
        self._max_y = max_y
        self._border = border
        self._default_line_rgba = default_line_rgba
        self._setup_context()

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

    def random_lines(self, n_lines):
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

    def get_image(self, invert=False):
        self._ctx.set_operator(cairo.OPERATOR_MULTIPLY)

        _, temp_filename = tempfile.mkstemp()

        self._surface.write_to_png(temp_filename)
        self._surface.flush()

        temp_image = Image.open(temp_filename).convert('L')

        image = temp_image.copy() if not invert else PIL.ImageOps.invert(temp_image)

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
