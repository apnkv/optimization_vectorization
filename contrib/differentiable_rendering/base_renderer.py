from abc import ABC, abstractmethod

import torch
from torch.nn import Parameter


class BaseRenderer(torch.nn.Module, ABC):
    def __init__(self, raster_resolution, integer_pixel_centers=False, linecaps='butt', division_epsilon=1e-12,
                 dtype=None, device=None):
        r"""Base class for differentiable renderers.

        Parameters
        ----------
        raster_resolution : tuple
            ``[height, width]`` of the raster grid.
        integer_pixel_centers : bool, optional
            If True, the coordinates of the pixel centers are integer, e.g ``range(0, width)``;
            if False, the coordinates of the pixel corners are integer and the coordinates of the pixel centers are
            halves, e.g ``range(.5, width+1)``
        linecaps : str, optional
            One of the 'butt', 'square', 'round'. Currently, only 'butt' is supported.
            See https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/stroke-linecap
        division_epsilon : scalar, optional
            A value added to the denominator in divisions for numerical stability.
        dtype : torch.dtype, optional
            The desired data type of the raster grid and all computations.
            If None, uses a global default see ``torch.set_default_tensor_type()``
        device : torch.device, optional
            The desired device of the raster grid and all computations. If None, uses the current device for `dtype`
        """
        assert linecaps == 'butt', 'For square linecaps extend your primitives by halfwidth at each end'

        super().__init__()
        self.h, self.w = raster_resolution
        self.division_epsilon = division_epsilon
        if dtype is None:
            dtype = torch.get_default_dtype()
        self.dtype = dtype
        if device is None:
            device = torch.empty([], dtype=dtype).device  # TODO: use the correct way of finding out the default device
        self.device = device
        self.integer_pixel_centers = integer_pixel_centers

        # Make raster grid
        raster_coordinates = torch.meshgrid(torch.arange(0, self.h), torch.arange(0, self.w))
        raster_coordinates = torch.stack([raster_coordinates[1], raster_coordinates[0]]).type(dtype).to(device)
        if not integer_pixel_centers:
            raster_coordinates += .5
        self.raster_coordinates = Parameter(raster_coordinates, requires_grad=False)

    @abstractmethod
    def render(self, *args, **kwargs) -> torch.Tensor:
        r"""Render things"""
        raise NotImplementedError

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.device = self.raster_coordinates.device
        self.dtype = self.raster_coordinates.dtype
