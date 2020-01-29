import torch
from torch.nn import Parameter

from ..base_renderer import BaseRenderer
from .bezier import get_associated_coordinates_bezier
from .line import get_associated_coordinates_line


def unsqueeze_right(tensor, ndim):
    return tensor.reshape(*tensor.shape, *[1]*(ndim - tensor.ndim))


def broadcast_right_tensors_no_expand(*tensors):
    max_ndim = max(tensor.ndim for tensor in tensors)
    return [unsqueeze_right(tensor, max_ndim) for tensor in tensors]


class Renderer(BaseRenderer):
    def __init__(self, raster_resolution, integer_pixel_centers=False, bezier_probes_n=160, linecaps='butt',
                 division_epsilon=1e-12, dtype=None, device=None):
        r"""Differentiable renderer based on distance fields.

        Parameters
        ----------
        raster_resolution : tuple
            ``[height, width]`` of the raster grid.
        integer_pixel_centers : bool, optional
            If True, the coordinates of the pixel centers are integer, e.g ``range(0, width)``;
            if False, the coordinates of the pixel corners are integer and the coordinates of the pixel centers are
            halves, e.g ``range(.5, width+1)``
        bezier_probes_n : int
            Number of uniform samples on Bezier curve used for finding the projection of the point on the curve.
            Ideally, it should be greater or equal to
             3 * max(|p1-p0|, |p2-p0|/2, |p3-p1|/2, |p3-p2|) / h,
            where h is the minimal required distance between the probes, e.g 1 pixel,
            see e.g https://github.com/servo/cairo/blob/master/src/cairo-mesh-pattern-rasterizer.c#L311.
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

        super().__init__(raster_resolution=raster_resolution, integer_pixel_centers=integer_pixel_centers,
                         linecaps=linecaps, division_epsilon=division_epsilon, dtype=dtype, device=device)

        # Make Bezier curve probes
        bezier_t_probes = torch.linspace(0, 1, bezier_probes_n).type(dtype).to(device)
        self.bezier_t_probes = Parameter(bezier_t_probes, requires_grad=False)

    def render(self, vector, raster_res=None, confidence_threshold=.5, sigmoid_rate=100) -> torch.Tensor:
        r"""Renders a vector drawing in a differentiable way.

        Parameters
        ----------
        vector : torch.Tensor
            Tensor of vector primitives of shape ``[batch_size, primitives_n, parameters_n]``.
            The type of the primitives is deduced from the number of parameters.
            Line segments are represented with 6 parameters: ``x0, y0, x1, y1, width, confidence``,
             where coordinates correspond to the end points.
            Bezier curves are represented with 10 parameters: ``x0, y0, x1, y1, x2, y2, x3, y3, width, confidence``,
             where coordinates correspond to the control points.
            Coordinates and width are in pixels.
        raster_res : optional
            Deprecated; the value from the constructor is used
        confidence_threshold : scalar or torch.Tensor, optional
            The confidence threshold: we render the line if its confidence exceeds the threshold.
            This parameter is broadcasted to the right, i.e as a tensor it can have shapes ``[batch_size]`` or
            ``[batch_size, primitives_n]``.
        sigmoid_rate : scalar or torch.Tensor, optional
            Parameter regulating steepness of the sigmoid: the larger the value, the more steep the change in sigmoid.
            This parameter is broadcasted to the right, i.e as a tensor it can have shapes ``[batch_size]`` or
            ``[batch_size, primitives_n]``.

        Returns
        -------
        rasterization : torch.Tensor
            Rasterization of shape ``[batch_size, primitives_n, parameters_n]``

        Notes
        -----
        The algorithm is based on a simple and a seemingly widespread idea that the rendering could be obtained as a
        thresholded distance transform of the pixel grid w.r.t. the vector primitive [1]_.

        The general idea is to associate with each primitive a potentially curvilinear orthogonal x,y coordinate system,
        isometric to the original coordinate system, in which the skeleton of the primitive is the [0, length] segment
        of the y coordinate axis and the whole offset primitive (with nonzero width) is the corresponding curvilinear
        rectangle.

        The drawing is then computed as a product of 4 indicator functions:
         - indicator (x_associated_abs <= halfwidth): the pixel is closer to the primitive skeleton than its halfwidth
         - indicator (y_associated >= 0): pixel projection lies on the primitive (and not outside first endpoint)
         - indicator (y_associated <= length): same as above, outside second endpoint
         - indicator (confidence >= .5): the primitive must exist

        To compute such indicator functions differentiably we use sigmoids.

        References
        ----------
        .. [1] Li, L., Zou, C., Zheng, Y., Su, Q., Fu, H., & Tai, C. L. (2018).
           Sketch-R2CNN: An Attentive Network for Vector Sketch Recognition.
        """
        if vector.ndim < 3:
            raise ValueError(f'Expected vector of shape [batch_size, primitives_n, parameters_n], got {vector.shape}')

        parameters_n = vector.shape[2]
        if (parameters_n != 6) and (parameters_n != 10):
            raise ValueError(f'We only handle primitives with 6 parameters -- lines, '
                             f'or with 10 parameters -- Bezier curves. Got {parameters_n} parameters.')

        # Deduce primitive type from the number of parameters # TODO: better make it explicit
        # and compute the coordinates of the pixels in the associated systems
        if parameters_n == 6:  # lines
            x_associated_abs, y_associated, length = get_associated_coordinates_line(
                vector[..., :4], self.raster_coordinates, division_epsilon=self.division_epsilon)
        elif parameters_n == 10:  # beziers
            x_associated_abs, y_associated, length = get_associated_coordinates_bezier(
                vector[..., :8], self.raster_coordinates, self.bezier_t_probes,
                division_epsilon=self.division_epsilon)

        # Prepare and broadcast things
        halfwidth = vector[..., -2] / 2
        confidence = vector[..., -1]
        del vector

        sigmoid_rate = torch.as_tensor(sigmoid_rate, dtype=self.dtype, device=self.device)
        confidence_threshold = torch.as_tensor(confidence_threshold, dtype=self.dtype, device=self.device)

        length, halfwidth, confidence, sigmoid_rate, confidence_threshold, y_associated = \
            broadcast_right_tensors_no_expand(length, halfwidth, confidence, sigmoid_rate, confidence_threshold,
                                              y_associated)

        # Compute the drawing as a product of 4 indicator functions:
        #  - indicator (x_associated_abs < halfwidth): the pixel is closer to the primitive skeleton than its halfwidth
        #  - indicator (y_associated > 0): pixel projection lies on the primitive (and not outside first endpoint)
        #  - indicator (y_associated < length): same as above, outside second endpoint
        #  - indicator (confidence > .5): the primitive must exist
        # To compute such indicator functions differentiably we use sigmoids.
        distance_indicator = halfwidth - x_associated_abs
        endpoint1_indicator = y_associated
        endpoint2_indicator = length - y_associated
        confidence_indicator = confidence - confidence_threshold
        del y_associated, x_associated_abs, length, halfwidth, confidence, confidence_threshold

        # Multiply the quantities with `sigmoid_rate` -- the larger the value, the more steep the change in sigmoid
        raster = torch.sigmoid(distance_indicator * sigmoid_rate) * \
                 torch.sigmoid(endpoint1_indicator * sigmoid_rate) * \
                 torch.sigmoid(endpoint2_indicator * sigmoid_rate) * \
                 torch.sigmoid(confidence_indicator * sigmoid_rate)
        del distance_indicator, endpoint1_indicator, endpoint2_indicator, confidence_indicator, sigmoid_rate

        # Sum the renderings of all lines
        # TODO: maybe better calculate the common distance field, w.r.t all lines at the same time,
        #  than sum individual distance fields
        raster = raster.sum(dim=1).clamp(0, 1)
        return raster
