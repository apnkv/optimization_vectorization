import numpy as np
import torch


def get_associated_coordinates_bezier(vector, points, t_probes, division_epsilon=1e-12):
    batch_size, curves_n = vector.shape[:2]

    p0 = vector[..., :2].permute(2, 0, 1)
    p1 = vector[..., 2:4].permute(2, 0, 1)
    p2 = vector[..., 4:6].permute(2, 0, 1)
    p3 = vector[..., 6:8].permute(2, 0, 1)
    del vector
    raster_grid_height, raster_grid_width = points.shape[1:]
    points_n = raster_grid_height * raster_grid_width
    points = points.reshape(2, 1, 1, points_n)

    # 1. Sample probes on the curve
    probes, arc_lens = probe_cbezier_arc(t_probes, p0, p1, p2, p3)
    del t_probes
    probes.unsqueeze_(-1)
    probes_n = len(probes)
    length = arc_lens[-1].reshape(batch_size, curves_n)

    # 2. For each point find the closest probe, the distance to this probe,
    #    and the corresponding arc length
    canonical_x_abs = torch.full([batch_size, curves_n, points_n], np.inf, dtype=points.dtype, device=points.device)
    projection_probe_id = torch.empty_like(canonical_x_abs, dtype=torch.long)
    for probe_i in range(probes_n):
        dist = torch.norm(probes.data[probe_i] - points.data, dim=0)
        closer = dist.data < canonical_x_abs.data
        canonical_x_abs = dist.where(closer, canonical_x_abs)
        projection_probe_id.masked_fill_(closer, probe_i)
    del dist, closer

    projection_probe_id = projection_probe_id.reshape(1, batch_size, curves_n, points_n) \
        .expand(2, batch_size, curves_n, points_n)
    probes = probes[..., 0].permute(1, 2, 3, 0)
    projections = torch.gather(probes, dim=-1, index=projection_probe_id, sparse_grad=False)
    del probes
    distance_vectors = points - projections
    del projections, points
    canonical_x_abs = torch.norm(distance_vectors, dim=0)

    arc_lens = arc_lens.permute(1, 2, 0)
    projection_probe_id = projection_probe_id[0]
    canonical_y = torch.gather(arc_lens, dim=-1, index=projection_probe_id, sparse_grad=False)
    del arc_lens

    left_points = projection_probe_id == 0
    right_points = projection_probe_id == probes_n - 1
    del projection_probe_id

    # 3. For the points, for which the end probes are the closest,
    #    redefine the coordinates w.r.t their projection to the tangent line in the end of the curve
    # Right end
    tangent_right = torch.nn.functional.normalize(p3 - p2, dim=0, eps=division_epsilon)
    # if p3 and p2 coincide then the tangent at the right end is p2 - p1
    tangent_right = tangent_right.where((tangent_right != 0).any(dim=0),
                                        torch.nn.functional.normalize(p2 - p1, dim=0, eps=division_epsilon))
    # if p3, p2 and p1 coincide then the tangent at the right end is p1 - p0
    tangent_right = tangent_right.where((tangent_right != 0).any(dim=0),
                                        torch.nn.functional.normalize(p1 - p0, dim=0, eps=division_epsilon))
    # if all four control points coincide -- we're fd, but lets pretend that this is fine
    # at least, that's what the version for lines does

    points_from_right = distance_vectors.where(right_points, distance_vectors.new_zeros([]))
    canonical_y_right = points_from_right[0] * tangent_right[0].unsqueeze(-1) \
                        + points_from_right[1] * tangent_right[1].unsqueeze(-1)
    canonical_y_right = canonical_y_right.clamp(min=0)
    canonical_y = canonical_y + canonical_y_right
    del canonical_y_right

    canonical_x_right = points_from_right[1] * tangent_right[0].unsqueeze(-1) \
                        - points_from_right[0] * tangent_right[1].unsqueeze(-1)
    canonical_x_right = canonical_x_right.abs()
    canonical_x_abs = canonical_x_right.where(right_points, canonical_x_abs)
    del points_from_right, canonical_x_right, right_points, tangent_right

    # Left end
    tangent_left = torch.nn.functional.normalize(p1 - p0, dim=0, eps=division_epsilon)
    tangent_left = tangent_left.where((tangent_left != 0).any(dim=0),
                                      torch.nn.functional.normalize(p2 - p1, dim=0, eps=division_epsilon))
    tangent_left = tangent_left.where((tangent_left != 0).any(dim=0),
                                      torch.nn.functional.normalize(p3 - p2, dim=0, eps=division_epsilon))
    del p0, p1, p2, p3

    points_from_left = distance_vectors.where(left_points, distance_vectors.new_zeros([]))
    del distance_vectors
    canonical_y_left = points_from_left[0] * tangent_left[0].unsqueeze(-1) \
                       + points_from_left[1] * tangent_left[1].unsqueeze(-1)
    canonical_y_left = canonical_y_left.clamp(max=0)
    canonical_y = canonical_y + canonical_y_left
    del canonical_y_left

    canonical_x_left = points_from_left[1] * tangent_left[0].unsqueeze(-1) \
                       - points_from_left[0] * tangent_left[1].unsqueeze(-1)
    canonical_x_left = canonical_x_left.abs()
    canonical_x_abs = canonical_x_left.where(left_points, canonical_x_abs)
    del points_from_left, canonical_x_left, left_points, tangent_left

    canonical_x_abs = canonical_x_abs.reshape(batch_size, curves_n, raster_grid_height, raster_grid_width)
    canonical_y = canonical_y.reshape(batch_size, curves_n, raster_grid_height, raster_grid_width)
    return canonical_x_abs, canonical_y, length


def probe_cbezier_arc(t, p0, p1, p2, p3):
    r"""
    Parameters
    ----------
    t : torch.Tensor
        of shape [points_n]
    p0 : torch.Tensor
        of shape [2, batches_n, curves_n]
    p1 : torch.Tensor
        of shape [2, batches_n, curves_n]
    p2 : torch.Tensor
        of shape [2, batches_n, curves_n]
    p3 : torch.Tensor
        of shape [2, batches_n, curves_n]

    Returns
    -------
    points : torch.Tensor
        of shape [points_n, 2, batches_n, curves_n]
    arc_len : torch.Tensor
        of shape [points_n, batches_n, curves_n]
    """
    t = t.reshape(-1, 1, 1, 1)
    omt = 1 - t
    c0 = omt ** 3
    c1 = omt ** 2 * t * 3
    c2 = omt * t ** 2 * 3
    c3 = t ** 3
    del omt, t
    p = p0 * c0 + p1 * c1 + p2 * c2 + p3 * c3
    del c0, c1, c2, c3, p0, p1, p2, p3

    arc_len = p[1:] - p[:-1]
    arc_len = arc_len.norm(dim=1)
    arc_len = arc_len.cumsum(dim=0)
    arc_len = torch.nn.functional.pad(arc_len, [0, 0, 0, 0, 1, 0])

    return p, arc_len
