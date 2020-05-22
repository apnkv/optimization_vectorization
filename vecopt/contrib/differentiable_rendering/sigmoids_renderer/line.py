import torch


def get_associated_coordinates_line(vector, points, division_epsilon=1e-12):
    batch_size, curves_n = vector.shape[:2]

    p0 = vector[..., :2].permute(2, 0, 1)
    p1 = vector[..., 2:4].permute(2, 0, 1)
    del vector

    # 1. Calculate lengths and line directions
    direction_vector = p1 - p0
    del p1
    length = torch.norm(direction_vector, dim=0)
    direction_vector = direction_vector / (length + division_epsilon)

    # 2. Calculate coordinates
    raster_grid_height, raster_grid_width = points.shape[1:]
    points_n = raster_grid_height * raster_grid_width
    points = points.reshape(2, 1, 1, points_n)

    p0 = p0.reshape(2, batch_size, curves_n, 1)
    direction_vector = direction_vector.reshape(2, batch_size, curves_n, 1)
    canonical_y = torch.sum((points - p0) * direction_vector, dim=0)

    projections = canonical_y * direction_vector + p0
    del p0, direction_vector

    canonical_x_abs = torch.norm(points - projections, dim=0)
    del projections, points

    canonical_y = canonical_y.reshape(batch_size, curves_n, raster_grid_height, raster_grid_width)
    canonical_x_abs = canonical_x_abs.reshape(batch_size, curves_n, raster_grid_height, raster_grid_width)
    return canonical_x_abs, canonical_y, length
