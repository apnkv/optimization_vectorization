import numpy as np

from vectran.data.syndata.datasets import SyntheticHandcraftedDataset
from vectran.data.syndata.utils import renormalize as _normalize_probas
from vectran.data import graphics_primitives


topologies_with_probas = _normalize_probas({
    'l': 1,
    'l-beam': 1,
    'l-outer': 1,
    't': 1,
    't-beam': 1,
    't-outer': 1,
    'x': 1,
    'x-beam': 1,
    'x-outer': 1,
})

strokes_probas = _normalize_probas({1: .45, 2: .45, 3: .1})

directions_probas = {}
for angle in np.linspace(0, np.pi, 13):
    if angle in [0, np.pi / 2]:                 # - horizontal, | vertical
        directions_probas[angle] = .30
    elif angle in [np.pi / 4, np.pi * 3 / 4]:   # /, \
        directions_probas[angle] = .15
    elif angle < np.pi:
        directions_probas[angle] = .1 / 8

directions_probas = _normalize_probas(directions_probas)

## def get_offset_angles():
##     rotations = [0, 90, 180, 270]
##     max_rot_deviation = 10
##     rotations += [base_rot + dev_rot * dev_sign for dev_rot in range(1, max_rot_deviation + 1, 1) for base_rot in (0, 90, 180, 270) for dev_sign in (1, -1)]
##     return rotations
def get_offset_angles():
    return range(0, 360, 4)
offset_directions_probas = _normalize_probas({np.deg2rad(angle): 1 for angle in get_offset_angles()})

# plot distribution of directions
directions, probas = zip(*directions_probas.items())

padding_factor = 2

patch_parameters = {
    'patch_width': 64,
    'patch_height': 64,
    'max_lines': 10,
    'max_curves': 0
}

syn_parameters = {
    'samples_n': 1000000,
    'border': 8,
    'min_directions': 1, 'max_directions': 2,
    'min_primitives_gap': 2, 'max_primitives_gap': 10,
    'min_stroke_width': 1, 'max_stroke_width': 7,
    'min_stroke_length': max(patch_parameters['patch_width'], patch_parameters['patch_height']) * .90 * padding_factor,
    'max_stroke_length': np.sqrt(patch_parameters['patch_width']**2 + patch_parameters['patch_height']**2) * padding_factor,
    'primitives_endpoint_noise_sigma': 0.5,
    'primitives_direction_noise_sigma': np.pi / 270.,
    'directions_probas': directions_probas,
    'offset_directions_probas': offset_directions_probas,
    'strokes_probas': strokes_probas,
}

random_seed = 78

# make primitive_types and max_primitives dicts based on the parameters
primitive_types = []
max_primitives = {}
if patch_parameters['max_lines'] > 0:
    primitive_types.append(graphics_primitives.PrimitiveType.PT_LINE)
    max_primitives[graphics_primitives.PrimitiveType.PT_LINE] = patch_parameters['max_lines']
if patch_parameters['max_curves'] > 0:
    primitive_types.append(graphics_primitives.PrimitiveType.PT_BEZIER)
    max_primitives[graphics_primitives.PrimitiveType.PT_BEZIER] = patch_parameters['max_curves']


# initialize syndataset
np.random.seed(random_seed)
syn_dataset = SyntheticHandcraftedDataset(
    patch_size=(patch_parameters['patch_width'], patch_parameters['patch_width']), normalize_image=True,
    primitive_types=primitive_types, max_primitives=max_primitives, size=syn_parameters['samples_n'],
    topologies_with_probas=topologies_with_probas, **syn_parameters)