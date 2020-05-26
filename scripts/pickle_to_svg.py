import argparse
import glob
import os

from vectran.util.evaluation_utils import vector_image_from_patches

from vecopt.inference.inference import load_intermediate_result


def image_from_patches(sample):
    return vector_image_from_patches(
           primitives=sample['patches_vector'],
           patch_offsets=sample['patches_offsets'],
           image_size=sample['cleaned_image_shape'],
           control_points_n=4,
           patch_size=[64, 64],
           pixel_center_coodinates_are_integer=False,
           min_width=.3,
           min_confidence=.5,
           min_length=1.7
    )


def main(args):
    if os.path.isfile(args.input):
        filenames = [args.input]
    else:
        filenames = glob.glob(os.path.join(args.input, '*.pickle'))

    samples = []
    for filename in filenames:
        sample = load_intermediate_result(filename)
        samples.append(sample)

    os.makedirs(args.output, exist_ok=True)
    for i, sample in enumerate(samples):
        filename = filenames[i][filenames[i].rfind('/') + 1:]
        image = image_from_patches(sample)
        image.save(os.path.join(args.output, filename + '.svg'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)

    args = parser.parse_args()
    main(args)
