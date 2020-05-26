import argparse
import glob
import os
import pandas as pd

from vectran.data.graphics.graphics import VectorImage
from vectran.metrics.cvpr20.skeleton_metrics import cpch_distance, number_of_primitives
from vectran.metrics.cvpr20.iou import iou_raster_reference, iou_vector_reference


def main(args):
    filenames = glob.glob(os.path.join(args.input, '*.svg'))
    all_metrics = []
    for filename in filenames:
        base_filename = filename[:filename.rfind('.')]
        if base_filename.endswith('.pickle'):
            base_filename = base_filename[:base_filename.rfind('.')]
        base_filename = base_filename[base_filename.rfind('/') + 1:]  # TODO: use os.path
        print(base_filename)

        gt_filename = os.path.join(args.ground_truth, f'{base_filename}.svg')

        gt_vector_image = VectorImage.from_svg(gt_filename).with_filled_removed()
        output_vector_image = VectorImage.from_svg(filename).with_filled_removed()

        cpch_metrics = cpch_distance(output_vector_image, gt_vector_image)
        metrics = {
            'base_filename': base_filename,

            'iou_at_pred': iou_vector_reference(output_vector_image, gt_vector_image, width=None),
            'iou_at_mean': iou_vector_reference(output_vector_image, gt_vector_image, width='mean'),
            'iou_at_1': iou_vector_reference(output_vector_image, gt_vector_image, width=1),

            'n_primitives_gt': number_of_primitives(gt_vector_image),
            'n_primitives': number_of_primitives(output_vector_image),

            'chamfer_px_sq': cpch_metrics['Chamfer distance in pixels squared'],
            'mmmd_px': cpch_metrics['Mean mean minimal distance in pixels'],
            'hausdorff_px': cpch_metrics['Hausdorff distance in pixels'],
        }
        print(metrics)
        all_metrics.append(metrics)

    pd.DataFrame(all_metrics).to_csv(args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-t', '--ground-truth', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)

    args = parser.parse_args()
    main(args)
