import argparse
import glob
import os

from bs4 import BeautifulSoup


def patch_svg_inplace(filename):
    with open(filename, 'r') as file:
        soup = BeautifulSoup(file.read())

    svg_tag = soup.select_one('svg')
    if 'px' in svg_tag['viewbox'] and 'px' in svg_tag['width'] and 'px' in svg_tag['height']:
        return

    minx, miny, width, height = map(float, svg_tag['viewbox'].split())
    svg_tag['viewbox'] = f'{minx}px {miny}px {width}px {height}px'
    svg_tag['width'] = f'{width}px'
    svg_tag['height'] = f'{height}px'

    with open(filename, 'w') as file:
        file.write(soup.prettify())


def main(args):
    filenames = glob.glob(os.path.join(args.input, '*.svg'))
    for filename in filenames:
        patch_svg_inplace(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', type=str, required=True)

    args = parser.parse_args()
    main(args)
