import argparse


def main(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)

    args = parser.parse_args()

    main(args)