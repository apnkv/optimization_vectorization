import argparse
import glob
import json
import os
import pickle
from collections import deque, namedtuple
from concurrent.futures import ThreadPoolExecutor, wait

from vecopt.inference import make_intermediate_output_aligner

IntermediateSample = namedtuple('IntermediateSample', ['worker_idx', 'sample', 'filename'])
AlignedSample = namedtuple('AlignedSample', ['sample', 'filename'])


class Worker:
    def __init__(self, data, gpu_idx, config):
        self.gpu_idx = gpu_idx
        self.data = data
        self.aligner = make_intermediate_output_aligner(f'cuda:{gpu_idx}', config)

    def __call__(self):
        results = []

        while self.data:
            try:
                intermediate_sample = self.data.popleft()
                filename = intermediate_sample.filename[intermediate_sample.filename.rfind('/') + 1:]
                print(filename)
                intermediate_sample.sample['patches_vector'] = self.aligner(intermediate_sample.sample)
                results.append(AlignedSample(intermediate_sample.sample, filename))
            except IndexError:
                pass

        return results


def align_data_folder(path, n_workers, config):
    # TODO: decouple loading from folder from making IntermediateSample's and aligning
    data = deque()

    worker_idx = -1
    for filename in glob.glob(os.path.join(path, '*')):
        path = os.path.join(path, filename)
        with open(path, 'rb') as handle:
            sample = pickle.load(handle)

        worker_idx = (worker_idx + 1) % n_workers
        data.append(IntermediateSample(worker_idx, sample, filename))

    workers = [Worker(data, gpu_idx, config) for gpu_idx in range(n_workers)]
    executor = ThreadPoolExecutor(max_workers=n_workers)

    jobs = []
    for worker in workers:
        jobs.append(executor.submit(worker))

    wait(jobs)
    results = []
    for job in jobs:
        results.extend(job.result())

    return results


def main(args):
    with open(args.config) as config_file:
        config = json.load(config_file)

    print(f'Will align with {args.n_workers} using config:')
    print(config)

    results = align_data_folder(args.input, args.n_workers, config)

    print('Finished. Writing files:')

    os.makedirs(args.output, exist_ok=True)
    for sample in results:
        filename = os.path.join(args.output, sample.filename)
        print(filename)
        with open(filename, 'wb') as file:
            pickle.dump(sample, file, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--n-workers', type=int, required=False, default=1)

    args = parser.parse_args()
    main(args)
