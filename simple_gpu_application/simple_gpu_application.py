#!/usr/bin/env python3
""" A simple application which shows the general flow of data through the GPU """
import argparse
import random

import torch

# Names had to be  letters, LOL.
RANDOM_NAMES = ["George", "Dennis", "Sowmya", "Lauren", "Markus", "Yongme", "Smokey"]

def get_args():
    parser = argparse.ArgumentParser(description="Simple GPU application")
    parser.add_argument("-r", "--repeats", type=int, default=5,
                        help = "Number of times to repeat string (to scale application)")
    return parser.parse_args()

def generate_sample_file(outfile: str, repeats: int = 5) -> None:
    """ Generates our sample file for consumption. """
    dummy_list = [f"Pleasure to be working with you, {random.choice(RANDOM_NAMES)}!\n" for _ in range(repeats)]
    with open(outfile, 'w') as f:
        for i in range(repeats):
            f.write(dummy_list[i])
    print(f"Generated text with {len(dummy_list)} sentences")

def encode_sample_data(infile: str) -> torch.Tensor:
    parsed = []
    with open(infile, 'r') as f:
        for line in f:
            parsed.append(line.strip())
    numbers = []
    print("Encoding data for GPU...")
    for line in parsed:
        nums = []
        for char in line:
            nums.append(ord(char))
        numbers.append(nums)
    return torch.tensor(numbers, dtype=torch.float16)

def main():
    args = get_args()
    sample_file = 'sample.txt'
    print("Generating sample data...")
    generate_sample_file(sample_file, repeats = args.repeats)
    tensor = encode_sample_data(sample_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Processing data with {device}.")
    gpu_data = tensor.to(device)

    print(f"Performing GPU mean reduction on input data on {tensor.shape[0]} rows and {tensor.shape[1]} elements per row")
    result = torch.mean(gpu_data, dim=1)

    print("Moving results back to cpu")
    result = result.cpu()
    print(f"First 5 results: {result[:5]}...")

    print("Writing binary tensor to disk...")
    torch.save(result, 'tensor.pt')



if __name__ == "__main__":
    main()
