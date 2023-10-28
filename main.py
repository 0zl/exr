import argparse
from exllama import EXL


if __name__ == '__main__':
    parser = argparse.ArgumentParser('exr')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--max-total-token', type=int, default=4096)
    parser.add_argument('--cache-8bit', action=argparse.BooleanOptionalAction)
    parser.add_argument('--print-warmup', action=argparse.BooleanOptionalAction)

    gs = parser.parse_args()
    ex = EXL(gs)
