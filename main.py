import argparse
from exllama import EXL


if __name__ == '__main__':
    parser = argparse.ArgumentParser('exr')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--max-total-token', type=int, default=4096)
    parser.add_argument('--cache-8bit', action='store_true', default=False)
    parser.add_argument('--print-warmup', action='store_true', default=False)

    gs = parser.parse_args()
    ex = EXL(gs)
