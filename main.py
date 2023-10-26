import argparse
from exllama import EXL


if __name__ == '__main__':
    parser = argparse.ArgumentParser('exr')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--max-total-token', type=int, default=4096)

    gs = parser.parse_args()
    ex = EXL(gs)
