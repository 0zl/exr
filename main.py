import argparse
from exllama import EXL
from redis_exr import RDSClient

if __name__ == '__main__':
    parser = argparse.ArgumentParser('exr')
    
    # exr arguments
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--max-total-token', type=int, default=4096)
    parser.add_argument('--cache-8bit', action='store_true', default=False)
    parser.add_argument('--print-warmup', action='store_true', default=False)

    # redis arguments
    parser.add_argument('--rd-h', type=str, required=True)
    parser.add_argument('--rd-p', type=int, required=True)
    parser.add_argument('--rd-s', type=str, required=True)
    parser.add_argument('--rd-id', type=str, required=True)
    parser.add_argument('--rd-m', type=str, required=True)
    parser.add_argument('--rd-g', type=str, required=True)

    gs = parser.parse_args()
    ex = EXL(gs)
    rd = RDSClient(gs, ex)

    print('exr - nyan~')
    rd.launch()
