import time
import random
import torch
from huggingface_hub import snapshot_download
from typing import List

import shared

from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Cache_8bit,
    ExLlamaV2Tokenizer,
)

from exllamav2.generator import (
    ExLlamaV2StreamingGenerator,
    ExLlamaV2BaseGenerator,
    ExLlamaV2Sampler
)


class EXL():
    config: ExLlamaV2Config
    model: ExLlamaV2
    cache: ExLlamaV2Cache | ExLlamaV2Cache_8bit
    tokenizer: ExLlamaV2Tokenizer
    generator: ExLlamaV2BaseGenerator
    streaming: ExLlamaV2StreamingGenerator

    def __init__(self, gs):
        model_path = snapshot_download(gs.model, local_dir='./model')
        print(f'model path: {model_path}')

        self.config = ExLlamaV2Config()
        self.config.model_dir = model_path
        self.config.max_seq_len = gs.max_total_token
        self.config.max_input_len = gs.max_total_token
        self.config.prepare()

        self.model = ExLlamaV2(self.config)

        if gs.cache_8bit:
            self.cache = ExLlamaV2Cache_8bit(
                self.model, lazy=not self.model.loaded)
            print('using 8bit cache')
        else:
            self.cache = ExLlamaV2Cache(self.model, lazy=not self.model.loaded)

        if not self.model.loaded:
            print('loading model...')
            self.model.load_autosplit(self.cache)

        self.tokenizer = ExLlamaV2Tokenizer(self.config)
        self.generator = ExLlamaV2BaseGenerator(
            self.model, self.cache, self.tokenizer)
        self.streaming = ExLlamaV2StreamingGenerator(
            self.model, self.cache, self.tokenizer)

        self.generator.warmup()
        self.streaming.warmup()

        self.exr_warmup(gs.print_warmup)
        shared.init_complete = True

    def exr_warmup(self, print_console):
        print('exr warmup...')

        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = 0.85
        settings.top_k = 50
        settings.top_p = 0.8
        settings.token_repetition_penalty = 1.15
        settings.disallow_tokens(self.tokenizer, [self.tokenizer.eos_token_id])

        prompt = "USER: Hey, what do you think about relationship between Human and Robot, lol.\nASSISTANT:"
        max_new_tokens = 150

        print('exr_warmup 1 ...')
        time_begin_a = time.time()
        output_a = self.generator.generate_simple(
            prompt, settings, max_new_tokens, seed=random.randint(1, int(1e7)))
        time_end_a = time.time()
        time_total_a = time_end_a - time_begin_a

        # print('exr_warmup 2 ...')
        # time_begin_b = time.time()
        # output_b = self.generator.generate_simple(prompt, settings, max_new_tokens, seed=random.randint(1, int(1e7)))
        # time_end_b = time.time()
        # time_total_b = time_end_b - time_begin_b

        if print_console:
            print(f'exr_warmup 1: {output_a}')
            print('')
            print(
                f'Response generated in {time_total_a:.2f} seconds, {max_new_tokens} tokens, {max_new_tokens / time_total_a:.2f} tokens/second')
            print('-------')
            # print(f'exr_warmup 2: {output_b}')
            # print('')
            # print(f'Response generated in {time_total_b:.2f} seconds, {max_new_tokens} tokens, {max_new_tokens / time_total_b:.2f} tokens/second')

    def encode(self, text: str):
        return self.tokenizer.encode(text)

    def prepare_stream(
            self,
            prompt: str,
            stop_condition: List[str],
            temperature: float,
            top_k: int,
            top_p: float,
            typical: int,
            repitition_penalty: float,
            max_response_tokens: int = 250
    ) -> tuple[torch.Tensor, ExLlamaV2Sampler.Settings, int]:
        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = temperature
        settings.top_k = top_k
        settings.top_p = top_p
        settings.typical = typical
        settings.token_repetition_penalty = repitition_penalty

        # stop condition
        self.streaming.set_stop_conditions(stop_condition)
        input_context = self.encode(prompt)

        return input_context, settings, max_response_tokens
