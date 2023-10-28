import time, random
from huggingface_hub import snapshot_download

from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
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
    cache: ExLlamaV2Cache
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
        self.cache = ExLlamaV2Cache(self.model, lazy=True)
        self.model.load_autosplit(self.cache)

        self.tokenizer = ExLlamaV2Tokenizer(self.config)
        self.generator = ExLlamaV2BaseGenerator(self.model, self.cache, self.tokenizer)
        self.streaming = ExLlamaV2StreamingGenerator(self.model, self.cache, self.tokenizer)

        self.generator.warmup()
        self.exr_warmup(True)
    
    def exr_warmup(self, print_console):
        print('exr warmup...')
        
        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = 0.85
        settings.top_k = 50
        settings.top_p = 0.8
        settings.token_repetition_penalty = 1.15
        settings.disallow_tokens(self.tokenizer, [self.tokenizer.eos_token_id])

        prompt = "Our story begins in the Scottish town of Auchtermuchty, where once"
        max_new_tokens = 150

        print('exr_warmup 1 ...')
        time_begin_a = time.time()
        output_a = self.generator.generate_simple(prompt, settings, max_new_tokens, seed=random.randint(1, 1e7))
        time_end_a = time.time()
        time_total_a = time_end_a - time_begin_a

        print('exr_warmup 2 ...')
        time_begin_b = time.time()
        output_b = self.generator.generate_simple(prompt, settings, max_new_tokens, seed=random.randint(1, 1e7))
        time_end_b = time.time()
        time_total_b = time_end_b - time_begin_b
        
        if print_console:
            print(f'exr_warmup 1: {output_a}')
            print('')
            print(f'Response generated in {time_total_a:.2f} seconds, {max_new_tokens} tokens, {max_new_tokens / time_total_a:.2f} tokens/second')
            print('-------')
            print(f'exr_warmup 2: {output_b}')
            print('')
            print(f'Response generated in {time_total_b:.2f} seconds, {max_new_tokens} tokens, {max_new_tokens / time_total_b:.2f} tokens/second')
