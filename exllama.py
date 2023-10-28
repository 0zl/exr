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
        #self.config.max_seq_len = gs.max_total_token
        #self.config.max_input_len = gs.max_total_token
        self.config.prepare()

        self.model = ExLlamaV2(self.config)
        self.cache = ExLlamaV2Cache(self.model)
        self.model.load_autosplit(self.cache)

        self.tokenizer = ExLlamaV2Tokenizer(self.config)
        self.generator = ExLlamaV2BaseGenerator(self.model, self.cache, self.tokenizer)
        self.streaming = ExLlamaV2StreamingGenerator(self.model, self.cache, self.tokenizer)
