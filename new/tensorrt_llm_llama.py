import tensorrt_llm
import tensorrt_llm.profiler as profiler
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization import QuantMode
import json
import os
import torch
from transformers import AutoTokenizer


def get_engine_name(model, dtype, tp_size, pp_size, rank):
    if pp_size == 1:
        return '{}_{}_tp{}_rank{}.engine'.format(model, dtype, tp_size, rank)
    return '{}_{}_tp{}_pp{}_rank{}.engine'.format(model, dtype, tp_size,
                                                pp_size, rank)


def get_decoder(engine_dir):
    config_path = os.path.join(engine_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    dtype = config['builder_config']['precision']
    tp_size = config['builder_config']['tensor_parallel']
    pp_size = config['builder_config']['pipeline_parallel']
    world_size = tp_size * pp_size

    assert world_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'

    num_heads = config['builder_config']['num_heads'] // tp_size
    hidden_size = config['builder_config']['hidden_size'] // tp_size
    vocab_size = config['builder_config']['vocab_size']
    num_layers = config['builder_config']['num_layers']
    use_gpt_attention_plugin = bool(
        config['plugin_config']['gpt_attention_plugin'])
    remove_input_padding = config['plugin_config']['remove_input_padding']
    num_kv_heads = config['builder_config'].get('num_kv_heads', num_heads)
    paged_kv_cache = config['plugin_config']['paged_kv_cache']
    tokens_per_block = config['plugin_config']['tokens_per_block']
    use_custom_all_reduce = config['plugin_config'].get('use_custom_all_reduce',
                                                        False)

    quant_mode = QuantMode(config['builder_config']['quant_mode'])
    if config['builder_config'].get('multi_query_mode', False):
        tensorrt_llm.logger.warning(
            "`multi_query_mode` config is deprecated. Please rebuild the engine."
        )
        num_kv_heads = 1
    num_kv_heads = (num_kv_heads + tp_size - 1) // tp_size

    model_config = tensorrt_llm.runtime.ModelConfig(
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        hidden_size=hidden_size,
        paged_kv_cache=paged_kv_cache,
        tokens_per_block=tokens_per_block,
        gpt_attention_plugin=use_gpt_attention_plugin,
        remove_input_padding=remove_input_padding,
        use_custom_all_reduce=use_custom_all_reduce,
        dtype=dtype,
        quant_mode=quant_mode)

    runtime_rank = tensorrt_llm.mpi_rank()
    runtime_mapping = tensorrt_llm.Mapping(world_size,
                                        runtime_rank,
                                        tp_size=tp_size,
                                        pp_size=pp_size)
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

    engine_name = get_engine_name('llama', dtype, tp_size, pp_size,
                                runtime_rank)
    serialize_path = os.path.join(engine_dir, engine_name)

    tensorrt_llm.logger.set_level('info')

    profiler.start('load tensorrt_llm engine')
    with open(serialize_path, 'rb') as f:
        engine_buffer = f.read()
    decoder = tensorrt_llm.runtime.GenerationSession(model_config,
                                                    engine_buffer,
                                                    runtime_mapping)
    profiler.stop('load tensorrt_llm engine')
    tensorrt_llm.logger.info(
        f'Load engine takes: {profiler.elapsed_time_in_sec("load tensorrt_llm engine")} sec'
    )
    return decoder


class TrtLLM:
    def __init__(self, engine_dir, tokenizer_dir):
        self.decoder = get_decoder(engine_dir=engine_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    def generate(self, prompts):

        line_encoded = []
        input_id = self.tokenizer.encode(prompts,
                                         return_tensors='pt').type(torch.int32)
        line_encoded.append(input_id)
        input_lengths = []
        input_lengths.append(input_id.shape[-1])
        max_length = max(input_lengths)

        pad_id = self.tokenizer.encode(self.tokenizer.pad_token, add_special_tokens=False)[0]
        end_id = self.tokenizer.encode(self.tokenizer.eos_token, add_special_tokens=False)[0]
        print(self.decoder.remove_input_padding)
        return prompts
