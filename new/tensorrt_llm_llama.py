import numpy as np
import tritonclient.grpc as grpcclient
from transformers import AutoTokenizer
from tritonclient.utils import InferenceServerException, np_to_triton_dtype
import threading


def _input(name: str, data: np.ndarray) -> grpcclient.InferInput:
    t = grpcclient.InferInput(name, data.shape, np_to_triton_dtype(data.dtype))
    t.set_data_from_numpy(data)
    return t

class TrtLLM:
    def __init__(self, engine_dir, tokenizer_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    def generate(self, prompts, max_new_tokens):
        batch_size = len(prompts)
        if len(prompts) == 1:
            prompts = prompts[0]
        input_id = self.tokenizer(prompts,
                                  padding=False).input_ids
        input_lengths = []
        for i in input_id:
            input_lengths.append(len(i))

        with grpcclient.InferenceServerClient("localhost:8001", verbose=False) as client:
            threads = []

            def send(client, model_name, input_id, input_length, max_new_tokens):
                inputs = [
                    _input("input_ids", np.array(input_id, dtype=np.uint32).reshape(1, -1)),
                    _input("input_lengths", np.array([input_length], dtype=np.int32).reshape(1, -1)),
                    _input("request_output_len", np.array([max_new_tokens], dtype=np.uint32).reshape(1, -1)),
                ]
                client.infer(model_name, inputs)
            # Create and start n threads
            for i in range(batch_size):
                thread = threading.Thread(target=send, args=(client, 'tensorrt_llm', input_id[i], input_lengths[i], max_new_tokens))
                threads.append(thread)
                thread.start()

            # Wait for all threads to finish
            for thread in threads:
                thread.join()
        output_beams_list = [
            self.tokenizer.batch_decode(output_ids[batch_idx, :,
                                                   input_lengths[batch_idx]:],
                                                   skip_special_tokens=True)
            for batch_idx in range(batch_size)
        ]
        for i in output_beams_list:
            print(i)
        return prompts
