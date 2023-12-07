import numpy as np
import tritonclient.grpc as grpcclient
from transformers import AutoTokenizer
from tritonclient.utils import InferenceServerException, np_to_triton_dtype
import multiprocessing
import time


def _input(name: str, data: np.ndarray) -> grpcclient.InferInput:
    t = grpcclient.InferInput(name, data.shape, np_to_triton_dtype(data.dtype))
    t.set_data_from_numpy(data)
    return t


class TrtLLM:
    def __init__(self, tokenizer_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    def generate(self, prompts, max_new_tokens):
        batch_size = len(prompts)
        input_id = self.tokenizer(prompts,
                                  padding=False).input_ids
        input_lengths = []
        output_lengths = []
        for i in input_id:
            input_lengths.append(len(i))
        with multiprocessing.Manager() as manager:
            def send(client, input_id, input_length, i, shared_list):
                inputs = [
                    _input("input_ids", np.array(input_id, dtype=np.int32).reshape(1, -1)),
                    _input("input_lengths", np.array([input_length], dtype=np.int32).reshape(1, -1)),
                    _input("request_output_len", np.array([max_new_tokens], dtype=np.uint32).reshape(1, -1)),
                    _input("end_id", np.array([2], dtype=np.uint32).reshape(1, -1)),
                ]
                with grpcclient.InferenceServerClient("localhost:8001", verbose=False) as client:
                    shared_list[i] = client.infer('tensorrt_llm', inputs).as_numpy('sequence_length')[0][0]
            processes = []
            shared_list = manager.list([""] * batch_size)
            start = time.time()
            for i in range(batch_size):
                process = multiprocessing.Process(target=send, args=(None, input_id[i], input_lengths[i], i, shared_list))
                processes.append(process)
                process.start()

            for process in processes:
                process.join()
            latency = time.time() - start
            for i in shared_list:
                output_lengths.append(i)
        return latency, input_lengths, output_lengths
