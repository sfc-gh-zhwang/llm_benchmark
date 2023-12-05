import numpy as np
import tritonclient.grpc as grpcclient
from transformers import AutoTokenizer
from tritonclient.utils import InferenceServerException, np_to_triton_dtype
import multiprocessing


def _input(name: str, data: np.ndarray) -> grpcclient.InferInput:
    t = grpcclient.InferInput(name, data.shape, np_to_triton_dtype(data.dtype))
    t.set_data_from_numpy(data)
    return t

class TrtLLM:
    def __init__(self, engine_dir, tokenizer_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    def generate(self, prompts, max_new_tokens):
        batch_size = len(prompts)
        input_id = self.tokenizer(prompts,
                                  padding=False).input_ids
        input_lengths = []
        for i in input_id:
            input_lengths.append(len(i))
        with grpcclient.InferenceServerClient("localhost:8001", verbose=False) as client:
            with multiprocessing.Manager() as manager:
                def send(client, input_id, input_length, i, shared_list):
                    inputs = [
                        _input("input_ids", np.array(input_id, dtype=np.int32).reshape(1, -1)),
                        _input("input_lengths", np.array([input_length], dtype=np.int32).reshape(1, -1)),
                        _input("request_output_len", np.array([max_new_tokens], dtype=np.uint32).reshape(1, -1)),
                        _input("end_id", np.array([2], dtype=np.uint32).reshape(1, -1)),
                    ]
                    shared_list[i] = client.infer('tensorrt_llm', inputs).as_numpy('sequence_length').reshape(-1)[0]
                processes = []
                shared_list = manager.list([""] * batch_size)
                for i in range(batch_size):
                    process = multiprocessing.Process(target=send, args=(client, input_id[i], input_lengths[i], i, shared_list))
                    processes.append(process)
                    process.start()

                for process in processes:
                    process.join()
        for i in shared_list:
            print(i)
        return prompts
