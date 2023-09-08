# llm_benchmark

Run on A100(40GiB)x8, with tensor para = 8.

|             | batch_size: 32, input_len: 1, output_len: 2048 | batch_size: 24, input_len: 1024, output_len: 1024 |
| ----------- | ---------------------------------------------- | ------------------------------------------------- |
| Huggingface | 948.26 s                                       | 439.02 s                                          |
| Triton      | 70.19 s                                        | 38.97 s                                           |
| vLLM        | 133.70 s                                       | 76.12 s                                           |

