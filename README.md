# PyTorch Transformer

A simple clean-readable and shape-annotated implementation of [Attention is All You Need](https://arxiv.org/abs/1706.03762) in PyTorch. A sample onnx file can be found in `assets/transformer.onnx` for visualization purposes.

It was tested on synthetic data, try to use the attention plots to figure out the transformation used to create the data!

# Implementation Details

- Positional Embeddings not included, similar to `nn.Transformer` but you can find an implementation in `usage.ipynb`.
- Parallel `MultiHeadAttention` outperforms the for loop implementation significantly, as expected.
- Assumes `batch_first=True` input by default and cna't be changed.
- Uses `einsum` for attention computation rather than `bmm` for readability, this might impact performance.

# KV Cache Branch

<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/4c11adec-6eab-45c4-ba51-723f71eefd52" />
