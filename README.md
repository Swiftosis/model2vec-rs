![Model2Vec Rust logo](https://github.com/Swiftosis/model2vec-rs/raw/refs/heads/main/assets/images/model2vec_rs_logo.png)

# Fast State-of-the-Art Static Embeddings in Rust

[![Crates.io](https://img.shields.io/crates/d/model2vec.svg)](https://crates.io/crates/model2vec)
[![Docs.rs](https://docs.rs/model2vec/badge.svg)](https://docs.rs/model2vec)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/Swiftosis/model2vec-rs/blob/main/LICENSE)

`model2vec-rs` is a Rust crate providing an efficient implementation for inference with [Model2Vec](https://huggingface.co/blog/Pringled/model2vec) static embedding models. Model2Vec is a technique for creating compact and fast static embedding models from sentence transformers, achieving significant reductions in model size and inference speed. This Rust crate is optimized for performance, making it suitable for applications requiring fast embedding generation.

## Quickstart

1. Add `model2vec` as a dependency:

```bash
cargo add model2vec
```

2. Load a model and generate embeddings:

```rust
use anyhow::Result;
use model2vec::model::Model2Vec;

fn main() -> Result<()> {
    // Load a model from a local directory
    // Arguments: (path, normalize_embeddings, subfolder_in_repo)
    let model = Model2Vec::from_pretrained(
        "tests/fixtures/test-model-float32", // Local path to model directory
        None, // Optional: bool to override model's default normalization. `None` uses model's config.
        None, // Optional: subfolder if model files are not at the root of the repo/path
    )?;

    // Any type that implements `AsRef<[S]>` works, where `S: AsRef<str>`
    // This includes Vec<String>, &[&str], etc.
    let sentences = [
        "Hello world",
        "Rust is awesome",
    ];

    // Generate embeddings using default parameters
    // (Default max_length: Some(512), Default batch_size: 1024)
    let embeddings = model.encode(&sentences)?;

    // `embeddings` is an ndarray::Array2<f32>, where each row is an embeddings
    // corresponding to a sentence, and each column is a different dimension.
    assert_eq!(embeddings.nrows(), sentences.len());
    println!("Generated {} embeddings.of {} dimensions", embeddings.nrows(), embeddings.ncols());

    // To generate embeddings with custom arguments:
    let custom_embeddings = model.encode_with_args(
        sentences,
        Some(256), // Optional: custom max token length for truncation
        512,       // Custom batch size for processing
    )?;
    assert_eq!(custom_embeddings.nrows(), sentences.len());
    println!("Generated {} custom embeddings of {} dimensions", custom_embeddings.nrows(), custom_embeddings.ncols());

    Ok(())
}
```

## Features

*   **Fast Inference:** Optimized Rust implementation for fast embedding generation.
*   **Model Formats:** Supports models with f32, f16, and i8 weight types stored in `safetensors` files.
*   **Batch Processing:** Encodes multiple sentences in batches.
*   **Configurable Encoding:** Allows customization of maximum sequence length and batch size during encoding.

## What is Model2Vec?

Model2Vec is a technique to distill large sentence transformer models into highly efficient static embedding models. This process significantly reduces model size and computational requirements for inference. For a detailed understanding of how Model2Vec works, including the distillation process and model training, please refer to the [main Model2Vec Python repository](https://github.com/MinishLab/model2vec) and its [documentation](https://github.com/MinishLab/model2vec/blob/main/docs/what_is_model2vec.md).

This `model2vec` crate provides a Rust-based engine specifically for **inference** using these Model2Vec models.

## Models

A variety of pre-trained Model2Vec models are available on the [HuggingFace Hub (MinishLab collection)](https://huggingface.co/collections/minishlab/model2vec-base-models-66fd9dd9b7c3b3c0f25ca90e). These can be loaded by `model2vec-rs` using their Hugging Face model ID or by providing a local path to the model files.

| Model                                                                 | Language    | Distilled From (Original Sentence Transformer)                  | Params  | Task      |
|-----------------------------------------------------------------------|------------|-----------------------------------------------------------------|---------|-----------|
| [potion-base-32M](https://huggingface.co/minishlab/potion-base-32M)   | English    | [bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | 32.3M   | General   |
| [potion-multilingual-128M](https://huggingface.co/minishlab/potion-multilingual-128M) | Multilingual | [bge-m3](https://huggingface.co/BAAI/bge-m3)      | 128M    | General   |
| [potion-retrieval-32M](https://huggingface.co/minishlab/potion-retrieval-32M) | English    | [bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | 32.3M   | Retrieval |
| [potion-base-8M](https://huggingface.co/minishlab/potion-base-8M)     | English    | [bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | 7.5M    | General   |
| [potion-base-4M](https://huggingface.co/minishlab/potion-base-4M)     | English    | [bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | 3.7M    | General   |
| [potion-base-2M](https://huggingface.co/minishlab/potion-base-2M)     | English    | [bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | 1.8M    | General   |


## Performance

We compared the performance of the Rust implementation with the Python version of Model2Vec. The benchmark was run single-threaded on a CPU.

| Implementation | Throughput                                         |
| -------------- | -------------------------------------------------- |
| **Rust**       | 8000 samples/second |
| **Python**     | 4650 samples/second |

The Rust version is roughly **1.7×** faster than the Python version.

## License

MIT

## Citing Model2Vec

If you use the Model2Vec methodology or models in your research or work, please cite the original Model2Vec project:
```bibtex
@article{minishlab2024model2vec,
  author = {Tulkens, Stephan and {van Dongen}, Thomas},
  title = {Model2Vec: Fast State-of-the-Art Static Embeddings},
  year = {2024},
  url = {https://github.com/MinishLab/model2vec}
}
```
