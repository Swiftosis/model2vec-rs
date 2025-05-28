//! Model2Vec loading and inference

use std::fs::{self, File};
use std::iter::zip;
use std::path::Path;
use anyhow::{Context, Result, anyhow, bail};
use ndarray::Array2;
use half::f16;
use serde_json::Value;
use safetensors::{SafeTensors, tensor::Dtype};
use tokenizers::Tokenizer;


/// Static embedding model for Model2Vec
pub struct Model2Vec {
    tokenizer: Tokenizer,
    embeddings: Array2<f32>,
    normalize: bool,
    median_token_length: usize,
    unk_token_id: Option<usize>,
}

impl Model2Vec {
    /// Load a Model2Vec model from a local directory.
    pub fn from_pretrained<P>(
        base_path: P,
        normalize: Option<bool>,
        subdir: Option<&str>,
    ) -> Result<Self>
    where
        P: AsRef<Path>
    {
        // Locate tokenizer.json, model.safetensors, config.json
        let base_path = base_path.as_ref();
        let directory = subdir.map(|s| base_path.join(s)).unwrap_or_else(|| base_path.to_path_buf());
        let tok_path = directory.join("tokenizer.json");
        let mdl_path = directory.join("model.safetensors");
        let cfg_path = directory.join("config.json");

        // Load the tokenizer
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow!("failed to load tokenizer: {e}"))?;

        // Median-token-length hack for pre-truncation
        let mut lens: Vec<usize> = tokenizer
            .get_vocab(false)
            .keys()
            .map(String::len)
            .collect();
        lens.sort_unstable();
        let median_token_length = lens.get(lens.len() / 2).copied().unwrap_or(1);

        // Read normalize default from config.json
        let cfg_file = File::open(&cfg_path).context("failed to read config.json")?;
        let cfg: Value = serde_json::from_reader(&cfg_file).context("failed to parse config.json")?;
        let cfg_norm = cfg.get("normalize").and_then(Value::as_bool).unwrap_or(true);
        let normalize = normalize.unwrap_or(cfg_norm);

        // Serialize the tokenizer to JSON, then parse it and get the unk_token
        let spec_json = tokenizer
            .to_string(false)
            .map_err(|e| anyhow!("failed to serialize tokenizer to JSON: {e}"))?;
        let spec: Value = serde_json::from_str(&spec_json)
            .context("failed to parse tokenizer JSON spec")?;
        let unk_token = spec
            .get("model")
            .and_then(|m| m.get("unk_token"))
            .and_then(Value::as_str)
            .unwrap_or("<unk>");
        let unk_token_id = tokenizer
            .token_to_id(unk_token)
            .ok_or_else(|| anyhow!("tokenizer JSON declared unk_token=\"{unk_token}\" but it’s not in the vocab"))?;
        let unk_token_id = Some(unk_token_id as usize);

        // Load the safetensors
        let model_bytes = fs::read(&mdl_path).context("failed to read safetensors file")?;
        let safet  = SafeTensors::deserialize(&model_bytes).context("failed to parse safetensors data")?;
        let tensor = safet
            .tensor("embeddings")
            .or_else(|_| safet.tensor("0"))
            .context("no 'embeddings' tensor found")?;
        let &[rows, cols] = tensor.shape().try_into().context("embedding tensor is not a 2D matrix")?;
        let raw   = tensor.data();
        let dtype = tensor.dtype();

        // Decode into f32
        let floats: Vec<f32> = match dtype {
            Dtype::F32 => {
                raw.chunks_exact(4)
                    .map(|bs| f32::from_le_bytes(bs.try_into().unwrap()))
                    .collect()
            }
            Dtype::F16 => {
                raw.chunks_exact(2)
                    .map(|bs| f16::from_le_bytes(bs.try_into().unwrap()).to_f32())
                    .collect()
            }
            Dtype::I8  => raw.iter().map(|&b| f32::from(b as i8)).collect(),
            other      => bail!("unsupported scalar dtype in tensor: {:?}", other),
        };

        let embeddings = Array2::from_shape_vec((rows, cols), floats)
            .context("failed to build embeddings array")?;

        Ok(Self {
            tokenizer,
            embeddings,
            normalize,
            median_token_length,
            unk_token_id,
        })
    }

    /// Char-level truncation to max_tokens * median_token_length
    fn truncate_str(s: &str, max_tokens: usize, median_len: usize) -> &str {
        let max_chars = max_tokens.saturating_mul(median_len);

        match s.char_indices().nth(max_chars) {
            Some((byte_idx, _)) => &s[..byte_idx],
            None => s,
        }
    }

    /// Encode texts into embeddings.
    ///
    /// # Arguments
    /// * `sentences` - the list of sentences to encode.
    /// * `max_length` - max tokens per text.
    /// * `batch_size` - number of texts per batch.
    pub fn encode_with_args<A, S>(
        &self,
        sentences: A,
        max_length: Option<usize>,
        batch_size: usize,
    ) -> Result<Array2<f32>>
    where
        A: AsRef<[S]>,
        S: AsRef<str>,
    {
        let sentences = sentences.as_ref();
        let mut embeddings = vec![0.0; sentences.len() * self.embedding_dim()];

        // Process in batches
        let batch_iter = zip(
            sentences.chunks(batch_size),
            embeddings.chunks_mut(batch_size * self.embedding_dim()),
        );

        for (batch, emb_batch) in batch_iter {
            let truncated: Vec<&str> = if let Some(max_tok) = max_length {
                // Truncate each sentence to max_length * median_token_length chars
                batch
                    .iter()
                    .map(|text| {
                        Self::truncate_str(text.as_ref(), max_tok, self.median_token_length)
                    })
                    .collect()
            } else {
                batch.iter().map(S::as_ref).collect()
            };

            let means = emb_batch.chunks_exact_mut(self.embedding_dim());

            assert_eq!(batch.len(), means.len());
            assert_eq!(truncated.len(), means.len());

            // Tokenize the batch
            let encodings = self
                .tokenizer
                .encode_batch_fast(
                    truncated,
                    /* add_special_tokens = */ false,
                )
                .map_err(|e| anyhow!("tokenization failed: {e}"))?;

            // Pool each token-ID list into a single mean vector
            assert_eq!(encodings.len(), means.len());

            for (encoding, out_mean) in zip(encodings, means) {
                let token_ids = encoding
                    .get_ids()
                    .iter()
                    .copied()
                    .map(|id| id as usize)
                    .filter(|&id| {
                        // this inequality comparison is always true when `self.unk_token_id == None`
                        self.unk_token_id != Some(id)
                    })
                    .take(max_length.unwrap_or(usize::MAX));

                self.pool_ids(token_ids, out_mean);
            }
        }

        Array2::from_shape_vec(
            (sentences.len(), self.embedding_dim()),
            embeddings,
        ).context(
            "embedding shape (input/output count or dimensionality) mismatch"
        )
    }

    /// Default encode: `max_length=512`, `batch_size=1024`
    pub fn encode<A, S>(&self, sentences: A) -> Result<Array2<f32>>
    where
        A: AsRef<[S]>,
        S: AsRef<str>,
    {
        self.encode_with_args(sentences, Some(512), 1024)
    }

    /// Mean-pool a single token-ID list into a vector.
    /// If `self.normalize` is true, normalize the result to unit length.
    fn pool_ids(&self, ids: impl IntoIterator<Item = usize>, mean: &mut [f32]) {
        assert_eq!(mean.len(), self.embedding_dim());

        // make an all-0 vector without reallocating
        mean.fill(0.0);

        let mut cnt = 0;

        for id in ids {
            let row = self.embeddings.row(id);

            for (x, &v) in zip(&mut *mean, row) {
                *x += v;
            }

            cnt += 1;
        }

        // If we need to normalize to unit length, do not perform an additional pass
        // over the vector elements: normalizing the _sum_ to unit length is exactly
        // the same as first dividing by the count, then re-normalizing the _mean_.
        let denominator = if self.normalize {
            mean.iter().map(|&v| v * v).sum::<f32>().sqrt().max(1e-12)
        } else {
            cnt.max(1) as f32
        };

        for x in mean {
            *x /= denominator;
        }
    }

    /// The dimensionality of the embedding space.
    pub fn embedding_dim(&self) -> usize {
        self.embeddings.ncols()
    }
}
