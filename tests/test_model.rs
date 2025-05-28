use std::fs;
use anyhow::{Result, Context, anyhow};
use common::load_test_model;
use approx::assert_relative_eq;
use model2vec::model::Model2Vec;

mod common;

#[test]
fn test_encode_matches_python_model2vec() -> Result<()> {
    // Load the test model
    let model = load_test_model();

    // Define the short and long text inputs
    let long_text = vec!["hello"; 1000].join(" ");
    let short_text = "hello world".to_string();
    let cases = vec![
        (
            "tests/fixtures/embeddings_short.json",
            vec![short_text],
        ),
        (
            "tests/fixtures/embeddings_long.json",
            vec![long_text],
        ),
    ];

    for (fixture_path, inputs) in cases {
        // Read and parse the Python‐generated embedding fixture
        let fixture = fs::read_to_string(fixture_path)
            .map_err(|_| anyhow!("fixture not found: {fixture_path}"))?;
        let expected: Vec<Vec<f32>> = serde_json::from_str(&fixture)
            .context("failed to parse fixture")?;

        // Encode with the Rust model
        let output = model.encode(&inputs)?;

        // Sanity checks
        assert_eq!(
            output.nrows(),
            expected.len(),
            "number of sentences mismatch for {}",
            fixture_path
        );
        assert_eq!(
            output.ncols(),
            expected[0].len(),
            "vector dimensionality mismatch for {}",
            fixture_path
        );

        // Element‐wise comparison
        for (o, e) in output.row(0).iter().zip(&expected[0]) {
            assert_relative_eq!(o, e, max_relative = 1e-5);
        }
    }

    Ok(())
}

/// Test that encoding an empty input slice yields an empty output
#[test]
fn test_encode_empty_input() -> Result<()> {
    use ndarray::Array2;

    let model = load_test_model();
    let embs: Array2<f32> = model.encode([] as [&str; 0])?;
    assert!(embs.is_empty(), "Expected no embeddings for empty input");

    Ok(())
}

/// Test that encoding a single empty sentence produces a zero vector
#[test]
fn test_encode_empty_sentence() -> Result<()> {
    let model = load_test_model();

    let embs = model.encode([""])?;
    assert_eq!(embs.nrows(), 1);

    let vec = embs.row(0);
    assert!(vec.iter().all(|&x| x == 0.0), "All entries should be zero");

    Ok(())
}

/// Test override of `normalize` flag in from_pretrained
#[test]
fn test_normalization_flag_override() -> Result<()> {
    // Load with normalize = true (default in config)
    let model_norm = Model2Vec::from_pretrained(
        "tests/fixtures/test-model-float32", None, None
    )?;
    let emb_norm = model_norm.encode(&["test sentence"])?;
    let norm_norm = emb_norm.iter().map(|&x| x*x).sum::<f32>().sqrt();
    assert_eq!(emb_norm.nrows(), 1);
    assert_eq!(emb_norm.ncols(), model_norm.embedding_dim());

    // Load with normalize = false override
    let model_no_norm = Model2Vec::from_pretrained(
        "tests/fixtures/test-model-float32", Some(false), None
    )?;
    let emb_no = model_no_norm.encode(&["test sentence".to_string()])?;
    let norm_no = emb_no.iter().map(|&x| x*x).sum::<f32>().sqrt();
    assert_eq!(emb_no.nrows(), 1);
    assert_eq!(emb_no.ncols(), model_no_norm.embedding_dim());

    // Normalized version should have unit length, override should give larger norm
    assert!((norm_norm - 1.0).abs() < 1e-5, "Normalized vector should have unit norm");
    assert!(norm_no > norm_norm + 0.1, "Without normalization override, norm should be larger");

    Ok(())
}
