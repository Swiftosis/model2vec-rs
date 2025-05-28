use approx::assert_relative_eq;
use model2vec::Model2Vec;

fn encode_with_model(path: &str) -> Vec<f32> {
    // Helper function to load the model and encode "hello world"
    let model = Model2Vec::from_pretrained(
        path,
        None,
        None,
    ).expect(
        &format!("Failed to load model at {path}")
    );

    let out = model
        .encode(["hello world"])
        .expect("embedding failed");

    assert_eq!(out.nrows(), 1, "output does not have exactly 1 element");
    assert_eq!(out.ncols(), model.embedding_dim(), "embedding dimension mismatch");

    out.row(0).to_vec()
}

#[test]
fn quantized_models_match_float32() {
    // Compare quantized models against the float32 model
    let base = "tests/fixtures/test-model-float32";
    let ref_emb = encode_with_model(base);

    for quant in &["float16", "int8"] {
        let path = format!("tests/fixtures/test-model-{quant}");
        let emb = encode_with_model(&path);

        assert_eq!(emb.len(), ref_emb.len());

        for (a, b) in ref_emb.iter().zip(emb.iter()) {
            assert_relative_eq!(a, b, max_relative = 1e-1);
        }
    }
}
