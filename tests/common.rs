use model2vec::Model2Vec;

/// Load the small float32 test model from fixtures
pub fn load_test_model() -> Model2Vec {
    Model2Vec::from_pretrained(
        "tests/fixtures/test-model-float32",
        None,   // normalize
        None,   // subdirectory
    ).expect(
        "Failed to load test model"
    )
}
