"""
Smoke test: verify HF repo downloads work and models load correctly.

Run: python test_hf_smoke.py
"""

import sys
import json
import tempfile
from pathlib import Path

def test_hf_download():
    """Test 1: Download all files from HF repo."""
    print("=" * 60)
    print("TEST 1: Download from HuggingFace")
    print("=" * 60)

    from huggingface_hub import hf_hub_download

    repo_id = "VirialyD/tcell-classifier"
    expected_files = [
        "m1_h512.pt", "m2_h512_s2.pt", "m4_8heads.pt",
        "m6_highdrop.pt", "m7_lr3e4.pt",
        "results.json", "label_encoder.pkl", "vj_encoder.pkl",
        "gene_list_3000.txt", "gene_scaling.npz",
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        for fname in expected_files:
            print(f"  Downloading {fname}...", end=" ")
            path = hf_hub_download(
                repo_id=repo_id,
                filename=fname,
                local_dir=tmpdir,
                local_dir_use_symlinks=False,
            )
            size_mb = Path(path).stat().st_size / 1e6
            print(f"OK ({size_mb:.1f} MB)")

    print("PASS: All files downloaded.\n")


def test_results_json():
    """Test 2: Validate results.json structure and top-5 selection."""
    print("=" * 60)
    print("TEST 2: Validate results.json")
    print("=" * 60)

    from huggingface_hub import hf_hub_download

    with tempfile.TemporaryDirectory() as tmpdir:
        path = hf_hub_download(
            repo_id="VirialyD/tcell-classifier",
            filename="results.json",
            local_dir=tmpdir,
            local_dir_use_symlinks=False,
        )
        with open(path) as f:
            results = json.load(f)

    assert "individual" in results, "Missing 'individual' key"
    assert "final_accuracy" in results, "Missing 'final_accuracy' key"

    ranked = sorted(results["individual"], key=lambda x: x["acc"], reverse=True)
    top5_names = [m["name"] for m in ranked[:5]]
    print(f"  Top-5 by accuracy: {top5_names}")
    print(f"  Ensemble accuracy: {results['final_accuracy']:.1%}")
    print(f"  Ensemble F1:       {results['final_f1']:.4f}")

    expected_top5 = {"m7_lr3e4", "m2_h512_s2", "m6_highdrop", "m4_8heads", "m1_h512"}
    actual_top5 = set(top5_names)
    assert actual_top5 == expected_top5, f"Top-5 mismatch: {actual_top5} != {expected_top5}"

    print("PASS: results.json valid, top-5 matches expected.\n")


def test_model_loading():
    """Test 3: Load all 5 models and verify architecture."""
    print("=" * 60)
    print("TEST 3: Load models")
    print("=" * 60)

    import torch
    from huggingface_hub import snapshot_download

    # Add project to path
    sys.path.insert(0, str(Path(__file__).parent))
    from src.models import FullGenesVJClassifier
    from src.inference.predict import load_ensemble

    with tempfile.TemporaryDirectory() as tmpdir:
        print("  Downloading all weights...")
        model_dir = snapshot_download(
            "VirialyD/tcell-classifier", local_dir=tmpdir
        )

        device = torch.device("cpu")
        models = load_ensemble(model_dir, device, top_k=5)

    assert len(models) == 5, f"Expected 5 models, got {len(models)}"

    for i, m in enumerate(models):
        n_params = sum(p.numel() for p in m.parameters())
        print(f"  Model {i+1}: {n_params:,} params")

    print("PASS: All 5 models loaded successfully.\n")


def test_inference():
    """Test 4: Run inference on random data."""
    print("=" * 60)
    print("TEST 4: Inference on random data")
    print("=" * 60)

    import torch
    import numpy as np
    from huggingface_hub import snapshot_download

    sys.path.insert(0, str(Path(__file__).parent))
    from src.models import FullGenesVJClassifier
    from src.inference.predict import load_ensemble, ensemble_predict
    from src.data.dataset import InferenceDataset

    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = snapshot_download(
            "VirialyD/tcell-classifier", local_dir=tmpdir
        )
        models = load_ensemble(model_dir, device, top_k=5)

    # Fake data: 16 cells
    n = 16
    gex = np.random.randn(n, 3000).astype(np.float32)
    tcr_a = np.random.randn(n, 768).astype(np.float32)
    tcr_b = np.random.randn(n, 768).astype(np.float32)
    vj = np.zeros((n, 161), dtype=np.float32)

    dataset = InferenceDataset(gex, tcr_a, tcr_b, vj)
    predictions, probabilities, agreement = ensemble_predict(
        models, dataset, device, batch_size=8
    )

    assert predictions.shape == (n,), f"Wrong predictions shape: {predictions.shape}"
    assert probabilities.shape == (n, 7), f"Wrong probs shape: {probabilities.shape}"
    assert agreement.shape == (n,), f"Wrong agreement shape: {agreement.shape}"
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5), "Probs don't sum to 1"

    print(f"  Predictions: {predictions}")
    print(f"  Agreement:   {agreement}")
    print(f"  Prob sums:   {probabilities.sum(axis=1)}")
    print("PASS: Inference works correctly.\n")


if __name__ == "__main__":
    tests = [test_hf_download, test_results_json, test_model_loading, test_inference]
    passed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {e}\n")

    print("=" * 60)
    print(f"Results: {passed}/{len(tests)} tests passed")
    print("=" * 60)

    sys.exit(0 if passed == len(tests) else 1)
