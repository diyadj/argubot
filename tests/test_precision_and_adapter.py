import json

import pytest

from argubot.generate.generator import adapter_base_model, check_adapter_matches, torch_dtype_from_name
from argubot.generate.training import resolve_precision


# --- training precision ----------------------------------------------------------------------

def test_bf16_is_used_on_a_card_that_supports_it():
    assert resolve_precision({"precision": "bf16"}, cuda_available=True, bf16_supported=True) == "bf16"


def test_bf16_on_a_t4_stops_with_a_clear_error_instead_of_switching_to_fp16():
    with pytest.raises(RuntimeError, match="cannot do bf16"):
        resolve_precision({"precision": "bf16"}, cuda_available=True, bf16_supported=False)


def test_fp16_stays_fp16_on_any_gpu():
    assert resolve_precision({"precision": "fp16"}, cuda_available=True, bf16_supported=False) == "fp16"


def test_without_a_gpu_everything_runs_in_fp32():
    for p in ("bf16", "fp16", "fp32"):
        assert resolve_precision({"precision": p}, cuda_available=False, bf16_supported=False) == "fp32"


def test_old_configs_with_the_fp16_flag_still_work():
    assert resolve_precision({"fp16": True}, cuda_available=True, bf16_supported=True) == "fp16"
    assert resolve_precision({"fp16": False}, cuda_available=True, bf16_supported=True) == "fp32"


def test_precision_wins_over_the_old_flag_and_typos_are_caught():
    assert resolve_precision({"precision": "BF16", "fp16": True}, True, True) == "bf16"
    with pytest.raises(ValueError):
        resolve_precision({"precision": "bfloat"}, True, True)


# --- generator dtype -------------------------------------------------------------------------

def test_auto_and_none_both_mean_the_dtype_the_model_ships_in():
    assert torch_dtype_from_name(None) == "auto"
    assert torch_dtype_from_name("auto") == "auto"


def test_a_bad_dtype_name_is_an_error():
    with pytest.raises(ValueError):
        torch_dtype_from_name("int8")


def test_dtype_names_map_to_torch_dtypes():
    torch = pytest.importorskip("torch")
    assert torch_dtype_from_name("bfloat16") is torch.bfloat16
    assert torch_dtype_from_name("bf16") is torch.bfloat16
    assert torch_dtype_from_name("float32") is torch.float32


# --- adapter belongs to the base model -------------------------------------------------------

def _adapter(tmp_path, base):
    d = tmp_path / "adapter"
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base}))
    return str(d)


def test_matching_adapter_passes(tmp_path):
    d = _adapter(tmp_path, "Qwen/Qwen3.5-0.8B-Base")
    assert adapter_base_model(d) == "Qwen/Qwen3.5-0.8B-Base"
    check_adapter_matches(d, "Qwen/Qwen3.5-0.8B-Base")


def test_an_adapter_trained_from_a_local_copy_of_the_same_model_passes(tmp_path):
    d = _adapter(tmp_path, "/common/scratch/hf/Qwen3.5-4B-Base/")
    check_adapter_matches(d, "Qwen/Qwen3.5-4B-Base")


def test_the_0_8b_adapter_on_the_4b_model_is_refused(tmp_path):
    d = _adapter(tmp_path, "Qwen/Qwen3.5-0.8B-Base")
    with pytest.raises(ValueError, match="config_4b.yaml"):
        check_adapter_matches(d, "Qwen/Qwen3.5-4B-Base")


def test_a_missing_adapter_says_how_to_get_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="Train it first"):
        check_adapter_matches(str(tmp_path / "nothing_here"), "Qwen/Qwen3.5-4B-Base")
