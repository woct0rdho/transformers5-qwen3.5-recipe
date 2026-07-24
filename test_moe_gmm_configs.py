import pytest

from moe_gmm_configs import (
    _GMM_TUNED_CONFIGS,
    _PTGMM_TUNED_CONFIGS,
    gmm_config,
    ptgmm_config,
)


def _expected_gmm_targets():
    targets = set()
    deepseek = (
        (4096, 4, True),
        (2048, 4, True),
        (4, 4096, True),
        (4, 4096, False),
        (4, 2048, False),
        (4096, 4, False),
        (4096, 2048, True),
        (2048, 4096, True),
        (2048, 4096, False),
        (4096, 2048, False),
    )
    qwen = (
        (2048, 4, True),
        (512, 4, True),
        (4, 1024, True),
        (4, 2048, True),
        (4, 2048, False),
        (4, 512, False),
        (1024, 4, False),
        (2048, 4, False),
        (2048, 512, True),
        (512, 2048, True),
        (512, 2048, False),
        (2048, 512, False),
    )
    for rows in (12288, 49152, 196608):
        targets.update((rows, *shape) for shape in deepseek)
    for rows in (16384, 65536, 262144):
        targets.update((rows, *shape) for shape in qwen)
    return targets


def _expected_ptgmm_targets():
    targets = set()
    deepseek = (
        (4096, 4),
        (2048, 4),
        (4, 4096),
        (4096, 2048),
        (2048, 4096),
    )
    qwen = (
        (2048, 4),
        (512, 4),
        (4, 1024),
        (4, 2048),
        (2048, 512),
        (512, 2048),
    )
    for rows in (12288, 49152, 196608):
        targets.update((rows, *shape) for shape in deepseek)
    for rows in (16384, 65536, 262144):
        targets.update((rows, *shape) for shape in qwen)
    return targets


def test_tuned_tables_contain_only_target_shapes() -> None:
    assert set(_GMM_TUNED_CONFIGS) == _expected_gmm_targets()
    assert set(_PTGMM_TUNED_CONFIGS) == _expected_ptgmm_targets()


def test_tuned_configs_are_valid_and_returned_by_value() -> None:
    for key in _GMM_TUNED_CONFIGS:
        config = gmm_config(*key)
        assert all(value > 0 for value in config.values())
        assert all(
            value & (value - 1) == 0
            for name, value in config.items()
            if name.startswith("BLOCK_SIZE_")
        )
    for key in _PTGMM_TUNED_CONFIGS:
        config = ptgmm_config(*key)
        assert all(value > 0 for value in config.values())
        assert all(
            value & (value - 1) == 0
            for name, value in config.items()
            if name.startswith("BLOCK_SIZE_")
        )

    config = gmm_config(12288, 4, 4096, True)
    config["GRID_DIM"] = -1
    assert gmm_config(12288, 4, 4096, True)["GRID_DIM"] == 256


def test_gmm_dispatch_distinguishes_layout_and_batch() -> None:
    transposed = gmm_config(12288, 4, 4096, True)
    row_major = gmm_config(12288, 4, 4096, False)
    larger_batch = gmm_config(49152, 4, 4096, True)
    assert transposed != row_major
    assert transposed != larger_batch


def test_unknown_shapes_fail_closed() -> None:
    with pytest.raises(ValueError, match="M=7, K=4096, N=4, RHS layout=transposed"):
        gmm_config(7, 4096, 4, True)
    with pytest.raises(ValueError, match="M=7, K=4, N=1024"):
        ptgmm_config(7, 4, 1024)
