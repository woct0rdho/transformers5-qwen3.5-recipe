from typing import TypeAlias

ConfigValues: TypeAlias = tuple[int, int, int, int, int, int, int]

# Target matrix inventory, all BF16. Expert counts come from the routed law:
#   family   routed law       experts top-k routed M at physical B1 / B4 / B16
#   DeepSeek deepseek-learned 256     6     12,288 / 49,152 / 196,608
#   Qwen     qwen-learned     256     8     16,384 / 65,536 / 262,144
#   Qwen3.8  qwen3.8-learned  512     10    20,480 / 81,920 / 327,680
#
# GMM T denotes a transposed factor/weight view. GMM N is row-major. PTGMM uses
# a transposed row-major lhs. Rank-4 LoRA target KxN shapes are:
#   DeepSeek T: 4096x4, 2048x4, 4x4096
#            N: 4x4096, 4x2048, 4096x4
#        PTGMM: 4096x4, 2048x4, 4x4096
#   Qwen     T: 2048x4, 512x4, 4x1024, 4x2048
#            N: 4x2048, 4x512, 1024x4, 2048x4
#        PTGMM: 2048x4, 512x4, 4x1024, 4x2048
#   Qwen3.8  T: 2560x4, 640x4, 4x1280, 4x2560
#            N: 4x2560, 4x640, 1280x4, 2560x4
#        PTGMM: 2560x4, 640x4, 4x1280, 4x2560
# Base-model target shapes are:
#   DeepSeek T: 4096x2048, 2048x4096
#            N: 2048x4096, 4096x2048
#        PTGMM: 4096x2048, 2048x4096
#   Qwen     T: 2048x512, 512x2048
#            N: 512x2048, 2048x512
#        PTGMM: 2048x512, 512x2048
#   Qwen3.8  T: 2560x640, 640x2560
#            N: 640x2560, 2560x640
#        PTGMM: 2560x640, 640x2560

_CONFIG_KEYS = (
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_K",
    "BLOCK_SIZE_N",
    "GROUP_SIZE",
    "GRID_DIM",
    "num_warps",
    "num_stages",
)

# Exact target keys prevent a transposed LoRA factor config from leaking into a
# row-major input gradient with the same logical K and N. Keys are
# (expert prior, total routed rows, K, N, transposed RHS) for GMM and
# (expert prior, total routed rows, K, N) for PTGMM. Values follow _CONFIG_KEYS.
_GMM_CONFIGS: dict[tuple[str, int, int, int, bool], ConfigValues] = {
    ("qwen-learned", 16384, 2048, 4, True): (16, 256, 16, 8, 80, 2, 1),
    ("qwen-learned", 16384, 512, 4, True): (64, 128, 32, 1, 40, 8, 3),
    ("qwen-learned", 16384, 4, 1024, True): (16, 16, 128, 1, 256, 2, 1),
    ("qwen-learned", 16384, 4, 2048, True): (32, 16, 64, 1, 256, 2, 2),
    ("qwen-learned", 16384, 4, 2048, False): (32, 32, 32, 1, 256, 4, 1),
    ("qwen-learned", 16384, 4, 512, False): (64, 16, 64, 1, 160, 4, 2),
    ("qwen-learned", 16384, 1024, 4, False): (32, 64, 16, 8, 256, 1, 1),
    ("qwen-learned", 16384, 2048, 4, False): (16, 256, 16, 8, 80, 4, 3),
    ("qwen-learned", 16384, 2048, 512, True): (64, 64, 64, 2, 80, 2, 3),
    ("qwen-learned", 16384, 512, 2048, True): (128, 64, 64, 1, 80, 4, 2),
    ("qwen-learned", 16384, 512, 2048, False): (64, 32, 64, 8, 256, 2, 1),
    ("qwen-learned", 16384, 2048, 512, False): (64, 64, 128, 8, 40, 8, 1),
    ("qwen-learned", 65536, 2048, 4, True): (16, 256, 16, 8, 40, 2, 1),
    ("qwen-learned", 65536, 512, 4, True): (32, 256, 16, 1, 40, 2, 3),
    ("qwen-learned", 65536, 4, 1024, True): (32, 16, 64, 2, 256, 2, 1),
    ("qwen-learned", 65536, 4, 2048, True): (32, 16, 64, 1, 160, 2, 2),
    ("qwen-learned", 65536, 4, 2048, False): (16, 16, 64, 1, 160, 1, 2),
    ("qwen-learned", 65536, 4, 512, False): (32, 16, 64, 1, 256, 2, 2),
    ("qwen-learned", 65536, 1024, 4, False): (32, 128, 16, 8, 80, 2, 2),
    ("qwen-learned", 65536, 2048, 4, False): (16, 256, 16, 4, 80, 4, 3),
    ("qwen-learned", 65536, 2048, 512, True): (128, 64, 128, 1, 40, 8, 3),
    ("qwen-learned", 65536, 512, 2048, True): (128, 64, 64, 1, 80, 8, 1),
    ("qwen-learned", 65536, 512, 2048, False): (128, 32, 256, 1, 40, 8, 2),
    ("qwen-learned", 65536, 2048, 512, False): (64, 32, 128, 1, 80, 8, 2),
    ("qwen-learned", 262144, 2048, 4, True): (16, 64, 16, 1, 80, 2, 1),
    ("qwen-learned", 262144, 512, 4, True): (32, 256, 16, 1, 20, 4, 3),
    ("qwen-learned", 262144, 4, 1024, True): (32, 16, 64, 1, 160, 2, 1),
    ("qwen-learned", 262144, 4, 2048, True): (32, 16, 64, 1, 160, 2, 2),
    ("qwen-learned", 262144, 4, 2048, False): (16, 16, 64, 1, 160, 1, 2),
    ("qwen-learned", 262144, 4, 512, False): (64, 16, 32, 1, 160, 2, 1),
    ("qwen-learned", 262144, 1024, 4, False): (32, 128, 16, 1, 40, 4, 3),
    ("qwen-learned", 262144, 2048, 4, False): (16, 128, 16, 1, 160, 2, 1),
    ("qwen-learned", 262144, 2048, 512, True): (64, 64, 128, 4, 80, 8, 1),
    ("qwen-learned", 262144, 512, 2048, True): (128, 64, 64, 1, 80, 8, 1),
    ("qwen-learned", 262144, 512, 2048, False): (64, 32, 256, 8, 40, 8, 1),
    ("qwen-learned", 262144, 2048, 512, False): (64, 32, 128, 1, 80, 8, 2),
    ("deepseek-learned", 12288, 4096, 4, True): (16, 256, 16, 1, 40, 4, 3),
    ("deepseek-learned", 12288, 2048, 4, True): (16, 256, 16, 2, 40, 4, 3),
    ("deepseek-learned", 12288, 4, 4096, True): (32, 16, 64, 8, 256, 2, 2),
    ("deepseek-learned", 12288, 4, 4096, False): (16, 16, 256, 4, 160, 2, 1),
    ("deepseek-learned", 12288, 4, 2048, False): (32, 32, 32, 1, 256, 4, 1),
    ("deepseek-learned", 12288, 4096, 4, False): (16, 256, 16, 2, 80, 4, 3),
    ("deepseek-learned", 12288, 4096, 2048, True): (64, 64, 32, 1, 40, 8, 2),
    ("deepseek-learned", 12288, 2048, 4096, True): (32, 64, 32, 8, 80, 1, 3),
    ("deepseek-learned", 12288, 2048, 4096, False): (64, 32, 256, 8, 80, 4, 2),
    ("deepseek-learned", 12288, 4096, 2048, False): (64, 32, 256, 8, 80, 4, 2),
    ("deepseek-learned", 49152, 4096, 4, True): (16, 64, 16, 1, 80, 2, 1),
    ("deepseek-learned", 49152, 2048, 4, True): (16, 256, 16, 4, 40, 2, 1),
    ("deepseek-learned", 49152, 4, 4096, True): (64, 16, 32, 1, 160, 2, 2),
    ("deepseek-learned", 49152, 4, 4096, False): (16, 16, 128, 1, 160, 1, 1),
    ("deepseek-learned", 49152, 4, 2048, False): (16, 16, 128, 1, 256, 1, 1),
    ("deepseek-learned", 49152, 4096, 4, False): (16, 256, 16, 8, 80, 4, 3),
    ("deepseek-learned", 49152, 4096, 2048, True): (128, 64, 64, 1, 40, 8, 2),
    ("deepseek-learned", 49152, 2048, 4096, True): (128, 64, 64, 1, 80, 4, 2),
    ("deepseek-learned", 49152, 2048, 4096, False): (128, 32, 256, 8, 40, 8, 1),
    ("deepseek-learned", 49152, 4096, 2048, False): (128, 32, 256, 2, 40, 8, 3),
    ("deepseek-learned", 196608, 4096, 4, True): (16, 256, 16, 1, 80, 4, 3),
    ("deepseek-learned", 196608, 2048, 4, True): (16, 256, 16, 1, 20, 4, 3),
    ("deepseek-learned", 196608, 4, 4096, True): (32, 16, 64, 1, 160, 2, 1),
    ("deepseek-learned", 196608, 4, 4096, False): (32, 32, 32, 1, 256, 4, 1),
    ("deepseek-learned", 196608, 4, 2048, False): (16, 16, 64, 1, 80, 1, 2),
    ("deepseek-learned", 196608, 4096, 4, False): (16, 64, 16, 1, 160, 1, 1),
    ("deepseek-learned", 196608, 4096, 2048, True): (128, 64, 128, 8, 40, 8, 2),
    ("deepseek-learned", 196608, 2048, 4096, True): (64, 32, 64, 8, 80, 8, 1),
    ("deepseek-learned", 196608, 2048, 4096, False): (128, 32, 256, 8, 40, 8, 3),
    ("deepseek-learned", 196608, 4096, 2048, False): (128, 32, 256, 2, 40, 8, 1),
    ("deepseek-hash", 12288, 4096, 4, True): (16, 256, 16, 1, 40, 4, 3),
    ("deepseek-hash", 12288, 2048, 4, True): (16, 256, 16, 2, 40, 4, 3),
    ("deepseek-hash", 12288, 4, 4096, True): (32, 16, 64, 8, 256, 2, 2),
    ("deepseek-hash", 12288, 4, 4096, False): (16, 16, 256, 8, 160, 2, 1),
    ("deepseek-hash", 12288, 4, 2048, False): (32, 32, 32, 1, 256, 4, 1),
    ("deepseek-hash", 12288, 4096, 4, False): (16, 256, 16, 8, 80, 4, 1),
    ("deepseek-hash", 12288, 4096, 2048, True): (64, 64, 32, 1, 40, 8, 2),
    ("deepseek-hash", 12288, 2048, 4096, True): (32, 32, 32, 2, 80, 1, 3),
    ("deepseek-hash", 12288, 2048, 4096, False): (64, 32, 256, 8, 80, 4, 2),
    ("deepseek-hash", 12288, 4096, 2048, False): (64, 32, 256, 8, 80, 4, 2),
    ("deepseek-hash", 49152, 4096, 4, True): (16, 64, 16, 1, 80, 2, 1),
    ("deepseek-hash", 49152, 2048, 4, True): (16, 256, 16, 4, 80, 2, 1),
    ("deepseek-hash", 49152, 4, 4096, True): (64, 16, 32, 4, 256, 2, 2),
    ("deepseek-hash", 49152, 4, 4096, False): (16, 16, 128, 1, 160, 1, 1),
    ("deepseek-hash", 49152, 4, 2048, False): (16, 16, 128, 1, 256, 1, 1),
    ("deepseek-hash", 49152, 4096, 4, False): (16, 256, 16, 8, 80, 4, 3),
    ("deepseek-hash", 49152, 4096, 2048, True): (128, 64, 128, 1, 40, 8, 3),
    ("deepseek-hash", 49152, 2048, 4096, True): (128, 64, 32, 1, 80, 4, 2),
    ("deepseek-hash", 49152, 2048, 4096, False): (64, 32, 256, 4, 40, 8, 3),
    ("deepseek-hash", 49152, 4096, 2048, False): (128, 32, 256, 2, 40, 8, 3),
    ("deepseek-hash", 196608, 4096, 4, True): (16, 256, 16, 4, 80, 4, 3),
    ("deepseek-hash", 196608, 2048, 4, True): (16, 256, 16, 1, 20, 4, 3),
    ("deepseek-hash", 196608, 4, 4096, True): (32, 16, 64, 1, 160, 2, 1),
    ("deepseek-hash", 196608, 4, 4096, False): (32, 32, 32, 1, 256, 4, 1),
    ("deepseek-hash", 196608, 4, 2048, False): (16, 16, 64, 1, 80, 1, 2),
    ("deepseek-hash", 196608, 4096, 4, False): (16, 64, 16, 1, 160, 1, 1),
    ("deepseek-hash", 196608, 4096, 2048, True): (128, 64, 128, 8, 40, 8, 2),
    ("deepseek-hash", 196608, 2048, 4096, True): (128, 64, 64, 8, 80, 8, 1),
    ("deepseek-hash", 196608, 2048, 4096, False): (128, 32, 256, 8, 40, 8, 3),
    ("deepseek-hash", 196608, 4096, 2048, False): (128, 64, 128, 8, 40, 8, 3),
    ("qwen3.8-learned", 20480, 2560, 640, True): (64, 64, 64, 2, 80, 4, 2),
    ("qwen3.8-learned", 20480, 640, 2560, True): (32, 64, 64, 1, 160, 4, 2),
    ("qwen3.8-learned", 20480, 640, 2560, False): (64, 32, 64, 8, 160, 4, 3),
    ("qwen3.8-learned", 20480, 2560, 640, False): (64, 32, 128, 8, 80, 8, 1),
    ("qwen3.8-learned", 20480, 2560, 4, True): (16, 256, 16, 8, 160, 2, 1),
    ("qwen3.8-learned", 20480, 640, 4, True): (32, 128, 16, 1, 160, 2, 1),
    ("qwen3.8-learned", 20480, 4, 1280, True): (16, 16, 64, 1, 512, 1, 2),
    ("qwen3.8-learned", 20480, 4, 2560, True): (32, 16, 64, 8, 256, 2, 2),
    ("qwen3.8-learned", 20480, 4, 2560, False): (32, 32, 32, 1, 256, 4, 1),
    ("qwen3.8-learned", 20480, 4, 640, False): (32, 16, 32, 1, 256, 4, 2),
    ("qwen3.8-learned", 20480, 1280, 4, False): (32, 64, 16, 8, 256, 1, 1),
    ("qwen3.8-learned", 20480, 2560, 4, False): (16, 128, 16, 1, 160, 2, 1),
    ("qwen3.8-learned", 81920, 2560, 640, True): (128, 64, 128, 1, 40, 8, 3),
    ("qwen3.8-learned", 81920, 640, 2560, True): (128, 64, 64, 1, 80, 8, 1),
    ("qwen3.8-learned", 81920, 640, 2560, False): (64, 32, 256, 1, 40, 8, 1),
    ("qwen3.8-learned", 81920, 2560, 640, False): (64, 32, 128, 1, 80, 8, 2),
    ("qwen3.8-learned", 81920, 2560, 4, True): (16, 256, 16, 8, 40, 2, 1),
    ("qwen3.8-learned", 81920, 640, 4, True): (64, 128, 16, 4, 40, 4, 1),
    ("qwen3.8-learned", 81920, 4, 1280, True): (32, 16, 64, 2, 256, 2, 1),
    ("qwen3.8-learned", 81920, 4, 2560, True): (32, 16, 64, 1, 160, 2, 2),
    ("qwen3.8-learned", 81920, 4, 2560, False): (16, 16, 64, 1, 160, 1, 2),
    ("qwen3.8-learned", 81920, 4, 640, False): (32, 16, 128, 1, 80, 4, 2),
    ("qwen3.8-learned", 81920, 1280, 4, False): (32, 256, 16, 1, 80, 4, 2),
    ("qwen3.8-learned", 81920, 2560, 4, False): (16, 256, 16, 4, 80, 4, 3),
    ("qwen3.8-learned", 327680, 2560, 640, True): (64, 64, 128, 4, 80, 8, 1),
    ("qwen3.8-learned", 327680, 640, 2560, True): (128, 64, 64, 1, 80, 8, 1),
    ("qwen3.8-learned", 327680, 640, 2560, False): (64, 32, 256, 8, 40, 8, 1),
    ("qwen3.8-learned", 327680, 2560, 640, False): (64, 32, 128, 1, 80, 8, 2),
    ("qwen3.8-learned", 327680, 2560, 4, True): (32, 256, 16, 1, 40, 2, 1),
    ("qwen3.8-learned", 327680, 640, 4, True): (32, 128, 16, 1, 80, 4, 3),
    ("qwen3.8-learned", 327680, 4, 1280, True): (32, 16, 64, 1, 160, 2, 1),
    ("qwen3.8-learned", 327680, 4, 2560, True): (32, 16, 64, 1, 160, 2, 2),
    ("qwen3.8-learned", 327680, 4, 2560, False): (16, 16, 64, 1, 80, 1, 2),
    ("qwen3.8-learned", 327680, 4, 640, False): (64, 16, 32, 1, 160, 2, 3),
    ("qwen3.8-learned", 327680, 1280, 4, False): (64, 128, 16, 4, 40, 4, 1),
    ("qwen3.8-learned", 327680, 2560, 4, False): (16, 128, 16, 1, 160, 2, 1),
}

# PTGMM uses a transposed row-major lhs.
_PTGMM_CONFIGS: dict[tuple[str, int, int, int], ConfigValues] = {
    ("qwen-learned", 16384, 2048, 4): (16, 256, 16, 8, 80, 4, 1),
    ("qwen-learned", 16384, 512, 4): (64, 256, 16, 1, 80, 8, 1),
    ("qwen-learned", 16384, 4, 1024): (32, 16, 128, 2, 256, 2, 1),
    ("qwen-learned", 16384, 4, 2048): (16, 16, 256, 8, 160, 4, 1),
    ("qwen-learned", 16384, 2048, 512): (32, 64, 128, 1, 80, 8, 1),
    ("qwen-learned", 16384, 512, 2048): (32, 64, 128, 4, 80, 8, 1),
    ("qwen-learned", 65536, 2048, 4): (32, 512, 16, 4, 80, 8, 1),
    ("qwen-learned", 65536, 512, 4): (32, 128, 16, 8, 256, 4, 2),
    ("qwen-learned", 65536, 4, 1024): (32, 16, 64, 4, 80, 2, 2),
    ("qwen-learned", 65536, 4, 2048): (64, 16, 256, 2, 256, 4, 1),
    ("qwen-learned", 65536, 2048, 512): (16, 64, 128, 1, 80, 8, 3),
    ("qwen-learned", 65536, 512, 2048): (16, 64, 128, 4, 80, 8, 3),
    ("qwen-learned", 262144, 2048, 4): (64, 256, 16, 1, 256, 4, 1),
    ("qwen-learned", 262144, 512, 4): (64, 128, 16, 4, 160, 8, 2),
    ("qwen-learned", 262144, 4, 1024): (32, 16, 64, 2, 80, 2, 2),
    ("qwen-learned", 262144, 4, 2048): (64, 16, 256, 8, 256, 4, 1),
    ("qwen-learned", 262144, 2048, 512): (16, 128, 128, 2, 80, 8, 1),
    ("qwen-learned", 262144, 512, 2048): (32, 128, 256, 1, 40, 8, 3),
    ("deepseek-learned", 12288, 4096, 4): (16, 64, 16, 1, 256, 2, 1),
    ("deepseek-learned", 12288, 2048, 4): (16, 64, 16, 1, 160, 1, 1),
    ("deepseek-learned", 12288, 4, 4096): (16, 16, 256, 8, 160, 2, 1),
    ("deepseek-learned", 12288, 4096, 2048): (16, 64, 512, 1, 40, 8, 3),
    ("deepseek-learned", 12288, 2048, 4096): (16, 64, 512, 1, 40, 8, 3),
    ("deepseek-learned", 49152, 4096, 4): (32, 512, 16, 2, 80, 8, 1),
    ("deepseek-learned", 49152, 2048, 4): (32, 512, 16, 1, 80, 8, 1),
    ("deepseek-learned", 49152, 4, 4096): (64, 16, 256, 8, 80, 4, 1),
    ("deepseek-learned", 49152, 4096, 2048): (16, 64, 256, 8, 80, 4, 3),
    ("deepseek-learned", 49152, 2048, 4096): (16, 64, 512, 2, 40, 8, 3),
    ("deepseek-learned", 196608, 4096, 4): (32, 512, 16, 2, 256, 4, 1),
    ("deepseek-learned", 196608, 2048, 4): (32, 512, 16, 1, 80, 8, 1),
    ("deepseek-learned", 196608, 4, 4096): (64, 16, 256, 8, 160, 4, 1),
    ("deepseek-learned", 196608, 4096, 2048): (16, 64, 256, 4, 80, 4, 1),
    ("deepseek-learned", 196608, 2048, 4096): (16, 64, 512, 1, 40, 8, 2),
    ("deepseek-hash", 12288, 4096, 4): (32, 256, 16, 1, 256, 4, 1),
    ("deepseek-hash", 12288, 2048, 4): (16, 64, 16, 1, 160, 1, 1),
    ("deepseek-hash", 12288, 4, 4096): (64, 16, 256, 1, 80, 4, 1),
    ("deepseek-hash", 12288, 4096, 2048): (16, 64, 512, 1, 40, 8, 1),
    ("deepseek-hash", 12288, 2048, 4096): (16, 64, 512, 1, 40, 8, 1),
    ("deepseek-hash", 49152, 4096, 4): (32, 512, 16, 2, 80, 8, 1),
    ("deepseek-hash", 49152, 2048, 4): (32, 512, 16, 1, 80, 8, 1),
    ("deepseek-hash", 49152, 4, 4096): (64, 16, 256, 8, 80, 4, 1),
    ("deepseek-hash", 49152, 4096, 2048): (16, 64, 256, 8, 80, 4, 3),
    ("deepseek-hash", 49152, 2048, 4096): (16, 64, 512, 2, 40, 8, 3),
    ("deepseek-hash", 196608, 4096, 4): (16, 512, 16, 1, 80, 4, 1),
    ("deepseek-hash", 196608, 2048, 4): (32, 512, 16, 1, 80, 8, 1),
    ("deepseek-hash", 196608, 4, 4096): (64, 16, 256, 8, 160, 4, 1),
    ("deepseek-hash", 196608, 4096, 2048): (16, 64, 256, 4, 80, 4, 1),
    ("deepseek-hash", 196608, 2048, 4096): (16, 64, 512, 4, 40, 8, 1),
    ("qwen3.8-learned", 20480, 2560, 640): (32, 64, 128, 1, 80, 8, 1),
    ("qwen3.8-learned", 20480, 640, 2560): (32, 64, 256, 2, 40, 8, 1),
    ("qwen3.8-learned", 20480, 2560, 4): (32, 128, 16, 1, 160, 4, 1),
    ("qwen3.8-learned", 20480, 640, 4): (64, 128, 16, 1, 80, 8, 1),
    ("qwen3.8-learned", 20480, 4, 1280): (32, 16, 128, 2, 256, 2, 1),
    ("qwen3.8-learned", 20480, 4, 2560): (32, 16, 256, 1, 80, 8, 1),
    ("qwen3.8-learned", 81920, 2560, 640): (32, 64, 128, 1, 80, 8, 3),
    ("qwen3.8-learned", 81920, 640, 2560): (32, 128, 128, 8, 40, 8, 2),
    ("qwen3.8-learned", 81920, 2560, 4): (32, 128, 16, 4, 80, 8, 1),
    ("qwen3.8-learned", 81920, 640, 4): (64, 128, 16, 4, 256, 4, 2),
    ("qwen3.8-learned", 81920, 4, 1280): (32, 16, 128, 1, 160, 4, 2),
    ("qwen3.8-learned", 81920, 4, 2560): (32, 16, 256, 8, 40, 4, 1),
    ("qwen3.8-learned", 327680, 2560, 640): (16, 128, 128, 1, 80, 8, 1),
    ("qwen3.8-learned", 327680, 640, 2560): (32, 128, 128, 2, 40, 8, 3),
    ("qwen3.8-learned", 327680, 2560, 4): (32, 256, 16, 4, 160, 4, 1),
    ("qwen3.8-learned", 327680, 640, 4): (64, 128, 16, 4, 160, 8, 2),
    ("qwen3.8-learned", 327680, 4, 1280): (32, 16, 128, 1, 512, 2, 2),
    ("qwen3.8-learned", 327680, 4, 2560): (32, 16, 256, 8, 40, 4, 1),
}


def _config(values: ConfigValues) -> dict[str, int]:
    return dict(zip(_CONFIG_KEYS, values, strict=True))


def gmm_config(
    m: int,
    k: int,
    n: int,
    transposed_rhs: bool,
    expert_prior: str,
) -> dict[str, int]:
    """Return an exact-target config selected by routed prior and layout."""

    prior = expert_prior
    key = (prior, m, k, n, transposed_rhs)
    values = _GMM_CONFIGS.get(key)
    if values is None:
        layout = "transposed" if transposed_rhs else "row-major"
        raise ValueError(
            "No tuned gfx1151 AITER GMM config for "
            f"prior={prior}, M={m}, K={k}, N={n}, RHS layout={layout}."
        )
    return _config(values)


def ptgmm_config(m: int, k: int, n: int, expert_prior: str) -> dict[str, int]:
    """Return an exact-target PTGMM config selected by routed prior."""

    prior = expert_prior
    key = (prior, m, k, n)
    values = _PTGMM_CONFIGS.get(key)
    if values is None:
        raise ValueError(
            f"No tuned gfx1151 AITER PTGMM config for "
            f"prior={prior}, M={m}, K={k}, N={n}."
        )
    return _config(values)
