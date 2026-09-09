"""Model configurations + exact parameter counts for the five LMs."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "common"))
from model import LMConfig, count_params

#           name            d    L   heads  ff     targets
SPECS = [
    ("potato",        128,  5,   4,   336),   # ~1.49M
    ("toaster",       192,  9,   6,   576),   # ~5M
    ("microwave",     288, 10,   9,   864),   # ~12M
    ("blender",       384, 14,  12,  1152),   # ~28M
    ("nuclearfridge", 512, 18,  16,  1152),   # ~52M
]
TARGETS = {
    "potato": 1_490_000,
    "toaster": 5_000_000,
    "microwave": 12_000_000,
    "blender": 28_000_000,
    "nuclearfridge": 52_000_000,
}

CONFIGS = {}
for name, d, L, h, ff in SPECS:
    CONFIGS[name] = LMConfig(name=name, d_model=d, n_layer=L, n_head=h, d_ff=ff)


def main():
    print(f"{'model':<14}{'d':>5}{'L':>4}{'heads':>7}{'ff':>6}{'#params':>12}{'target':>12}{'diff':>8}")
    for name, cfg in CONFIGS.items():
        n = count_params(cfg)
        t = TARGETS[name]
        print(f"{name:<14}{cfg.d_model:>5}{cfg.n_layer:>4}{cfg.n_head:>7}{cfg.d_ff:>6}{n:>12,}{t:>12,}{(n-t)/t*100:>+7.2f}%")


if __name__ == "__main__":
    main()
