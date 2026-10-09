"""Reproduces Figure 1 of the paper.

Usage:
    python experiments/figure1.py

Output is saved to results/figure1.png
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np

# from src.method import ...


def main():
    os.makedirs("results", exist_ok=True)

    # --- replace with the actual experiment ---
    rng = np.random.default_rng(0)  # fixed seed => reproducible
    x = np.linspace(0, 10, 200)
    y = np.sin(x) + 0.1 * rng.standard_normal(x.size)

    plt.figure(figsize=(5, 3.5))
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig("results/figure1.png", dpi=300)
    print("Saved results/figure1.png")


if __name__ == "__main__":
    main()
