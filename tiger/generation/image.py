"""Portable image-only version of the original vLLM generation script."""

from tiger.generation.client import main


if __name__ == "__main__":
    main("image")
