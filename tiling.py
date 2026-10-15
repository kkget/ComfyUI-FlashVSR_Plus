"""Whole-pipeline spatial tiles with positive, normalized overlap weights."""
from __future__ import annotations

import torch


def starts(length: int, size: int, overlap: int):
    size = min(size, length)
    if length <= size:
        return [0]
    values = list(range(0, length - size + 1, size - overlap))
    if values[-1] != length - size:
        values.append(length - size)
    return values


def tile_grid(height, width, size, overlap):
    if size < 256 or size % 128:
        raise ValueError("tile_size must be >=256 and divisible by 128.")
    if overlap < 0 or overlap % 128 or overlap >= size:
        raise ValueError("tile_overlap must be divisible by 128 and smaller than tile_size.")
    return [(y, x, min(size, height), min(size, width))
            for y in starts(height, size, overlap) for x in starts(width, size, overlap)]


def blend_mask(tile, height, width, overlap):
    y, x, h, w = tile
    wy, wx = torch.ones(h), torch.ones(w)
    for vector, start, length, full in ((wy, y, h, height), (wx, x, w, width)):
        count = min(overlap, length)
        if not count:
            continue
        ramp = torch.arange(1, count + 1, dtype=torch.float32) / (count + 1)
        if start:
            vector[:count] = torch.minimum(vector[:count], ramp)
        if start + length < full:
            vector[-count:] = torch.minimum(vector[-count:], ramp.flip(0))
    return wy[:, None] * wx[None, :]
