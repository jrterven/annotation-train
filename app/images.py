"""Shared image policy: preserve original rasters without a fixed pixel ceiling.

Hosted uploads are authenticated and quota-reserved before decoding. Corrupt
formats still fail validation; memory and execution budgets remain independent
of the dimensions of a valid source image.
"""
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
