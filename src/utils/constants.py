"""Project-wide constants confirmed from the Kaggle Data tab."""

# (z, y, x) physical voxel size in micrometers. z is ~4x coarser than xy —
# any physical-distance calculation (watershed spacing, NN-linking distance,
# motion models) must scale z by this ratio or 3D distances will be wrong.
VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)

# Typical volume shape (T, Z, Y, X). Individual samples may vary — always
# read the real shape from the array/zarr.json rather than assuming this.
TYPICAL_SHAPE_TZYX = (100, 64, 256, 256)
