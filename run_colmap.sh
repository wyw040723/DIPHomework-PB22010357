#!/usr/bin/env bash
# Simple COLMAP pipeline script (adjust paths as needed)
# Usage: bash run_colmap.sh /path/to/images /path/to/workspace

IMAGE_PATH="./data/images"
WORKSPACE="./colmap_work"

mkdir -p "$WORKSPACE"

echo "Step 1: Feature extraction"
colmap feature_extractor \
    --database_path "$WORKSPACE/database.db" \
    --image_path "$IMAGE_PATH" \
    --ImageReader.camera_model PINHOLE

echo "Step 2: Exhaustive feature matching"
colmap exhaustive_matcher \
    --database_path "$WORKSPACE/database.db"

echo "Step 3: Sparse reconstruction (mapper)"
mkdir -p "$WORKSPACE/sparse"
colmap mapper \
    --database_path "$WORKSPACE/database.db" \
    --image_path "$IMAGE_PATH" \
    --output_path "$WORKSPACE/sparse"

echo "Step 4: Image undistortion"
mkdir -p "$WORKSPACE/dense"
colmap image_undistorter \
    --image_path "$IMAGE_PATH" \
    --input_path "$WORKSPACE/sparse/0" \
    --output_path "$WORKSPACE/dense" \
    --output_type COLMAP

echo "Step 5: PatchMatch stereo"
colmap patch_match_stereo \
    --workspace_path "$WORKSPACE/dense" \
    --PatchMatchStereo.geom_consistency true

echo "Step 6: Stereo fusion"
colmap stereo_fusion \
    --workspace_path "$WORKSPACE/dense" \
    --output_path "$WORKSPACE/dense/fused.ply"

echo "Done. Results: $WORKSPACE/sparse and $WORKSPACE/dense/fused.ply"
#!/bin/bash
# COLMAP 3D reconstruction pipeline
# Usage: bash run_colmap.sh

set -e

DATASET_PATH="data"
IMAGE_PATH="$DATASET_PATH/images"
COLMAP_PATH="$DATASET_PATH/colmap"

mkdir -p "$COLMAP_PATH/sparse"
mkdir -p "$COLMAP_PATH/dense"

echo "=== Step 1: Feature Extraction ==="
colmap feature_extractor \
    --database_path "$COLMAP_PATH/database.db" \
    --image_path "$IMAGE_PATH" \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.single_camera 1

echo "=== Step 2: Feature Matching ==="
colmap exhaustive_matcher \
    --database_path "$COLMAP_PATH/database.db"

echo "=== Step 3: Sparse Reconstruction (Bundle Adjustment) ==="
colmap mapper \
    --database_path "$COLMAP_PATH/database.db" \
    --image_path "$IMAGE_PATH" \
    --output_path "$COLMAP_PATH/sparse"

echo "=== Step 4: Image Undistortion ==="
colmap image_undistorter \
    --image_path "$IMAGE_PATH" \
    --input_path "$COLMAP_PATH/sparse/0" \
    --output_path "$COLMAP_PATH/dense"

echo "=== Step 5: Dense Reconstruction (Patch Match Stereo) ==="
colmap patch_match_stereo \
    --workspace_path "$COLMAP_PATH/dense"

echo "=== Step 6: Stereo Fusion ==="
colmap stereo_fusion \
    --workspace_path "$COLMAP_PATH/dense" \
    --output_path "$COLMAP_PATH/dense/fused.ply"


echo "=== Done! ==="
echo "Results:"
echo "  Sparse: $COLMAP_PATH/sparse/0/"
echo "  Dense:  $COLMAP_PATH/dense/fused.ply"
