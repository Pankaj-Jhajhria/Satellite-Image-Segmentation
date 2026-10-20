import os
import torch

# this file lives in .../Satellite_image_Segmentation/src_landcover_model3/config.py
# MODEL 3 - separate weight-tuning experiment, does not touch model 2 at all
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# ---- LandCover.ai paths ----
# LANDCOVER_DIR should point at the folder split.py was run from - the one
# that now contains images/, masks/, output/, train.txt, val.txt, test.txt
LANDCOVER_DIR = os.path.join(PROJECT_ROOT, "landcover.ai_data2")

# split.py writes every 512x512 tile (both image and mask) into one flat
# "output" folder - this is where the actual training data lives
TILES_DIR = os.path.join(LANDCOVER_DIR, "output")

TRAIN_LIST = os.path.join(LANDCOVER_DIR, "train.txt")
VAL_LIST = os.path.join(LANDCOVER_DIR, "val.txt")
TEST_LIST = os.path.join(LANDCOVER_DIR, "test.txt")

# separate "_model3" folders - model 2's checkpoints/outputs/visualizations/
# logs are never read or written by any file in this folder
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints_landcover_model3")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs_landcover_model3")
VISUALIZE_DIR = os.path.join(PROJECT_ROOT, "visualize_landcover_model3")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs_landcover_model3")
HISTORY_CSV = os.path.join(LOG_DIR, "history.csv")
TEST_METRICS_CSV = os.path.join(LOG_DIR, "test_metrics.csv")

# ---- image settings ----
IMAGE_SIZE = 256
NUM_CLASSES = 5  # background, building, woodland, water, road

# LandCover.ai masks store the class id directly as the pixel value
# (0-4) - there is no class_dict.csv like DeepGlobe shipped with, so
# these display colors are just picked by hand for visualize.py/predict.py
CLASS_COLORS = [
    ("background", (0, 0, 0)),
    ("building",   (255, 0, 0)),
    ("woodland",   (0, 153, 0)),
    ("water",      (0, 102, 255)),
    ("road",       (255, 255, 0)),
]

# ---- class weighting ----
# model 2 used raw inverse-frequency weights (very aggressive - building
# ended up weighted ~82x background), which caused it to hallucinate
# scattered building/road pixels into plain background areas. Run
# compute_class_weights.py, compare the formulas it prints, and paste
# whichever one you pick here. Leave as None for plain unweighted loss.
CLASS_WEIGHTS = [0.2318, 2.0956, 0.3332, 0.7721, 1.5673]  # fill in after running compute_class_weights.py

# ---- training settings ----
BATCH_SIZE = 4              # dropped from 16 - smaller batches mean the GPU
                             # does less work per step and never has to run
                             # at full tilt, which is what drives up temp.
                             # Slower overall, but lighter and cooler.
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3

NUM_WORKERS = 1              # dropped from 2 - fewer background processes,
                              # less RAM overhead, less CPU feeding pressure
                              # on the GPU
USE_AMP = False               # turned off - AMP pushes the GPU's tensor
                               # cores harder to go faster, which raises
                               # power draw and heat. Plain FP32 is slower
                               # but runs the GPU less aggressively.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"