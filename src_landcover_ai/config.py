import os
import torch

# this file lives in .../Satellite_image_Segmentation/src_landcover_ai/config.py
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

# deliberately separate from the DeepGlobe run so nothing overwrites it
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints_landcover")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs_landcover")
VISUALIZE_DIR = os.path.join(PROJECT_ROOT, "visualize_landcover")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs_landcover")
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
# LandCover.ai is heavily imbalanced (woodland ~39x more common than
# building by area) - run compute_class_weights.py and paste its output
# here. Leave as None to fall back to plain unweighted CrossEntropyLoss.
CLASS_WEIGHTS = [0.0353, 2.8858, 0.0729, 0.3917, 1.6142]  # background, building, woodland, water, road

# ---- training settings ----
BATCH_SIZE = 16            # bumped up from 8 - lower this back down if you
                            # hit "CUDA out of memory"
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3

NUM_WORKERS = 4             # parallel CPU workers feeding the GPU, so it's
                             # not sitting idle waiting on image loading/resizing
USE_AMP = True               # mixed-precision training - big speedup and
                              # memory saving on CUDA, automatically has no
                              # effect at all when running on CPU

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"