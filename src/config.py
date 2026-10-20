import os
import torch

# this file lives in .../Satellite_image_Segmentation/src/config.py
# so the project root is one folder up from here - computing it this way
# means paths work no matter WHERE you run "python ..." from
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# ---- paths ----
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
METADATA_CSV = os.path.join(DATA_DIR, "metadata.csv")
CLASS_DICT_CSV = os.path.join(DATA_DIR, "class_dict.csv")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# ---- image settings ----
IMAGE_SIZE = 256          # every image/mask gets resized to 256x256
NUM_CLASSES = 7           # urban_land, agriculture_land, rangeland, forest_land, water, barren_land, unknown
                           # (this comes straight from class_dict.csv - 7 rows)

# ---- training settings ----
BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
