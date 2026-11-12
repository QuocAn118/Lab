"""Compatibility entry point; implementation moved to src/lab_2/model.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from lab_2.model import (
    HEADS, NUM_CLASSES, SUPPORTED_MODELS, build_model, count_parameters,
    head_of, load_checkpoint, main, save_checkpoint, set_train_mode, trainable_parameters,
)

if __name__ == "__main__":
    main()
