import os
from typing import TYPE_CHECKING

from art.utils.output_dirs import get_model_dir

if TYPE_CHECKING:
    from art.model import TrainableModel


STEP_FILE_NAME = "step.txt"


def get_step_from_file(output_dir: str) -> int | None:
    """Read the step from the step file if it exists.
    
    Returns:
        The step number if the file exists and is valid, None otherwise.
    """
    step_file = os.path.join(output_dir, STEP_FILE_NAME)
    if os.path.exists(step_file):
        try:
            with open(step_file, "r") as f:
                return int(f.read().strip())
        except (ValueError, IOError):
            return None
    return None


def get_step_from_checkpoints(output_dir: str) -> int:
    """Get the step by looking at checkpoint directories.
    
    This is the legacy method of determining the step.
    
    Returns:
        The highest step number found in checkpoint directories, or 0 if none.
    """
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        return 0

    return max(
        (
            int(subdir)
            for subdir in os.listdir(checkpoint_dir)
            if os.path.isdir(os.path.join(checkpoint_dir, subdir)) and subdir.isdigit()
        ),
        default=0,
    )


def get_step_from_dir(output_dir: str) -> int:
    """Get the current step for a model.
    
    First checks for a step file (new method), then falls back to 
    checkpoint directories (legacy method) for backward compatibility.
    
    Returns:
        The current step number.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # First, try to read from step file
    step_from_file = get_step_from_file(output_dir)
    if step_from_file is not None:
        return step_from_file
    
    # Fall back to checkpoint directories for backward compatibility
    return get_step_from_checkpoints(output_dir)


def write_step(output_dir: str, step: int) -> None:
    """Write the step to the step file.
    
    Args:
        output_dir: The model output directory.
        step: The step number to write.
    """
    os.makedirs(output_dir, exist_ok=True)
    step_file = os.path.join(output_dir, STEP_FILE_NAME)
    with open(step_file, "w") as f:
        f.write(str(step))


def get_model_step(model: "TrainableModel", art_path: str) -> int:
    return get_step_from_dir(get_model_dir(model=model, art_path=art_path))
