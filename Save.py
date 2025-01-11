from pathlib import Path
import torch
"""
Saves the model.

Args:
    model: PyTorch model to save.
    target_dir: Path to target directory.
    model_name: Name of the model file to save.
"""
def save_model(model, target_dir, model_name):
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True, exist_ok=True)
  assert model_name.endswith(".pth") or model_name.endswith(".pt")
  model_save_path = target_dir_path / model_name
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(), f=model_save_path)
  print(f"Model is successfully saved.")
