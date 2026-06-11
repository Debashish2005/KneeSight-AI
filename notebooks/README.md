# Training Notebook

`medicalnet_resnet18_top50_training.ipynb` is the cleaned final notebook used
for the pretrained 3D MedicalNet experiment.

## Experiment Summary

- Dataset: OAI 3D DESS knee MRI volumes
- Input volume: `128 x 128 x 64`
- Selection: 50 slices ranked by intensity variation
- Backbone: pretrained MedicalNet ResNet-18
- Split: volume-level stratified train, validation, and independent test sets
- Training: differential learning rates, gradient accumulation, augmentation,
  early stopping, and validation-AUC checkpointing
- Final independent test ROC-AUC: **0.8251**

## Running on Kaggle

1. Add the OAI MRI dataset containing:
   `normal-3DESS-128-64.npy` and `abnormal-3DESS-128-64.npy`.
2. Select a GPU accelerator.
3. Run the notebook from top to bottom.
4. Download the best checkpoint and generated evaluation artifacts from
   `/kaggle/working`.

The dataset, generated checkpoints, and pretrained weights are not committed
to this repository. They have separate storage and licensing requirements.

The notebook outputs are cleared for a lightweight, readable Git history.
