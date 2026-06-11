# Model Artifacts

Place the trained checkpoint here as:

```text
medicalnet_resnet18_top50.pth
```

Model weights are excluded from Git because the current checkpoint is larger
than GitHub's normal 100 MB file limit and may have separate redistribution
terms.

For deployment, store the checkpoint in private object storage or a model
registry and configure:

```text
MODEL_URL=https://example.com/medicalnet_resnet18_top50.pth
MODEL_SHA256=cf82197cfbafdb38730ae06e7f5d0be5d77e232f87a19cfe21ec9acace802022
```

Run `python backend/scripts/download_model.py` from the repository root to
download it. The backend also supports a custom local path through
`MODEL_PATH`.

The application reconstructs the final MedicalNet ResNet-18 architecture and
loads the state dictionary with `strict=True`. The displayed class threshold
remains `0.50` until the validation-selected threshold is exported.
