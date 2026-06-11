# Input Formats

## Recommended Input

For reproducing the evaluated pipeline, use a prepared `.npy` volume with
shape `128 x 128 x 64`. This is the only input path directly represented in
the reported independent test ROC-AUC of `0.8251`.

For a realistic demonstration, the application also accepts raw-format NIfTI
and zipped DICOM. Those conversion paths are experimental.

## NumPy

- Extension: `.npy`
- Required shape: `128 x 128 x 64`
- Required content: one numeric, finite 3D volume
- Conversion: none

Pickled object arrays are disabled.

## NIfTI

- Extensions: `.nii`, `.nii.gz`
- Required content: one scalar 3D knee MRI
- Intended sequence: sagittal 3D DESS

NIfTI files often do not contain a reliable scanner sequence name, so the API
cannot prove that the scan is DESS. The uploader must verify the acquisition.

The server:

1. Reads the image with SimpleITK.
2. Standardizes physical orientation to LPS.
3. Resamples physical x, y, z dimensions to `64 x 128 x 128`.
4. Converts the SimpleITK array from z, y, x order to the resulting NumPy shape
   `128 x 128 x 64`.

This places the physical left-right axis last, corresponding to sagittal slice
progression.

## DICOM

- Extension: `.zip`
- Required content: one DICOM MRI series, optionally inside a folder
- Required modality tag: `MR`
- Required sequence metadata by default: `DESS` or
  `Dual Echo Steady State`

The archive is rejected when it:

- contains unsafe paths
- exceeds the file-count or extracted-size limits
- contains no readable MR series
- does not identify a DESS series
- produces a multi-channel or non-3D image

When several MR series are present, the converter prefers a DESS-labelled
series and then the series with the largest number of slices.

The ZIP is extracted to an operating-system temporary directory. That
directory is deleted after the series is converted. Patient-identifying
metadata is not returned to the browser, but the server still receives and
briefly processes the original files. Use de-identified research data only.

## Why Conversion Is Experimental

The training notebook loaded already prepared OAI 3D DESS arrays; it did not
start from raw DICOM or NIfTI. Consequently, scanner-specific reconstruction,
field of view, coil, resolution, intensity distribution, and original OAI
cropping cannot be reconstructed from that notebook.

Geometric resampling makes a scan technically compatible with the neural
network, but does not establish clinical or statistical compatibility with the
training distribution. Raw-format predictions must be treated as a software
demonstration until the full conversion pipeline is validated on a labelled
external dataset.
