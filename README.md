# Code install

This code has some mandatory dependencies - some that are only used for training and some that are used during evaluation.

### Training dependencies
* ``torch``
* ``monai`` (A deep learning + medical imaging library)
* ``matplotlib``

# Main task

Predict CTA (CT angiography) from NCCT (Non contrast CT). 
1. Architecture
2. Loss
3. Preprocessing
4. Training Schedule
5. Framework
6. Evaluate and Present


# Data
ISLES2024 data set. 
Link to data: https://zenodo.org/records/17652035/files/train.7z?download=1
Link to winning framework: https://github.com/mic-dkfz/nnunet
Link to viewer: https://www.neuropsis.org/nifti_viewer.html (does not work so well)
* We choose to ignore patient meta data.
* What is CTA?
    * Contrast agent scan of brain makes vessels visible
    * Vessels can be visible due to:
        1) Plaque (calcifications with high HU)
        2) Blood clots, which is denser than blood (50/70 HU) - hyperdense vessel signs
* Difficulty
    * Bone is uninteresting but obfuscating.
    * Want to separate bone, background and tissue.
        1) This is made easier by the initial NCCT (use for masking bone)
* Data processing: 
    * NCCT minus CTA + bone filter, approximates the vessels. 
    * labels are vessels, inputs are ncct.
    * Normalize data to be around -1, 1, by a fixed transform for all samples.
        * for ncct: min = -30 HU, max = 100 HU # captures the important tissue variability
        * for vessels: min = 0, max = 300      # Captures the important vessel structure
    * Split the data into 20 / 80 validation + training (apart from the hold out set)


# Baselines
We use two simple baselines
* full 3D volume KNN with K=3
* A convolutional KNN with K=1. Can be thought of as a very wide 3-layer CNN with a simple attention mechanism.
    - input x is copied into N channels, multiplied with identity and subtracted from N data X (1st layer) 
    - trilinear interpolation of X to x grid.
    - x - X is passed through a square activation, Z=abs(x - X)**2 (just needs to be monotone) 
    - Z is filtered with 9x9x3 filter of 1s, creating patch wise difference norms.
    - We pass Z through a hard max, M = m(Z), creating a sparse mask that filters out the best patch in the data
    - Multiply M with Y and sum over all channels to obtain reconstruction.
    - The result is an interpretable patch-wise nearest neighbors search
    - The CKNN is clearly interpretable (it searches through the data for the best candidate patch), and will be used as a baseline.
* There is a cycle GAN implementation online but i couldn't find code.

# Architecture
Train a flow model (generative) to do conditional sampling subject to a patient scan.
Given conditional information Z (NCCT), sample from posterior X of CTA images.
- Major drawback, requires immense amounts of VRAM.
- Train a flow model: (random noise, Z) -> X. 
- Compared to simply: Z -> X. Is there a reasonable interpretation?
I will train a flow matching model
- using eDiff-I (3 expert models trained at three noise ranges)
- The models will take single steps.


# Loss
Angiography is designed to detect blood vessels. We are looking for small increases of high HU values in the brain, concentrated to small regions. 
* Can the vessels be used as a focal region where we concentrate our error measurement?

# Evaluation
SSIM, PSNR, L2, closest data point / Patient in data set.


# Presentation
- Max value over several slices brings out the vessels clearly