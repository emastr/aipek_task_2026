## Assumptions

I made a number of assumptions about the data and relevant applications.
1. The purpose of brain CTA is mainly to find topology changes (tears, blockages) and deformations (aneurysms, narrowing, malformations) to the vessels. 
2. Any measurement fluctuations in the soft tissue (<100 HU range) outside the vessel structure is not important for clinicians.
3. The registration is "perfect" in the sense that any deformation of the NCCT data will necessarily degrade the results. The optimal approximation to CTA conditioned on a known vascular system is simply to superimpose the vessels on top of the NCCT image.
4. I am not allowed to use any meta data or patient data as a part of the regression task. The predictions must be made solely from the NCCT voxel values.

I propose the following pipeline.

## Data processing
Let $\{(x_n, y_n)\}_{n=1}^N$ denote the original data set, where $x_n\in \mathbb{R}^d$ are the NCCT images measured in HU , and $y_n\in \mathbb{R}^d$ the CTA images, also in HU. I define a data transform $T\colon \mathbb{R}^d \times \mathbb{R}^d\to \mathbb{R}^d\times \mathbb{R}^d$ with $(x,y)\mapsto (\tilde x, \tilde y)$, as follows.
1. Create a bone mask from $x$ by a gaussian blurring of a thresholding:
$$
    m_x = 1 -\Phi_\sigma\circledast\mathbf{1}_{[150, \infty)}(x),
$$
where the characteristic function acts voxel wise, and $\Phi_\sigma$ is a gaussian $2\sigma/p_x \times 2\sigma/p_y \times 2\sigma/p_z$ (rounded to nearest integer) filter with gaussian entries, and $p_x, p_y, p_z$ are the pixel widths extracted from the data header.

2. Create a masked difference of $x$ and $y$:
$$
    z = m_x \odot (y - x).
$$
The order of subtraction is important because we expect $y> x$.

3. For $x$, the interesting soft tissue is in the range $-30$ to $90$ HU and for $z$ it is $0$ to $400$ HU. I apply a normalization step and then clamp to $[-1, 1]$:
$$
    \hat x = \psi_{[-30,90]}(x), \; \hat y = \psi_{[0, 400]}(z)
$$
Where $\psi_{[a,b]}(x) = \mathrm{clamp}(\tfrac{2z-a-b}{b-a})$.


4. Interpolate the result to a $128 \times 128\times D$ grid using trilinear interpolation.
$$
    (\tilde x, \tilde y) = (P(\hat x), P(\hat y)),
$$
where $P$ is an interpolator. This was only done to decrease the VRAM requirements during training.


The data transform is then 
$$
    T(x, y) = (P(\psi_{[-30,90]}(x)), P(\psi_{[0,400]}(m_x\odot (y -x)))).
$$
These are my input-output pairs for training. Below is an example of some data points in the set (I have taken the maximum value of a chunk of slices to make the vessel topology visible) 

\< image here \>

I split the data into the hold out set, and then 20% validation and 80% training data.

## Training loss
Due to limited time filtering the data, there are multiple outliers where the bone still dominates the output even after filtering. I try to remedy the effect of these outliers by implementing a weighted L1 loss :
$$
    L(y_{pred}, \tilde y) = ((1 + \tilde y)\alpha + (1 - \tilde y)(1-\alpha)) |\tilde y - y_{pred}| ,
$$
where $\alpha$ is a weight that governs the relative penalization of false positives (type I) over false negatives (type II). Most of the image is close to -1, meaning an equal preference for type I and II errors likely results in a completely black image. I put $\alpha=0.9$. Note that the overparameterized optimal solution is still $y_{pred} = \tilde y$.The L1 is less sensitive to outliers than L2.


## Architectures

I have two naive handcrafted, interpretable baseline models, a UNet for reference and the main model is a SwinUNetR:
#### Baselines
* A K-Nearest neighbors with K=3 and 2-norm on the interpolated data $(\tilde x, \tilde y)$, 
* A sliding window 1-NN that sets the voxel value for the center voxel $i$ of a 9 x 9 x 3 patch of indices $I\ni i$ to
$$
 y^{1-NN}[i] := \tilde y_{n^*}[i],\quad  n^* = \arg\min_n \| \tilde x[I] - \tilde x_n[I]\|
$$
The sliding window 1-NN can be implemented as a 3 layer CNN with one filter per data point and global bias. The idea is to identify scans with a similar patch and extract the value.
* UNet (Ronneberger), a classic 2D Unet architecture from the Monai library. Evaluates on 8 contiguous horizontal slices (each slice is treated as a channel in the input) and positional embeddings.
#### Main
* SwinUNetR a development of UNet that uses Shifted window transformers. Evaluates on the full volume.

## Evaluation

Clamping degrades the result, but after training a neural network to predict $\tilde y$ from $\tilde x$, we can use the original NCCT data ($x$) to construct a pseudoinverse $t^\dag$:
$$
t^\dag(x, \tilde y) = x + \psi^\dag_{[0, 400]}(P^\dag (\tilde y)),
$$
where $\psi^\dag_{[a,b]}(x) = \tfrac{1}{2}(b-a)x + \tfrac{1}{2}(a + b)$ and $P^\dag$ is an upsampling using trilinear interpolation. I then run a number of tests on the validation data.

#### Quantitative:
For quantitative metrics, I mainly look at the transformed output $\tilde y$ because it disregards errors in the bone structure. Mainly,
1. Simple 2-norm relative error on the filtered  data:
$$
    \mathrm{RMSE}(\tilde y, y_{net}) = \| \tilde y - y_{net}\|/\|\tilde y\|.
$$
2. Intersection over union (I o U) on a 0-1 mask constructed from $\tilde y$ by thresholding:
$$
    \mathrm{IOU}(\tilde y, y_{net}) = \frac{|\{ijk\colon \tilde y_{ijk} > \beta\}\cap\{ijk \colon y_{ijk}^{net}>\beta\}|}{|\{ijk\colon \tilde y_{ijk} > \beta\}\cup\{ijk \colon y_{ijk}^{net}>\beta\}|},
$$
where I put $\beta = 0.5$.
3. For completeness: 2-norm error over the original CTA image $y$:
$$
    \mathrm{RMSE}(x,y,y_{net}) = \|y - t^\dag (x, y_{net})\|/\|y\|
$$
My point will be that this last metric will be largely misleading in terms of performance (good results will be obtained by even the baselines).

#### Qualitative
For the qualitative presentation, I visualize the outputs $y_{net}$ and its CTA mapping $t^\dag(x, y_{net})$. We mainly use 3d slicing similar to https://www.neuropsis.org/.
1. We compare slices of the vessel images $\tilde y, y_{net}$. 
2. We compare slices of the CTA images $y, t^{\dag}(x,y_{net})$.
3. We compare the segmentation results from CTA to CoW (Circle of Willis) segmentation models from the TopCoWSubmissions challenge, evaluated on true CTA scans $y$ and the synthetic scan $t^{\dag}(y_{net})$. The segmentation model (nnDet for ROI detection and nnUNet for segmentation) is found here: https://github.com/fmusio/TopCoWSubmissions/tree/main.



## Results



=========================================================
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


# Architectures
For the architectures, I used two hand-crafted baselines and for the predictions, an SwinUNetR architecture:
#### Baselines
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