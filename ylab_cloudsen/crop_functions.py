from typing import List, Union

import numpy as np
import numpy.typing as npt
import xarray as xr


def prepare_indices_one_dim(dim: int, patch_size: int, overlap: int) -> npt.NDArray:
    """
    We want to generate overlapping patches. Then after processing we want to reassemble these patches to
    the size of the original image. In doing so, we need to remove the intersections.

    Let's imagine an image with a width of 100x1px. It is divided horizontally into 2 patches 60px wide.
    The intersection is 20px. In this case, X indices of the top left corners of the patches = 0 and 40.
    To reassemble the image back after processing the patches, we have to take the first 50 pixels from
    the first patch and place them at indices 0 to 49. And the last 50 pixels from the second image and
    place them at positions from 50 to 99.

    To do this, for each patch we need:
    -- The original coordinates of the top left corner.
    -- The indexes of the first pixel of the patch (part of the patch) that will
       participate in the reconstructed image = inds_p_s.
    -- the length of the patch section (starting from inds_p_s) that will be used for reconstruction
    -- The upper left index of the final image from which the selected part of the patch will be inserted.

    So we need 4 indices for height and width, for a total of 10.
    This function returns an array nx4, where n -- the number of patches along one dimension.
    In the rows of the matrix, there are 5 indices:
    1. H or W of the upper left corner of the patch in the original image.
    2. H or W index on the patch -- the beginning of the patch segment to be used
       for reconstruction.
    3. Length of the patch section -- the end of the patch segment to be used for restoration.
    4. H or W index of the top left corner pixel of the original image
       from which the selected patch section will be placed.
    5. Also we will add patch size (H or W)
    """
    # Top left corner indices (point 1 in description) Index_top_left = inds_tl

    inds_tl = np.arange(0, dim, patch_size - overlap, dtype=int)
    # Check if last patch is not necessary, i.e. only overlap left
    if (inds_tl[-2] + patch_size) >= dim:
        inds_tl = inds_tl[:-1]
    # Check last patch and move its start index such that right border = dim_x
    inds_tl[-1] = inds_tl[-1] if inds_tl[-1] + patch_size < (dim - 1) else (dim - patch_size)
    # H or W index of the top left corner pixel of the original image
    # from which the selected patch section will be placed.
    # Index_top_left_reconstruction = inds_tl_r
    inds_tl_r = inds_tl + overlap / 2
    inds_tl_r[0] = 0
    inds_tl_r[-1] = inds_tl_r[-2] + patch_size - overlap
    # Just in case reconstruction indices of the bottom right corner
    inds_br_r = inds_tl_r + patch_size - overlap
    inds_br_r[0] += overlap // 2
    inds_br_r[-1] = dim
    # H or W index on the patch -- the beginning of the patch segment to be used
    # for reconstruction. Start index of patch = inds_p_s
    inds_p_s = np.full(len(inds_tl), overlap // 2)
    inds_p_s[0] = 0
    # deal with last patch
    inds_p_s[-1] = (inds_tl[-2] + patch_size - overlap / 2) - inds_tl[-1]
    # Length of the patch section -- the end of the patch segment to be used for restoration.
    #  patch section length
    patch_sl = (patch_size - overlap // 2) - inds_p_s
    patch_sl[-1] = patch_size - inds_p_s[-1]
    # Patch size array
    patch_s = np.repeat(patch_size, len(inds_tl))
    res = np.vstack((inds_tl, inds_tl_r, inds_p_s, patch_sl, patch_s)).astype(int)
    return res.T


def prepare_indices_2d(
    dims: Union[int, tuple[int, int]],
    patch_size: Union[int, tuple[int, int]],
    overlap: Union[int, tuple[int, int]],
) -> npt.NDArray:
    """
    Combines an output of `prepare_indices_one_dim` for two dimensions
    The function returns an array of length Nx8, in each row there are indices
    along y and x dimension.
    If one row of the returned array is `ind`, and the image is `img` then
      To crop a 2d Patch:
        patch = img[ind[0] : ind[0] + ind[8], ind[1] : ind[1] + ind[9]
      To get the desired portion of the patch:
        pp = patch[ind[4] : ind[4] + ind[6], ind[5] : ind[5] + ind[7]]
      To place patch portion back to initial image:
        img[ind[2] : ind[2] + ind[6], ind[3] : ind[3] + ind[7]]

    """
    # Check types and initiate vars
    # Dimensions
    if isinstance(dims, int):
        dim_y, dim_x = dims, dims
    elif isinstance(dims, tuple) and len(dims) == 2 and all(isinstance(i, int) for i in dims):
        dim_y, dim_x = dims
    else:
        raise ValueError("Invalid input type for `dims`. Please pass either an integer or a tuple of two integers.")
    # Patch size
    if isinstance(patch_size, int):
        patch_s_y, patch_s_x = patch_size, patch_size
    elif isinstance(patch_size, tuple) and len(patch_size) == 2 and all(isinstance(i, int) for i in patch_size):
        patch_s_y, patch_s_x = patch_size
    else:
        raise ValueError(
            "Invalid input type for `patch_size`. Please pass either an integer or a tuple of two integers."
        )
    # Overlap size
    if isinstance(overlap, int):
        overlap_y, overlap_x = overlap, overlap
    elif isinstance(overlap, tuple) and len(overlap) == 2 and all(isinstance(i, int) for i in overlap):
        overlap_y, overlap_x = overlap
    else:
        raise ValueError("Invalid input type for `overlap`. Please pass either an integer or a tuple of two integers.")

    # Generate patches indices
    inds_x = prepare_indices_one_dim(dim_x, patch_s_x, overlap_x)
    inds_y = prepare_indices_one_dim(dim_y, patch_s_y, overlap_y)
    # Now we need to expand x, y indices into 2D combination
    n_rows = inds_y.shape[0]
    n_cols = inds_x.shape[0]
    inds_x_exp = np.vstack((inds_x,) * n_rows)
    inds_y_exp = np.repeat(inds_y, n_cols, 0)
    inds_xy = np.zeros((n_rows * n_cols, 10), dtype=int)
    inds_xy[:, 0::2] = inds_y_exp
    inds_xy[:, 1::2] = inds_x_exp
    return inds_xy


def indices_to_batches(inds: npt.NDArray, batch_s: int) -> List[npt.NDArray]:
    """
    Split indices into batches
    """
    blocks = []
    length = inds.shape[0]
    block_start = np.arange(0, length, batch_s)
    for s in block_start:
        blocks.append(inds[s : s + batch_s, :])
    return blocks


def crop_patches(img: xr.DataArray, inds: npt.NDArray) -> npt.NDArray:
    """
    Get image patches basing on indices
    """
    # Check, that input array is 3D
    assert len(img.shape) == 3, f"The input array should be 3D, but {len(img.shape)}D provided"
    # ! Be aware, in xr.DataArray bands go first, i.e. the shape is (bands, y, x)
    bands_n = img.shape[0]
    # Get batch size:
    batch_s = inds.shape[0]
    # Prepare list of cropped patches
    patches_l = [img[:, ind[0] : ind[0] + ind[8], ind[1] : ind[1] + ind[9]] for ind in inds]
    patch_y_s, patch_x_s = patches_l[0].shape[1:3]  # Yet again, bands go first
    # Convert patches into array of shape (batch_s, patch_y_s, patch_x_s, :)
    patches = np.stack(patches_l, axis=0)
    # Prepare target output shape tuple
    out_shape = (batch_s, bands_n, patch_y_s, patch_x_s)
    patches = np.reshape(patches, out_shape)
    return patches


def insert_patches(img: npt.NDArray, patches: npt.NDArray, inds: npt.NDArray):
    """
    Reconstruct an image given patches and the indices object
    """
    for i in range(inds.shape[0]):
        ind = inds[i, :]
        patch = patches[i]
        if len(img.shape) == 3:
            img[:, ind[2] : ind[2] + ind[6], ind[3] : ind[3] + ind[7]] = patch[
                :, ind[4] : ind[4] + ind[6], ind[5] : ind[5] + ind[7]
            ]
        elif len(img.shape) == 2:
            img[ind[2] : ind[2] + ind[6], ind[3] : ind[3] + ind[7]] = patch[
                ind[4] : ind[4] + ind[6], ind[5] : ind[5] + ind[7]
            ]
        else:
            raise AttributeError(f"An `img` array should be 2D or 3D, instead got {len(img.shape)}D")
