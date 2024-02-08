import pathlib
import subprocess

import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch


def prepare_ckpt_weights(target_dir: str = "./unetmobv2_weights") -> pathlib.Path:
    """
    Download the model weights if it doesn't exist
    Args:
        target_dir: target directory for the model
    Returns:
        pathlib.Path: path to the model
    """
    # Download the model if it doesn't exist
    pathlib.Path(target_dir).mkdir(exist_ok=True)
    filename = pathlib.Path(target_dir, "unetmobv2.ckpt")
    if not filename.is_file():
        # download `.ckpt` file from Dropbox
        url = "https://www.dropbox.com/scl/fi/ehkpfck7ref06vvrth1kw/UNetMobV2.ckpt?rlkey=2joyddxg36r1haut1cxwkdhwx&dl=0"
        subprocess.run(["wget", "--no-check-certificate", url, "-O", filename.as_posix()])
    return filename


def model_setup():
    """
    Load the model
    Args:
        weights_path: path to the model
    Returns:
        pl.LightningModule: model
    """

    class UnetMobV2Class(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.model = smp.Unet(
                encoder_name="mobilenet_v2",
                encoder_weights=None,
                in_channels=13,
                classes=4,
            )

        def forward(self, x):
            return self.model(x)

    # Load weights
    weights_path = prepare_ckpt_weights("./unetmobv2_weights")
    # Load the model
    map_location = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = UnetMobV2Class.load_from_checkpoint(weights_path.as_posix(), map_location=map_location)
    model.eval()
    return model


# It is from CloudSen12 Masky library...
def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """
    # make X at least 2d
    y = np.atleast_2d(X)
    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
    # multiply y against the theta parameter,
    y = y * float(theta)
    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)
    # exponentiate y
    y = np.exp(y)
    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)
    # finally: divide elementwise
    p = y / ax_sum
    # flatten if X was 1D
    if len(X.shape) == 1:
        p = p.flatten()
    return p
