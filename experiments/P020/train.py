from itertools import product
from pathlib import Path

import torch
from diffdrr.data import read
from diffdrr.drr import DRR
from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d
from pytorch_transformers.optimization import WarmupCosineSchedule
from timm.utils.agc import adaptive_clip_grad as adaptive_clip_grad_
from tqdm import tqdm

from diffpose.metrics import DoubleGeodesic, GeodesicSE3
from diffpose.registration import PoseRegressor
from torchvision import transforms
from torch import Tensor
from torchvision.transforms import Compose, Lambda, Normalize, Resize
import numpy as np
from diffdrr.pose import convert

def torch_random_noise(image, mode='gaussian', seed=None, **kwargs):
    """
    From scikit-image numpy function
    Function to add random noise of various types to a floating-point image.
    Parameters
    ----------
    image : ndarray
        Input image data. Will be converted to float.
    mode : str, optional
        One of the following strings, selecting the type of noise to add:
        - 'gaussian'  Gaussian-distributed additive noise.
        - 'localvar'  Gaussian-distributed additive noise, with specified
                      local variance at each point of `image`.
        - 'poisson'   Poisson-distributed noise generated from the data.
        - 'salt'      Replaces random pixels with max.
        - 'pepper'    Replaces random pixels with min.
        - 's&p'       Replaces random pixels with max or min.
        - 'speckle'   Multiplicative noise using out = image + n*image, where
                      n is Gaussian noise with specified mean & variance.
    seed : {None, int, `torch.Generator`}, optional
        If `seed` is None the `torch.Generator` singleton is
        used.
        If `seed` is an int, a new ``Generator`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` instance then that
        instance is used.
        This will set the random seed before generating noise,
        for valid pseudo-random comparisons.
    mean : float, optional
        Mean of random distribution. Used in 'gaussian' and 'speckle'.
        Default : 0.
    var : float, optional
        Variance of random distribution. Used in 'gaussian' and 'speckle'.
        Note: variance = (standard deviation) ** 2. Default : 0.01
    local_vars : ndarray, optional
        Array of positive floats, same shape as `image`, defining the local
        variance at every image point. Used in 'localvar'.
    amount : float, optional
        Proportion of image pixels to replace with noise on range [0, 1].
        Used in 'salt', 'pepper', and 'salt & pepper'. Default : 0.05
    salt_vs_pepper : float, optional
        Proportion of salt vs. pepper noise for 's&p' on range [0, 1].
        Higher values represent more salt. Default : 0.5 (equal amounts)
    Returns
    -------
    out : ndarray
        Output floating-point image data on range [0, 1] or [-1, 1] if the
        input `image` was unsigned or signed, respectively.
    Notes
    -----
    Speckle, Poisson, Localvar, and Gaussian noise may generate noise outside
    the valid image range. The default is to clip (not alias) these values,
    but they may be preserved by setting `clip=False`. Note that in this case
    the output may contain values outside the ranges [0, 1] or [-1, 1].
    Use this option with care.
    Because of the prevalence of exclusively positive floating-point images in
    intermediate calculations, it is not possible to intuit if an input is
    signed based on dtype alone. Instead, negative values are explicitly
    searched for. Only if found does this function assume signed input.
    Unexpected results only occur in rare, poorly exposes cases (e.g. if all
    values are above 50 percent gray in a signed `image`). In this event,
    manually scaling the input to the positive domain will solve the problem.
    The Poisson distribution is only defined for positive integers. To apply
    this noise type, the number of unique values in the image is found and
    the next round power of two is used to scale up the floating-point result,
    after which it is scaled back down to the floating-point image range.
    To generate Poisson noise against a signed image, the signed image is
    temporarily converted to an unsigned image in the floating point domain,
    Poisson noise is generated, then it is returned to the original range.
    """
    mode = mode.lower()

    # Detect if a signed image was input
    image_min = unsqueeze_as(image.amin(dim=tuple(d for d in range(len(image.shape)))), image, -1)
    image_max = unsqueeze_as(image.amax(dim=tuple(d for d in range(len(image.shape)))), image, -1)

    image = image.to(torch.float32)
    device = image.device

    if isinstance(seed, torch.Generator):
        rng = seed
    else:
        rng = torch.Generator(device)
        if seed is not None:
            rng.manual_seed(seed)

    allowedtypes = {
        'gaussian': 'gaussian_values',
        'localvar': 'localvar_values',
        'poisson': 'poisson_values',
        'salt': 'sp_values',
        'pepper': 'sp_values',
        's&p': 's&p_values',
        'speckle': 'gaussian_values'}

    kwdefaults = {
        'mean': 0.,
        'var': 0.01,
        'amount': 0.05,
        'salt_vs_pepper': 0.5,
        'local_vars': torch.zeros_like(image) + 0.01}

    allowedkwargs = {
        'gaussian_values': ['mean', 'var'],
        'localvar_values': ['local_vars'],
        'sp_values': ['amount'],
        's&p_values': ['amount', 'salt_vs_pepper'],
        'poisson_values': []}

    for key in kwargs:
        if key not in allowedkwargs[allowedtypes[mode]]:
            raise ValueError('%s keyword not in allowed keywords %s' %
                             (key, allowedkwargs[allowedtypes[mode]]))

    # Set kwarg defaults
    for kw in allowedkwargs[allowedtypes[mode]]:
        kwargs.setdefault(kw, kwdefaults[kw])

    if mode == 'gaussian':
        noise = torch.normal(kwargs['mean'],
                             kwargs['var'] ** 0.5, generator=rng).to(image)
        out = image + unsqueeze_as(noise, image, -1)

    elif mode == 'localvar':
        # Ensure local variance input is correct
        if (kwargs['local_vars'] <= 0).any():
            raise ValueError('All values of `local_vars` must be > 0.')

        # Safe shortcut usage broadcasts kwargs['local_vars'] as a ufunc
        noise = torch.normal(0.,
                             (kwargs['local_vars'] ** 0.5).to(image), generator=rng)
        out = image + unsqueeze_as(noise, image, -1)

    elif mode == 'poisson':
        # Determine number of unique values in image & calculate the next power of two
        n_vals = len(torch.unique(image))
        n_vals = 2 ** np.ceil(np.log2(n_vals))

        # Ensure image is exclusively positive
        image = ((image - image_min) / (image_max - image_min))

        # Generating noise for each unique value in image.
        # This scaling thing doesn't seem to change anything
        out = torch.poisson(image * n_vals, generator=rng) / float(n_vals)

        # Return image to original range if input was signed
        out = (out * (image_max - image_min) + image_min)

    elif mode == 'salt':
        # Re-call function with mode='s&p' and p=1 (all salt noise)
        out = torch_random_noise(image, mode='s&p', seed=rng,
                                 amount=kwargs['amount'], salt_vs_pepper=1.)

    elif mode == 'pepper':
        # Re-call function with mode='s&p' and p=1 (all pepper noise)
        out = torch_random_noise(image, mode='s&p', seed=rng,
                           amount=kwargs['amount'], salt_vs_pepper=0.)

    elif mode == 's&p':
        out = image.clone()
        p = kwargs['amount']
        q = kwargs['salt_vs_pepper']
        flipped = torch.bernoulli(torch.ones_like(image)*p, generator=rng).to(bool)
        salted = torch.bernoulli(torch.ones_like(image)*q, generator=rng).to(bool)
        peppered = ~salted
        # Indexing along the batch dimension, could not find vectorized way to do it
        for i, (t, ms, mp) in enumerate(zip(out, flipped & salted, flipped & peppered)):
            out[i, ms] = image_max[i]
            out[i, mp] = image_min[i]

    elif mode == 'speckle':
        noise = torch.normal(kwargs['mean'], kwargs['var'] ** 0.5, generator=rng).to(image)
        out = image + image * unsqueeze_as(noise, image, -1)

    else:
        raise ValueError

    return out


def unsqueeze_as(a, b, dim=0):
    n_unsqueeze = len(b.shape) - len(a.shape)
    if dim==0:
        return a[(None,)*n_unsqueeze+(...,)]
    elif dim==-1:
        return a[(...,)+(None,)*n_unsqueeze]
    else:
        raise ValueError
    

class RandomSmoothing(transforms.GaussianBlur):
    """Like GaussianBlur but with equal proba kernel size 3x3 and 5x5."""
    def __init__(self):
        kernel_size = np.random.choice([3,5],1)
        super().__init__(kernel_size.item())


class RandomOffset(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.min = -0.2
        self.max = 0.15

    def forward(self, tensor: Tensor):
        m = tensor.amax(dim=tuple(d for d in range(1, len(tensor.shape))))
        r = torch.rand_like(tensor) * (self.max - self.min) + self.min
        return tensor + unsqueeze_as(m, tensor, -1)*r


class RandomScaling(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.min = 0.8
        self.max = 1.15

    def forward(self, tensor: Tensor):
        r = torch.rand_like(tensor) * (self.max - self.min) + self.min
        return tensor * unsqueeze_as(r, tensor -1)


class RandomRenormalize(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lower_min = -0.04
        self.lower_max = 0.02
        self.upper_min = 0.9
        self.upper_max = 1.05

    def forward(self, tensor: Tensor):
        lower = unsqueeze_as(torch.rand(tensor.shape[0], device=tensor.device, dtype=tensor.dtype) * \
                (self.lower_max - self.lower_min) + self.lower_min, tensor, -1)
        upper = unsqueeze_as(torch.rand(tensor.shape[0], device=tensor.device, dtype=tensor.dtype) * \
                (self.upper_max - self.upper_min) + self.upper_min, tensor, -1)
        m = unsqueeze_as(tensor.amax(dim=tuple(d for d in range(1, len(tensor.shape)))), tensor, -1)
        return ((tensor / m) * (upper - lower) + lower) * m


class NonLinearPixelWiseOffset(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a_min = self.b_min = 0.9
        self.a_max = self.b_max = 1.05
        self.c_min = -0.4
        self.c_max = 0.4

    def forward(self, tensor: Tensor):
        a = torch.rand(tensor.shape[0], device=tensor.device, dtype=tensor.dtype) * \
            (self.a_max - self.a_min) + self.a_min
        b = torch.rand(tensor.shape[0], device=tensor.device, dtype=tensor.dtype) * \
            (self.b_max - self.b_min) + self.b_min
        c = torch.rand(tensor.shape[0], device=tensor.device, dtype=tensor.dtype) * \
            (self.c_max - self.c_min) + self.c_min
        offset = unsqueeze_as(a, tensor, -1) * torch.sin(unsqueeze_as(b, tensor, -1) * tensor + unsqueeze_as(c, tensor, -1))
        return tensor + offset


class SaltAndPepperNoise(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.min = 0.02
        self.max = 0.04

    def forward(self, tensor: Tensor):
        proportion = torch.rand(tensor.shape[0], device=tensor.device, dtype=tensor.dtype) * \
                     (self.max - self.min) + self.min
        return torch_random_noise(tensor, mode='s&p', amount=unsqueeze_as(proportion, tensor, -1))


class GaussianNoise(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.min = -0.15
        self.max = 0.1

    def forward(self, tensor: Tensor):
        mean = torch.rand(tensor.shape[0], device=tensor.device, dtype=tensor.dtype) * (self.max - self.min) + self.min
        return torch_random_noise(tensor, mode='gaussian', mean=mean)


class PoissonNoise(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor: Tensor):
        return torch_random_noise(tensor, mode='poisson')


class NonLinearPixelWiseScaling(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a_min = self.b_min = 0.8
        self.a_max = self.b_max = 1.1
        self.c_min = -0.5
        self.c_max = 0.5

    def forward(self, tensor: Tensor):
        a = torch.rand(tensor.shape[0], device=tensor.device, dtype=tensor.dtype) * \
            (self.a_max - self.a_min) + self.a_min
        b = torch.rand(tensor.shape[0], device=tensor.device, dtype=tensor.dtype) * \
            (self.b_max - self.b_min) + self.b_min
        c = torch.rand(tensor.shape[0], device=tensor.device, dtype=tensor.dtype) * \
            (self.c_max - self.c_min) + self.c_min
        scaled = unsqueeze_as(a, tensor, -1) * torch.sin(unsqueeze_as(b, tensor, -1) * tensor +
                                                         unsqueeze_as(c, tensor, -1))
        return scaled


class RandomApply(torch.nn.Module):
    """Apply randomly a list of transformations with a given probability.

    Args:
        transforms_list (sequence or torch.nn.Module): list of transformations
        p (float): probability
    """

    def __init__(self, transforms_list, p=0.5):
        super().__init__()
        self.transforms = transforms_list
        self.p = p

    def forward(self, img):
        if self.p < torch.rand(1):
            return img
        for t in self.transforms:
            if self.p > torch.rand(1):
                img = t(img)
        return img


class ClipTransform(torch.nn.Module):
    def __init__(self, min_clip=0., max_clip=None):
        super().__init__()
        self.min_clip = min_clip
        self.max_clip = max_clip

    def forward(self, img):
        return torch.clip(img, min=self.min_clip, max=self.max_clip)


# noinspection PyTypeChecker
class DRRDomainRandomizerTransform(RandomApply):
    def __init__(self, p=0.5):
        transforms_list = torch.nn.Sequential(
            torch.nn.Sequential(
                RandomSmoothing(),
                ClipTransform(min_clip=0.)
            ),
            torch.nn.Sequential(
                RandomOffset(),
                ClipTransform(min_clip=0.)
            ),
            torch.nn.Sequential(
                RandomScaling(),
                ClipTransform(min_clip=0.)
            ),
            torch.nn.Sequential(
                RandomRenormalize(),
                ClipTransform(min_clip=0.)
            ),
            torch.nn.Sequential(
                NonLinearPixelWiseOffset(),
                ClipTransform(min_clip=0.)
            ),
            torch.nn.Sequential(
                SaltAndPepperNoise(),
                ClipTransform(min_clip=0.)
            ),
            torch.nn.Sequential(
                GaussianNoise(),
                ClipTransform(min_clip=0.)
            ),
            torch.nn.Sequential(
                PoissonNoise(),
                ClipTransform(min_clip=0.)
            ),
            torch.nn.Sequential(
                NonLinearPixelWiseScaling(),
                ClipTransform(min_clip=0.)
            )
        )
        super(DRRDomainRandomizerTransform, self).__init__(transforms_list, p)


def get_random_offset(view, batch_size: int, device):
    if view == "ap":
        t1 = torch.distributions.Normal(0., 30).sample((batch_size,))
        t2 = torch.distributions.Normal(0., 30).sample((batch_size,))
        t3 = torch.distributions.Normal(0., 30).sample((batch_size,))
        r1 = torch.distributions.Normal(0., 0.1).sample((batch_size,))
        r2 = torch.distributions.Normal(0., 0.1).sample((batch_size,))
        r3 = torch.distributions.Normal(0., 0.25).sample((batch_size,))
    elif view == "lat":
        t1 = torch.distributions.Normal(0., 30).sample((batch_size,))
        t2 = torch.distributions.Normal(0., 30).sample((batch_size,))
        t3 = torch.distributions.Normal(0., 30).sample((batch_size,))
        r1 = torch.distributions.Normal(0., 0.1).sample((batch_size,))
        r2 = torch.distributions.Normal(0., 0.05).sample((batch_size,))
        r3 = torch.distributions.Normal(0., 0.05).sample((batch_size,))
    else:
        raise ValueError(f"view must be 'ap' or 'lat', not '{view}'")

    log_R_vee = torch.stack([r1, r2, r3], dim=1).to(device)
    log_t_vee = torch.stack([t1, t2, t3], dim=1).to(device)
    return convert(
        log_R_vee,
        log_t_vee,
        parameterization="se3_log_map",
    )

def load(data_dir, orientation, subsample, device):
    # Load the subject
    data_dir = Path(data_dir)
    path = data_dir / "P020"
    vol_name = "P020_CT"
    vol_file = path / f"{vol_name}.nii.gz"
    subject = read(vol_file, 
               labelmap=None,
               orientation=orientation) 

    from diffdrr.pose import RigidTransform
    isocenter_pose = RigidTransform(torch.tensor([[[ 0.0000e+00,  1.0000e+00,  0.0000e+00,  8.9961e+02],
         [ 0.0000e+00, -2.6450e-16, -1.0000e+00, -2.9212e+01],
         [-1.0000e+00,  1.1921e-07,  0.0000e+00,  1.8550e+01],
         [ 0.0000e+00,  1.4533e-18,  0.0000e+00,  1.0000e+00]]]).to(device))

    # Make the DRR
    SDD = 1293
    height = 500
    width = 500
    delx =  0.6
    dely = 0.6
    reverse_x_axis = True
    height //= subsample
    width //= subsample
    delx *= subsample
    dely *= subsample

    drr = DRR(
        subject,
        sdd=SDD,
        height=height,
        delx=delx,
        reverse_x_axis=reverse_x_axis
    ).to(device)
    
    from os.path import join
    from skimage import io
    rotate_gt = False
    flipH=False
    flipV=True
    
    mask_file = join(path, "P020_CBCT1_173_mask.png")
    # mask image width
    mask = torch.ones((1,1,height,width), device=device)
    mask_amount_x = [int(height*0.03), int(height*0.03)]
    mask[..., :, :mask_amount_x[0]] = 0
    mask[..., :, -mask_amount_x[1]:] = 0
    # mask image height
    mask_amount_y = [int(height*0.03), int(height*0.1)]
    mask[..., :mask_amount_y[0], :] = 0
    mask[..., -mask_amount_y[1]:, :] = 0
    if mask_file is not None:
        add_mask = io.imread(mask_file)
        if len(add_mask.shape) > 2:
            add_mask = add_mask.sum(-1)
        add_mask = add_mask.T
        if flipH:
            add_mask = add_mask[::-1]
        if flipV:
            add_mask = add_mask[:, ::-1]
        add_mask = torch.as_tensor(add_mask.copy().astype(np.float32)).to(mask)
        add_mask = torch.nn.functional.interpolate(add_mask[None,None], (height, height)).squeeze()
        add_mask = (add_mask > 0).to(float)
        add_mask = 1 - add_mask
        mask = (mask * add_mask).to(torch.float32)    
    
    transform = Compose([Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)),
                         DRRDomainRandomizerTransform(),
                         Lambda(lambda x: x * mask)])

    return drr, isocenter_pose, transform


def train(
    model,
    optimizer,
    scheduler,
    drr,
    transform,
    isocenter_pose,
    batch_size,
    n_epochs,
    n_batches_per_epoch,
    model_params,
    orientation,
    device,
):  
    view = orientation.lower()
    # Random pose perturbation is independent from
    if view == 'pa':
        view = 'ap'
    metric = MultiscaleNormalizedCrossCorrelation2d(eps=1e-4)
    geodesic = GeodesicSE3()
    double = DoubleGeodesic(drr.detector.sdd/2)

    best_loss = torch.inf

    model.train()
    for epoch in range(n_epochs):
        losses = []
        for _ in (itr := tqdm(range(n_batches_per_epoch), leave=False)):
            try:
                offset = get_random_offset(view, batch_size, device)
                pose = isocenter_pose.compose(offset)
                img = drr(pose)
                img = transform(img)
    
                pred_offset = model(img)
                pred_pose = isocenter_pose.compose(pred_offset)
                pred_img = drr(pred_pose)
                pred_img = transform(pred_img)
    
                ncc = metric(pred_img, img)
                log_geodesic = geodesic(pred_pose, pose)
                geodesic_rot, geodesic_xyz, double_geodesic = double(pred_pose, pose)
                loss = 1 - ncc + 1e-2 * (log_geodesic + double_geodesic)
    
                optimizer.zero_grad()
                loss.mean().backward()
                adaptive_clip_grad_(model.parameters())
                optimizer.step()
                scheduler.step()
    
                losses.append(loss.mean().item())
    
                # Update progress bar
                itr.set_description(f"Epoch [{epoch}/{n_epochs}]")
                itr.set_postfix(
                    geodesic_rot=geodesic_rot.mean().item(),
                    geodesic_xyz=geodesic_xyz.mean().item(),
                    geodesic_dou=double_geodesic.mean().item(),
                    geodesic_se3=log_geodesic.mean().item(),
                    loss=loss.mean().item(),
                    ncc=ncc.mean().item(),
                )
    
                prev_pose = pose
                prev_pred_pose = pred_pose
            except:
                print("Aaaaaaand we've crashed...")
                print(ncc)
                print(log_geodesic)
                print(geodesic_rot)
                print(geodesic_xyz)
                print(double_geodesic)
                print(pose.get_matrix())
                print(pred_pose.get_matrix())
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "height": drr.detector.height,
                        "width": drr.detector.width,
                        "epoch": epoch,
                        "batch_size": batch_size,
                        "n_epochs": n_epochs,
                        "n_batches_per_epoch": n_batches_per_epoch,
                        "pose": pose.get_matrix().cpu(),
                        "pred_pose": pred_pose.get_matrix().cpu(),
                        **model_params,
                    },
                    f"checkpoints/specimen_{orientation}_crashed.ckpt",
                )
                raise RuntimeError("NaN loss")

        losses = torch.tensor(losses)
        tqdm.write(f"Epoch {epoch + 1:04d} | Loss {losses.mean().item():.4f}")
        if losses.mean() < best_loss and not losses.isnan().any():
            best_loss = losses.mean().item()
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "height": drr.detector.height,
                    "width": drr.detector.width,
                    "epoch": epoch,
                    "loss": losses.mean().item(),
                    "batch_size": batch_size,
                    "n_epochs": n_epochs,
                    "n_batches_per_epoch": n_batches_per_epoch,
                    **model_params,
                },
                f"checkpoints/specimen_{orientation}_best.ckpt",
            )

        if epoch % 25 == 0 and epoch != 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "height": drr.detector.height,
                    "width": drr.detector.width,
                    "epoch": epoch,
                    "loss": losses.mean().item(),
                    "batch_size": batch_size,
                    "n_epochs": n_epochs,
                    "n_batches_per_epoch": n_batches_per_epoch,
                    **model_params,
                },
                f"checkpoints/specimen_{orientation}_epoch{epoch:03d}.ckpt",
            )

def main(
    data_dir, 
    orientation,
    subsample=2,
    restart=None,
    model_name="resnet18",
    parameterization="se3_log_map",
    convention=None,
    lr=1e-3,
    batch_size=1,
    n_epochs=1000,
    n_batches_per_epoch=100,
):
    device = torch.device("cuda")
    drr, isocenter_pose, transform = load(data_dir, orientation, subsample, device)

    model_params = {
        "model_name": model_name,
        "parameterization": parameterization,
        "convention": convention,
        "norm_layer": "groupnorm",
    }
    model = PoseRegressor(**model_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if restart is not None:
        ckpt = torch.load(restart)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    model = model.to(device)

    scheduler = WarmupCosineSchedule(
        optimizer,
        5 * n_batches_per_epoch,
        n_epochs * n_batches_per_epoch - 5 * n_batches_per_epoch,
    )

    train(
        model,
        optimizer,
        scheduler,
        drr,
        transform,
        isocenter_pose,
        batch_size,
        n_epochs,
        n_batches_per_epoch,
        model_params,
        orientation,
        device,
    )

if __name__ == "__main__":
    Path("checkpoints").mkdir(exist_ok=True)
    data_dir = "/home/francois/Projects/data/raw_data"
    orientation = "AP"
    
    main(data_dir, orientation, n_epochs=3000)
