import gc
import h5py
import math
import matplotlib.pyplot as plt
import os
import psutil
import sys
import torch
import torch.nn.init as init
import uuid
import pytorch_msssim as py_ms
from math import log10
from PIL import Image
import torchvision.utils as vutils
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def text_to_array(text, width=640, height=40):
    """
    Creates a numpy array of shape height x width x 3 with
    text written on it using PIL

    Args:
        text (str): text to write
        width (int, optional): Width of the resulting array. Defaults to 640.
        height (int, optional): Height of the resulting array. Defaults to 40.

    Returns:
        np.ndarray: Centered text
    """
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (width, height), (255, 255, 255))
    try:
        font = ImageFont.truetype("UnBatang.ttf", 25)
    except OSError:
        font = ImageFont.load_default()

    d = ImageDraw.Draw(img)
    text_width, text_height = d.textsize(text)
    h = 40 // 2 - 3 * text_height // 2
    w = width // 2 - text_width
    d.text((w, h), text, font=font, fill=(30, 30, 30))
    return np.array(img)


def all_texts_to_array(texts, width=640, height=40):
    """
    Creates an array of texts, each of height and width specified
    by the args, concatenated along their width dimension

    Args:
        texts (list(str)): List of texts to concatenate
        width (int, optional): Individual text's width. Defaults to 640.
        height (int, optional): Individual text's height. Defaults to 40.

    Returns:
        list: len(texts) text arrays with dims height x width x 3
    """
    return [text_to_array(text, width, height) for text in texts]


def all_texts_to_tensors(texts, width=640, height=40):
    """
    Creates a list of tensors with texts from PIL images

    Args:
        texts (list(str)): texts to write
        width (int, optional): width of individual texts. Defaults to 640.
        height (int, optional): height of individual texts. Defaults to 40.

    Returns:
        list(torch.Tensor): len(texts) tensors 3 x height x width
    """
    arrays = all_texts_to_array(texts, width, height)
    arrays = [array.transpose(2, 0, 1) for array in arrays]
    return [torch.tensor(array) for array in arrays]


def upload_images(
    image_outputs, epoch, exp=None, im_per_row=4, rows_per_log=10, legends=[],
):
    """
    Save output image

    Args:
        image_outputs (list(torch.Tensor)): all the images to log
        im_per_row (int, optional): umber of images to be displayed per row.
            Typically, for a given task: 3 because [input prediction, target].
            Defaults to 3.
        rows_per_log (int, optional): Number of rows (=samples) per uploaded image.
            Defaults to 5.
        comet_exp (comet_ml.Experiment, optional): experiment to use.
            Defaults to None.
    """
    nb_per_log = im_per_row * rows_per_log
    n_logs = len(image_outputs) // nb_per_log + 1

    header = None
    if len(legends) == im_per_row and all(isinstance(t, str) for t in legends):
        header_width = max(im.shape[-1] for im in image_outputs)
        headers = all_texts_to_tensors(legends, width=header_width)
        header = torch.cat(headers, dim=-1)

    for logidx in range(n_logs):
        ims = image_outputs[logidx * nb_per_log : (logidx + 1) * nb_per_log]
        if not ims:
            continue
        ims = torch.stack([im.squeeze() for im in ims]).squeeze()
        image_grid = vutils.make_grid(
            ims, nrow=im_per_row, normalize=True, scale_each=True, padding=0
        )

        if header is not None:
            image_grid = torch.cat([header.to(image_grid.device), image_grid], dim=1)

        image_grid = image_grid.permute(1, 2, 0).cpu().numpy()
        exp.log_image(
            Image.fromarray((image_grid * 255).astype(np.uint8)),
            name=f"{str(epoch)}_#{logidx}",
        )


def nice_print(**kwargs):
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            print(f'Tensor "{k}" has shape {v.shape}')
        else:
            print(f'Variable "{k}" has value `{v}`')


def print_shape(o):
    dims = []
    while True:
        try:
            dims.append(len(o))
            o = next(iter(o))
        except:
            print(dims)
            return


def mem_report():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())


def cpu_stats():
    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2.0 ** 30  # memory use in GB...I think
    print("memory GB:", memoryUse)


def grouper(iterable, n):
    """Collect data into fixed-length chunks or blocks
       E.g., grouper('ABCDEFG', 3) --> ABC DEF
    """
    args = [iter(iterable)] * n
    return zip(*args)


def window(seq, size=2, stride=1):
    """Returns a sliding window (of width n) over data from the iterable
       E.g., s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...  
    """
    it = iter(seq)
    result = []
    for elem in it:
        result.append(elem)
        if len(result) == size:
            yield result
            result = result[stride:]


def draw(imgs):
    size = len(imgs)
    fig, axs = plt.subplots(2, size, figsize=(5, 5), constrained_layout=True)
    for img, ax1, ax2 in zip(imgs, axs[0], axs[1]):
        ax1.imshow(img[0])
        ax2.imshow(img[1])
    plt.show()


def weights_init(init_type="gaussian"):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find("Conv") == 0 or classname.find("Linear") == 0) and hasattr(
            m, "weight"
        ):
            # print m.__class__.__name__
            if init_type == "gaussian":
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == "default":
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


def h5_virtual_file(filenames, name="data"):
    """
    Assembles a virtual h5 file from multiples
    """
    vsources = []
    total_t = 0
    for path in filenames:
        data = h5py.File(path, "r").get(name)
        t, *features_shape = data.shape
        total_t += t
        vsources.append(h5py.VirtualSource(path, name, shape=(t, *features_shape)))

    # Assemble virtual dataset
    layout = h5py.VirtualLayout(shape=(total_t, *features_shape), dtype=data.dtype)
    cursor = 0
    for vsource in vsources:
        # we generate slices like layour[0:10,:,:,:]
        indices = (slice(cursor, cursor + vsource.shape[0]),) + (slice(None),) * (
            len(vsource.shape) - 1
        )
        layout[indices] = vsource
        cursor += vsource.shape[0]
    # Add virtual dataset to output file
    f = h5py.File(f"{uuid.uuid4()}.h5", "w", libver="latest")
    f.create_virtual_dataset(name, layout)
    return f


def psnr(Ft_p, IFrame, outputTensor=False):
    if outputTensor:
        return 10 * log10(1 / F.mse_loss(Ft_p, IFrame))
    else:
        return 10 * log10(1 / F.mse_loss(Ft_p, IFrame).item())


def ssim(Ft_p, IFrame, outputTensor=False):
    if outputTensor:
        return py_ms.ssim(Ft_p, IFrame, data_range=1, size_average=True)
    else:
        return py_ms.ssim(Ft_p, IFrame, data_range=1, size_average=True).item()
