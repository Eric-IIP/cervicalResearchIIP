import cv2
import os
import numpy as np

def save_image_unique(filename, image):
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename

    # Keep incrementing until a non-existing filename is found
    while os.path.exists(new_filename):
        new_filename = f"{base}{counter}{ext}"
        counter += 1

    cv2.imwrite(new_filename, image)
    return new_filename

# def tensor_to_image(tensor):
#     # Assume tensor is shape [C, H, W] or [1, C, H, W]
#     if tensor.dim() == 4:
#         tensor = tensor[0]  # remove batch dim
#     tensor = tensor[0]
#     if tensor.shape[0] == 1:
#         tensor = tensor.squeeze(0)  # grayscale [H, W]
#     else:
#         tensor = tensor.permute(1, 2, 0)  # to [H, W, C]

#     # Normalize to [0, 255] and convert to uint8
#     tensor = tensor.detach().cpu().numpy()
#     tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8) * 255
#     return tensor.astype(np.uint8)

# def tensor_to_image(tensor):
#     """
#     Converts a PyTorch tensor to a uint8 NumPy image that cv2.imwrite can save.
#     Works with grayscale or RGB (first 3 channels) tensors.
#     """
#     if tensor.dim() == 4:
#         tensor = tensor[0]  # Remove batch dimension -> [C, H, W]

#     if tensor.dim() == 3:
#         c, h, w = tensor.shape
#         if c == 1:
#             tensor = tensor.squeeze(0)  # [H, W], grayscale
#         elif c >= 3:
#             tensor = tensor[:3]  # take first 3 channels
#             tensor = tensor.permute(1, 2, 0)  # [H, W, C]
#         else:
#             tensor = tensor[0]  # fallback to first channel

#     elif tensor.dim() == 2:
#         pass  # already [H, W], grayscale

#     else:
#         raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

#     tensor = tensor.detach().cpu().numpy()
#     #tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8) * 255
#     return tensor.astype(np.uint8)


def tensor_to_image(tensor):
    """
    Converts a segmentation mask tensor [H, W] to a NumPy uint8 or uint16 image.
    """
    if tensor.dim() == 3:  # [B, H, W]
        tensor = tensor[0]
    mask = tensor.detach().cpu().numpy()
    return mask.astype(np.uint8 if mask.max() <= 255 else np.uint16)
