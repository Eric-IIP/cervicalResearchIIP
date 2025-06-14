import torch
import numpy as np
from PIL import Image

def predict(img,
            model,
            preprocess,
            postprocess,
            device,
            ):
    model.eval()
    img = preprocess(img)  # preprocess image
    x = torch.from_numpy(img).to(device)  # to torch, send to device
    with torch.no_grad():
        out = model(x)  # send through model/network
    out_softmax = torch.softmax(out, dim=1)  # perform softmax on outputs
    result = postprocess(out_softmax)  # postprocess outputs

    return result

def predict_mrcnn(img,
                  model,
                  preprocess,
                  postprocess,
                  device,
                  score_threshold=0.5):
    if img is None:
        raise ValueError("Input image is None. Check if the path is correct and image is readable.")
    model.eval()

    # preprocess
    x = torch.from_numpy(img).float().to(device)
    if x.ndim == 2:
        x = x.unsqueeze(0).repeat(3, 1, 1)  # [H, W] → [3, H, W]
    elif x.shape[0] == 1:
        x = x.repeat(3, 1, 1)              # [1, H, W] → [3, H, W]

    # inference
    with torch.no_grad():
        output = model([x])

    output = output[0]
    
    masks = output['masks'] > 0.5  # shape: [N, 1, H, W]
    masks = masks.squeeze(1)       # shape: [N, H, W]
    labels = output['labels']      # shape: [N]

    if len(masks) == 0:
        # No predictions, return empty mask
        mask_result = np.zeros((img.shape[1], img.shape[2]), dtype=np.uint8)

    # Combine all masks into one labeled mask
    instance_mask = torch.zeros_like(masks[0], dtype=torch.uint8)
    for i, mask in enumerate(masks):
        class_label = labels[i].item()
        instance_mask[mask] = class_label

    # Convert to NumPy and append
    instance_mask_np = instance_mask.cpu().numpy().astype(np.uint8)
    return [instance_mask_np]

