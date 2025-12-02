import torch
import numpy as np

def predict(img, model, preprocess, postprocess, device):
    model.eval()

    # Preprocess input for UNet
    img = preprocess(img)  # [1, C, H, W]
    x = torch.from_numpy(img).to(device)

    with torch.no_grad():
        logits = model(x)  # [1, 11, H, W]

    # Softmax for probability maps
    prob = torch.softmax(logits, dim=1)  # [1, 11, H, W]

    # Raw label prediction
    pred = torch.argmax(prob, dim=1).cpu().numpy()[0]  # [H, W]

    # Apply postprocess (convert prob → image for Stage-1 output only)
    result = postprocess(prob)  # same behavior as before

    # ---- Return what each stage needs ----
    # result → for saving PNG (Stage-1 output usage)
    # prob → for Stage-2 input
    # pred → class mask if needed
    return result, prob.cpu(), pred
