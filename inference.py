import torch


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
            score_threshold = 0.5
            ):
    model.eval()
    x = preprocess(img).to(device)  # preprocess image
    with torch.no_grad():
        out = model([x])  # send through model/network
    pred = out[0]
    
    result = {
        "boxes": pred["boxes"].cpu(),
        "labels": pred["labels"].cpu(),
        "scores": pred["scores"].cpu(),
        "masks": (pred["masks"] > 0.5).squeeze(1).cpu()  # [N, H, W]
    }
    
    # Filter by score threshold
    keep = result["scores"] > score_threshold
    for k in result:
        result[k] = result[k][keep]
        
    result = postprocess(result) 

    return result