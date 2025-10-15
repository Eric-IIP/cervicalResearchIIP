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
        outs = model(x)  # send through model/network
    out_softmax = []
    for out in outs:
        out_softmax_i = torch.softmax(out, dim=1)  # perform softmax on outputs
        out_softmax.append(out_softmax_i)
    # for debugging purposes, print unique predicted labels for each decoder
    # for i, o in enumerate(out_softmax):
    #         preds = torch.argmax(o, dim=1)  # [B, H, W]
    #         unique_vals = torch.unique(preds)
    #         print(f"Decoder {i} unique predicted labels:", unique_vals.cpu().numpy())
    result = postprocess(out_softmax)  # postprocess outputs

    return result
