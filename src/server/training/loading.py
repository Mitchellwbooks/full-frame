import torch
from PIL import Image
from torchvision.transforms import transforms


def download():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    model.eval()
    return model


def convert_to_onnx(model):
    import urllib
    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    torch.onnx.export(
        model,  # model being run
        input_batch,  # model input (or a tuple for multiple inputs)
        "../onnx_models/resnet_50.onnx",  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=13,  # the ONNX version to export the model to
        do_constant_folding=False,  # whether to execute constant folding for optimization
        input_names=['input'],  # the model's input names
        output_names=['output'],  # the model's output names
        dynamic_axes={
            'input': {
                0: 'batch_size'
            },  # variable length axes
            'output': {
                0: 'batch_size'
            }
        }
    )


if __name__ == '__main__':
    model = download()
    convert_to_onnx( model )
