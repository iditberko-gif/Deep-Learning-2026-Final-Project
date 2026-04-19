import argparse
import os
import time
import json
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from Image_Processing.biggest_shadow import get_molecular
from Image_Processing.resize_img import resize_img520

from deeplab_v3.src import deeplabv3_resnet50
from yolov5.utils.general import check_requirements


@torch.no_grad()
def run(
    weights="./weights/model.pth",
    input_path="./yolov5/runs/detect/exp/crops/exist_molecular",
    palette_path="./deeplab_v3/palette.json",
):
    aux = False  # inference time not need aux_classifier
    classes = 1
    weights_path = weights
    path = input_path

    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Input folder not found: {path}. Expected YOLO crops in '<run_dir>/crops/exist_molecular'."
        )
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not os.path.exists(palette_path):
        raise FileNotFoundError(f"Palette file not found: {palette_path}")

    paths = os.listdir(path)
    if len(paths) == 0:
        raise FileNotFoundError(
            f"No images found in input folder: {path}."
        )

    for img_name in paths:
        img_path = os.path.join(path, img_name)

        assert os.path.exists(weights_path), f"weights {weights_path} not found."
        assert os.path.exists(img_path), f"image {img_path} not found."
        assert os.path.exists(palette_path), f"palette {palette_path} not found."

        with open(palette_path, "rb") as f:
            pallette_dict = json.load(f)
            pallette = []
            for v in pallette_dict.values():
                pallette += v

        # get devices
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("using {} device.".format(device))

        # create model
        model = deeplabv3_resnet50(aux=aux, num_classes=classes + 1)

        # delete weights about aux_classifier
        weights_dict = torch.load(weights_path, map_location="cpu")["model"]
        for k in list(weights_dict.keys()):
            if "aux" in k:
                del weights_dict[k]

        # load weights
        model.load_state_dict(weights_dict)
        model.to(device)

        # load image
        original_img = Image.open(img_path).convert("RGB")

        # from pil image to tensor and normalize
        data_transform = transforms.Compose(
            [
                transforms.Resize(520),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        model.eval()
        with torch.no_grad():
            # init model
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            t_start = time_synchronized()
            output = model(img.to(device))
            t_end = time_synchronized()
            print("inference+NMS time: {}".format(t_end - t_start))

            prediction = output["out"].argmax(1).squeeze(0)
            prediction = prediction.to("cpu").numpy().astype(np.uint8)
            mask = Image.fromarray(prediction)
            mask.putpalette(pallette)
            # mask_path= ".//mask_img"
            if not os.path.exists(".//mask_img"):
                os.makedirs(".//mask_img")

            mask.save("./mask_img/{}.png".format(img_name))

    # The size of the chemical molecular structure map trimmed by YOLOV5 was consistent with the size of the mask obtained by deeplabv3
    if not os.path.exists(".//resize_img"):
        os.makedirs(".//resize_img")
    resize_img520(path, "./resize_img")


def parse_opt():
    parser = argparse.ArgumentParser(description="pytorch deeplabv3 segmentation only")

    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Detect experiment folder containing crops/exist_molecular",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="./weights/model.pth",
        help="Path to DeepLab-v3 weights",
    )
    parser.add_argument(
        "--palette-path",
        type=str,
        default="./deeplab_v3/palette.json",
        help="Path to color palette JSON",
    )

    opt = parser.parse_args()

    if not os.path.isdir(opt.run_dir):
        raise FileNotFoundError(f"Run directory not found: {opt.run_dir}")

    opt.input = os.path.join(opt.run_dir, "crops", "exist_molecular")
    if not os.path.isdir(opt.input):
        raise FileNotFoundError(
            f"Expected crops folder not found: {opt.input}"
        )

    return opt


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main(opt):
    check_requirements(exclude=("tensorboard", "thop"))
    run(
        weights=opt.weights,
        input_path=opt.input,
        palette_path=opt.palette_path,
    )

    # To get the mask of figure and generate the noise without noise chemical molecular structure
    get_molecular("./resize_img", "./mask_img", 0.5)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
