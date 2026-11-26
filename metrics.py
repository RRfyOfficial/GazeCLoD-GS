#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
import csv

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def readImages2(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname.split("_")[0] + ".png")
        print(gt_dir / fname.split("_")[0] + ".png")
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def readImages3(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    #00000.png can match 00000_rvd1_psnr17.31_3026717_3026717_1000_opa_noDecayTrain_time0.06s.png
    for gt_path in gt_dir.glob("*.png"):
        stem = gt_path.stem                 # 00000
        # set a loose glob pattern to match the render image
        glob_pattern = f"{stem}_rvd1_psnr*_*_*_1000_opa_noDecayTrain_time*.png"
        matches = list(renders_dir.glob(glob_pattern))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"No render image matches pattern {glob_pattern}"
            )
        # if multiple matches, you can change as needed, here we take the lexicographically smallest one
        render_path = sorted(matches)[0]
        render = Image.open(render_path)
        gt = Image.open(gt_path)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(os.path.basename(render_path))
    return renders, gts, image_names

def readImages_scale1(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for gt_path in gt_dir.glob("*.png"):
        stem = gt_path.stem                 # 00000
        # set a loose glob pattern to match the render image
        glob_pattern = f"{stem}_rvd1_psnr*_*_*_*_decayOpa_decayTrain_time*.png"
        matches = list(renders_dir.glob(glob_pattern))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"No render image matches pattern {glob_pattern}"
            )
        # if multiple matches, you can change as needed, here we take the lexicographically smallest one
        render_path = sorted(matches)[0]
        render = Image.open(render_path)
        gt = Image.open(gt_path)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(os.path.basename(render_path))
    return renders, gts, image_names

def evaluate(model_paths, enable_scale1):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    all_scenes =["360v2_bicycle", "360v2_bonsai", "360v2_counter", "360v2_flowers", "360v2_garden", 
    "360v2_kitchen", "360v2_room", "360v2_stump", "360v2_treehill", "bungeenerf_amsterdam", 
    "bungeenerf_barcelona", "bungeenerf_bilbao", "bungeenerf_chicago", "bungeenerf_hollywood", 
    "bungeenerf_pompidou", "bungeenerf_quebec", "bungeenerf_rome", "tant_db_drjohnson", "tant_db_playroom",
     "tant_db_train", "tant_db_truck"] 

    for scene in all_scenes:
        scene_dir = f"{model_paths}/{scene}"
        # try:
        print("Scene:", scene_dir)
        full_dict[scene] = {}
        per_view_dict[scene] = {}
        full_dict_polytopeonly[scene] = {}
        per_view_dict_polytopeonly[scene] = {}

        test_dir = Path(scene_dir) / "test"
        method = "ours_30000"
        print("Method:", method)

        full_dict[scene][method] = {}
        per_view_dict[scene][method] = {}
        full_dict_polytopeonly[scene][method] = {}
        per_view_dict_polytopeonly[scene][method] = {}

        method_dir = test_dir / method
        gt_dir = method_dir/ "gt"
        renders_dir = method_dir / "renders3"
        if enable_scale1:
            renders, gts, image_names = readImages_scale1(renders_dir, gt_dir)
        else:
            renders, gts, image_names = readImages3(renders_dir, gt_dir)

        ssims = []
        psnrs = []
        lpipss = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

        print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
        print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
        print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
        print("")

        full_dict[scene][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                "PSNR": torch.tensor(psnrs).mean().item(),
                                                "LPIPS": torch.tensor(lpipss).mean().item()})
        per_view_dict[scene][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

        with open(scene_dir + "/results.json", 'w') as fp:
            json.dump(full_dict[scene], fp, indent=True)
        with open(scene_dir + "/per_view.json", 'w') as fp:
            json.dump(per_view_dict[scene], fp, indent=True)
        # except:
        #     print("Unable to compute metrics for model", scene_dir)

def evaluate2(model_paths_list):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    for model_paths in model_paths_list:
        for scene_dir in os.listdir(model_paths):
            try:
                print("Scene:", scene_dir)
                full_dict[scene_dir] = {}
                per_view_dict[scene_dir] = {}
                full_dict_polytopeonly[scene_dir] = {}
                per_view_dict_polytopeonly[scene_dir] = {}

                test_dir = model_paths/Path(scene_dir) / "test"
                print("Test directory:", test_dir)  

                full_dict[scene_dir] = {}
                per_view_dict[scene_dir] = {}
                full_dict_polytopeonly[scene_dir] = {}
                per_view_dict_polytopeonly[scene_dir] = {}

                method_dir = test_dir / "ours_90000"
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                print("Renders directory:", renders_dir)
                renders, gts, image_names = readImages2(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")

                full_dict[scene_dir].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

                with open(scene_dir + "/results.json", 'w') as fp:
                    json.dump(full_dict[scene_dir], fp, indent=True)
                with open(scene_dir + "/per_view.json", 'w') as fp:
                    json.dump(per_view_dict[scene_dir], fp, indent=True)
            except:
                print("Unable to compute metrics for model", scene_dir)


def collect_results(root_folder):
    results = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        if 'results.json' in filenames:
            scene_name = os.path.basename(dirpath)
            with open(os.path.join(dirpath, 'results.json'), 'r') as f:
                data = json.load(f)
                if 'ours_30000' in data:
                    metrics = data['ours_30000']
                    results.append({
                        'Scene': scene_name,
                        'SSIM': metrics['SSIM'],
                        'PSNR': metrics['PSNR'],
                        'LPIPS': metrics['LPIPS']
                    })
    return results

def write_to_csv(results, output_file):
    fieldnames = ['Scene', 'SSIM', 'PSNR', 'LPIPS']
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    # parser.add_argument('--model_paths', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--model_path', '-m', required=True, type=str)
    parser.add_argument('--enable_scale1', action='store_true', help="是否启用scale=1的评估")

    args = parser.parse_args()
    evaluate(args.model_path, args.enable_scale1)
    # example usage:
    output_file = 'combined_results.csv'  # name of the output CSV file
    results = collect_results(args.model_path)
    write_to_csv(results, output_file)

    # evaluate2(args.model_paths)
# python metrics.py -m /home/czg/work/results/results/gs
# python metrics.py -m /home/czg/work/results/results/gs
# python metrics.py -m /home/czg/work/results/results/CullAllScale1b1start5000_reg --enable_scale1