import time
import os
import datetime
import numpy as np
import random

import torch

#torch.multiprocessing.set_sharing_strategy("file_system")

from src import deeplabv3_resnet50
from train_utils import (
    train_one_epoch,
    evaluate,
    create_lr_scheduler,
    init_distributed_mode,
    save_on_master,
    mkdir,
)
from my_dataset import VOCSegmentation
import transforms as T
from augmentations import RandomColorAugmentation, RandomBackgroundColor, RandomMoleculeColor
#from fallback_logger import get_logger

def compute_validation_loss(model, data_loader, device):
    """Compute validation loss on a data loader."""
    model.eval()
    total_loss = 0.0
    num_samples = 0
    non_blocking = device.type == "cuda"
    
    criterion = torch.nn.functional.cross_entropy
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)
            outputs = model(images)
            
            # Handle both single output and dict output (with aux)
            if isinstance(outputs, dict):
                loss = criterion(outputs["out"], targets, ignore_index=255)
                if "aux" in outputs:
                    loss += 0.5 * criterion(outputs["aux"], targets, ignore_index=255)
            else:
                loss = criterion(outputs, targets, ignore_index=255)
            
            total_loss += loss.item() * images.size(0)
            num_samples += images.size(0)
    
    return total_loss / num_samples if num_samples > 0 else 0.0

# def worker_init_fn(worker_id):
#     """Initialize worker with proper random seeds and identification."""
#     import os
#     # Set worker ID in environment for logging
#     os.environ['WORKER_ID'] = str(worker_id)
    
#     # Set random seeds - avoid torch.initial_seed() in workers as it causes issues
#     np.random.seed(worker_id)
#     random.seed(worker_id)


class SegmentationPresetTrain:
    def __init__(
        self,
        base_size,
        crop_size,
        hflip_prob=0.5,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        color_aug_prob=0.5,
        background_id=0,
        molecule_id=1,
    ):
        min_size = int(0.5 * base_size)
        #max_size = int(1.5 * base_size) #WE CHANGED IT FROM 2.0 TO 1.5 TO AVOID OOM ERROR
        max_size = int(2.0 * base_size)
        
        
        # Add color augmentation before tensor conversion
        # Randomly applies one of: A (background), B (partial molecule), or C (both)
        # trans = [
        #     RandomColorAugmentation(
        #         p=color_aug_prob,
        #         background_id=background_id,
        #         molecule_id=molecule_id,
        #         p_aug_a=0.3,        # Background augmentation probability
        #         p_aug_b=color_aug_prob,  # Partial molecule augmentation probability
        #         p_aug_c=0.4,        # Both augmentation probability
        #     )
        # ]

        trans = []  # Start with empty transforms

        trans.append(T.RandomResize(min_size, max_size))
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        
        trans.extend(
            [
                #T.RandomCrop(crop_size),      #CHANGED TO AVOID CROPPED MOLECUES
                T.Resize((crop_size, crop_size)),  # Resize to fixed size (instead of random crop)          
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(
        self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    ):
        self.transforms = T.Compose(
            [
                T.RandomResize(base_size, base_size),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train):
    base_size = 256 #WE CHANGED IT FROM 520 TO 256 TO AVOID OOM ERROR
    crop_size = 224 #WE CHANGED IT FROM 480 TO 224 TO AVOID OOM ERROR
    #base_size = 520 
    #crop_size = 480 

    return (
        SegmentationPresetTrain(base_size, crop_size)
        if train
        else SegmentationPresetEval(base_size)
    )


def create_model(aux, num_classes):
    model = deeplabv3_resnet50(aux=aux, num_classes=num_classes)
    
    # WE CHANGED IT TO Load YoDe pre-trained weights (domain-specific, trained on binary classification)
    yode_weights_path = "../weights/model.pth"
    
    if os.path.exists(yode_weights_path):
        print(f"Loading YoDe pre-trained weights from {yode_weights_path}") #debug print
        checkpoint = torch.load(yode_weights_path, map_location="cpu")
        
        # Handle both direct model state and checkpoint dict format
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            weights_dict = checkpoint["model"]
        else:
            weights_dict = checkpoint
        
        # WE removed the if satetment that checks if num_classes != 21, because we want to load YoDe's weights
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)
    else:
        print(f"Warning: YoDe weights not found at {yode_weights_path}")
        print("Model initialized with random weights. Consider training from scratch.")
    
    return model


def main(args):
    init_distributed_mode(args)
    print(args)

    # # Initialize fallback logger
    # fallback_log_file = os.path.join(args.output_dir, "fallbacks.log") if args.output_dir else None
    # fallback_logger = get_logger(log_file=fallback_log_file, max_prints_per_epoch=10)

    device = torch.device(args.device)
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1

    # 用来保存coco_info的文件
    results_file = "results{}.txt".format(
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    VOC_root = args.data_path
    # check voc root
    if os.path.exists(os.path.join(VOC_root, "VOCdevkit")) is False:
        raise FileNotFoundError("VOCdevkit dose not in path:'{}'.".format(VOC_root))

    # load train data set
    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> train.txt
    train_dataset = VOCSegmentation(
        args.data_path,
        year="2012",
        transforms=get_transform(train=True),
        txt_name="train.txt",
    )
    # load validation data set
    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> val.txt
    val_dataset = VOCSegmentation(
        args.data_path,
        year="2012",
        transforms=get_transform(train=False),
        txt_name="val.txt",
    )

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(val_dataset)

    pin_memory = device.type == "cuda"
    persistent_workers = args.workers > 0

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=train_dataset.collate_fn,
        drop_last=True,
        #worker_init_fn=worker_init_fn if args.workers > 0 else None,
    )

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=train_dataset.collate_fn,
        #worker_init_fn=worker_init_fn if args.workers > 0 else None,
    )

    # Create fixed subset of 500 samples for frequent validation
    subset_size = min(500, len(val_dataset))
    subset_indices = torch.randperm(len(val_dataset))[:subset_size].tolist()
    subset_sampler = torch.utils.data.SubsetRandomSampler(subset_indices)
    val_subset_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=subset_sampler,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=train_dataset.collate_fn,
        drop_last=False,
    )

    print("Creating model")
    # create model num_classes equal background + 20 classes
    model = create_model(aux=args.aux, num_classes=num_classes)
    model.to(device)

    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params_to_optimize = [
        {
            "params": [
                p for p in model_without_ddp.backbone.parameters() if p.requires_grad
            ]
        },
        {
            "params": [
                p for p in model_without_ddp.classifier.parameters() if p.requires_grad
            ]
        },
    ]
    if args.aux:
        params = [
            p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad
        ]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})
    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(
        optimizer, len(train_data_loader), args.epochs, warmup=True
    )

    # 如果传入resume参数，即上次训练的权重地址，则接着上次的参数训练
    if args.resume:
        # If map_location is missing, torch.load will first load the module to CPU
        # and then copy each parameter to where it was saved,
        # which would result in all processes on the same machine using the same set of devices.
        checkpoint = torch.load(
            args.resume, map_location="cpu"
        )  # 读取之前保存的权重文件(包括优化器以及学习率策略)
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        confmat = evaluate(
            model, val_data_loader, device=device, num_classes=num_classes
        )
        val_info = str(confmat)
        print(val_info)
        return

    print("Start training")
    start_time = time.time()
    
    is_master = (not args.distributed) or (getattr(args, "rank", 0) == 0)

    # Define validation callback for periodic validation
    def validation_callback(epoch, batch_idx):
        subset_val_loss = compute_validation_loss(model, val_subset_loader, device)
        print(f"[Epoch {epoch}, Batch {batch_idx}] Validation loss on subset (500 samples): {subset_val_loss:.4f}")

    # Avoid wasting compute on non-master ranks for logging-only losses.
    if args.distributed and not is_master:
        validation_callback = None
    
    for epoch in range(args.start_epoch, args.epochs):
        # Reset fallback counters for new epoch
        #fallback_logger.reset_epoch()
        
        if args.distributed:
            train_sampler.set_epoch(epoch)
        mean_loss, lr = train_one_epoch(
            model,
            optimizer,
            train_data_loader,
            device,
            epoch,
            lr_scheduler=lr_scheduler,
            print_freq=args.print_freq,
            scaler=scaler,
            validation_callback=validation_callback,
            validation_freq=200,
        )

        confmat = evaluate(
            model, val_data_loader, device=device, num_classes=num_classes
        )
        val_info = str(confmat)
        print(val_info)
        
        # Compute validation loss on full validation set (logging only).
        # This is intentionally done on master only to avoid redundant compute.
        full_val_loss = None
        if is_master:
            full_val_loss = compute_validation_loss(model, val_data_loader, device)
            print(f"[Epoch {epoch}] Validation loss on full set: {full_val_loss:.4f}")

        # 只在主进程上进行写操作
        #if args.rank in [-1, 0]:
        if (not args.distributed) or (getattr(args, "rank", 0) in [-1, 0]):
            # write into txt
            with open(results_file, "a") as f:
                # 记录每个epoch对应的train_loss、lr以及验证集各指标
                train_info = (
                    f"[epoch: {epoch}]\n"
                    f"train_loss: {mean_loss:.4f}\n"
                    f"lr: {lr:.6f}\n"
                    f"val_loss_full: {full_val_loss:.4f}\n"
                )
                f.write(train_info + val_info + "\n\n")

        if args.output_dir:
            # 只在主节点上执行保存权重操作
            save_file = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "args": args,
                "epoch": epoch,
            }
            if args.amp:
                save_file["scaler"] = scaler.state_dict()
            save_on_master(
                save_file, os.path.join(args.output_dir, "model_{}.pth".format(epoch))
            )

        # Print epoch fallback summary
        #fallback_logger.epoch_summary(epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    import argparse

    ###
    import torch.multiprocessing as mp

    # On Linux, DataLoader workers default to "fork". Forked workers can inherit
    # CUDA state and crash. Use "spawn" so workers start clean (required for
    # num_workers >= 1 on GPU servers).
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass  # already set


    parser = argparse.ArgumentParser(description=__doc__)

    # 训练文件的根目录(VOCdevkit)
    parser.add_argument("--data-path", default="/data/", help="dataset")
    # 训练设备类型
    parser.add_argument("--device", default="cuda", help="device")
    # 检测目标类别数(不包含背景)
    parser.add_argument("--num-classes", default=20, type=int, help="num_classes")
    # 每块GPU上的batch_size
    parser.add_argument(
        "-b",
        "--batch-size",
        default=4,
        type=int,
        help="images per gpu, the total batch size is $NGPU x batch_size",
    )
    #parser.add_argument("--aux", default=True, type=bool, help="auxilier loss")

    parser.add_argument("--aux", action="store_true", help="use auxiliary loss")
    parser.add_argument("--no-aux", dest="aux", action="store_false")
    parser.set_defaults(aux=True)

    # 指定接着从哪个epoch数开始训练
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    # 训练的总epoch数
    parser.add_argument(
        "--epochs",
        default=20,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    # 是否使用同步BN(在多个GPU之间同步)，默认不开启，开启后训练速度会变慢
    parser.add_argument(
        "--sync_bn", type=bool, default=False, help="whether using SyncBatchNorm"
    )
    # 数据加载以及预处理的线程数
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    # 训练学习率，这里默认设置成0.0001，如果效果不好可以尝试加大学习率
    parser.add_argument(
        "--lr", default=0.0001, type=float, help="initial learning rate"
    )
    # SGD的momentum参数
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    # SGD的weight_decay参数
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    # 训练过程打印信息的频率
    parser.add_argument("--print-freq", default=20, type=int, help="print frequency")
    # 文件保存地址
    parser.add_argument(
        "--output-dir", default="./multi_train", help="path where to save"
    )
    # 基于上次的训练结果接着训练
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    # 不训练，仅测试
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    # 分布式进程数
    parser.add_argument(
        "--world-size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist-url", default="env://", help="url used to set up distributed training"
    )
    # Mixed precision training parameters
    parser.add_argument(
        "--amp",
        default=False,
        type=bool,
        help="Use torch.cuda.amp for mixed precision training",
    )

    args = parser.parse_args()

    # 如果指定了保存文件地址，检查文件夹是否存在，若不存在，则创建
    if args.output_dir:
        mkdir(args.output_dir)

    main(args)
