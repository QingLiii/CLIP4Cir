from comet_ml import Experiment
import json
import multiprocessing
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from statistics import mean, geometric_mean, harmonic_mean
from typing import List
import clip
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image
import os

from data_utils import base_path, squarepad_transform, targetpad_transform, CIRRDataset, FashionIQDataset
from utils import collate_fn, update_train_running_results, set_train_bar_description, extract_index_features, \
    save_model, generate_randomized_fiq_caption, element_wise_sum, device
from validate import compute_cirr_val_metrics, compute_fiq_val_metrics

"""
python src/clip_fine_tune.py --dataset pathology --data-root e:\path\to\your\pathology\dataset --clip-model-name RN50x4 --batch-size 32 --save-training --save-best
"""
# ... 现有代码保持不变 ...

# 添加病理图像数据集类
class PathologyDataset(Dataset):
    """
    用于加载病理图像和对应文本描述的数据集
    """
    def __init__(self, data_root, split='train', transform=None):
        """
        初始化病理图像数据集
        :param data_root: 数据集根目录
        :param split: 数据集划分，'train'或'val'
        :param transform: 图像预处理转换
        """
        self.data_root = data_root
        self.split = split
        self.transform = transform
        
        # 加载数据集索引文件
        index_file = os.path.join(data_root, f"{split}_index.json")
        with open(index_file, 'r', encoding='utf-8') as f:
            self.data_index = json.load(f)
        
        self.image_pairs = []
        self.captions = []
        
        # 处理数据索引
        for item in self.data_index:
            self.image_pairs.append((item['reference_image'], item['target_image']))
            self.captions.append(item['caption'])
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        ref_img_path, target_img_path = self.image_pairs[idx]
        caption = self.captions[idx]
        
        # 加载图像
        ref_img = Image.open(os.path.join(self.data_root, 'images', ref_img_path)).convert('RGB')
        target_img = Image.open(os.path.join(self.data_root, 'images', target_img_path)).convert('RGB')
        
        # 应用转换
        if self.transform:
            ref_img = self.transform(ref_img)
            target_img = self.transform(target_img)
        
        return ref_img, target_img, caption


def clip_finetune_pathology(data_root: str, num_epochs: int, clip_model_name: str, learning_rate: float, batch_size: int,
                           validation_frequency: int, transform: str, save_training: bool, encoder: str, save_best: bool,
                           **kwargs):
    """
    在病理图像数据集上微调CLIP模型，使用图像-文本元素级求和作为组合函数
    :param data_root: 病理图像数据集的根目录
    :param num_epochs: 训练轮数
    :param clip_model_name: 要使用的CLIP模型："RN50", "RN101", "RN50x4"等
    :param learning_rate: 微调学习率
    :param batch_size: 批量大小
    :param validation_frequency: 验证频率（以epoch为单位）
    :param transform: 要使用的预处理转换。应为['clip', 'squarepad', 'targetpad']之一
                     当使用targetpad时，还需要提供`target_ratio`参数
    :param save_training: 为True时保存微调后的CLIP模型权重
    :param encoder: 要微调的CLIP编码器，应为['both', 'text', 'image']之一
    :param save_best: 为True时仅保存在平均召回率指标上表现最佳的CLIP模型权重
    :param kwargs: 如果使用`targetpad`转换，应提供`target_ratio`作为kwargs
    """
    training_start = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    training_path: Path = Path(
        base_path / f"models/clip_finetuned_on_pathology_{clip_model_name}_{training_start}")
    training_path.mkdir(exist_ok=False, parents=True)

    # 保存所有超参数到文件
    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)

    clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)

    if encoder == 'text':
        print('只微调CLIP文本编码器')
        for param in clip_model.visual.parameters():
            param.requires_grad = False
    elif encoder == 'image':
        print('只微调CLIP图像编码器')
        for param in clip_model.parameters():
            param.requires_grad = False
        for param in clip_model.visual.parameters():
            param.requires_grad = True
    elif encoder == 'both':
        print('同时微调CLIP两个编码器')
    else:
        raise ValueError("encoder参数应为['text', 'image', both']之一")

    clip_model.eval().float()
    input_dim = clip_model.visual.input_resolution

    if transform == "clip":
        preprocess = clip_preprocess
        print('使用CLIP默认预处理管道')
    elif transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print('使用方形填充预处理管道')
    elif transform == "targetpad":
        target_ratio = kwargs['target_ratio']
        preprocess = targetpad_transform(target_ratio, input_dim)
        print(f'使用目标比例为{target_ratio}的目标填充预处理管道')
    else:
        raise ValueError("预处理转换应为['clip', 'squarepad', 'targetpad']之一")

    # 定义验证数据集
    val_dataset = PathologyDataset(data_root, 'val', preprocess)
    
    # 当只微调文本编码器时，可以预计算索引特征，因为它们在各个epoch之间不会改变
    if encoder == 'text':
        val_index_features, val_index_names = extract_index_features(val_dataset, clip_model)

    # 定义训练数据集和组合函数
    train_dataset = PathologyDataset(data_root, 'train', preprocess)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                             num_workers=multiprocessing.cpu_count(), pin_memory=False, collate_fn=collate_fn,
                             drop_last=True, shuffle=True)
    combining_function = element_wise_sum

    # 定义优化器、损失函数和梯度缩放器
    optimizer = optim.AdamW(
        [{'params': filter(lambda p: p.requires_grad, clip_model.parameters()), 'lr': learning_rate,
          'betas': (0.9, 0.999), 'eps': 1e-7}])
    crossentropy_criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler()

    # 当save_best == True时初始化最佳结果为零
    if save_best:
        best_recall = 0

    # 定义用于CSV日志记录的数据框
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    # 开始训练循环
    print('训练循环已开始')
    for epoch in range(num_epochs):
        with experiment.train():
            train_running_results = {'images_in_epoch': 0, 'accumulated_train_loss': 0}
            train_bar = tqdm(train_loader, ncols=150)
            for idx, (reference_images, target_images, captions) in enumerate(train_bar):
                images_in_batch = reference_images.size(0)
                step = len(train_bar) * epoch + idx

                optimizer.zero_grad()

                reference_images = reference_images.to(device, non_blocking=True)
                target_images = target_images.to(device, non_blocking=True)

                # 对于病理图像，我们直接使用原始描述，不进行随机化
                text_inputs = clip.tokenize(captions, context_length=77, truncate=True).to(device, non_blocking=True)

                # 提取特征，计算logits和损失
                with torch.amp.autocast('cuda'):
                    reference_features = clip_model.encode_image(reference_images)
                    caption_features = clip_model.encode_text(text_inputs)
                    predicted_features = combining_function(reference_features, caption_features)
                    target_features = F.normalize(clip_model.encode_image(target_images), dim=-1)

                    logits = 100 * predicted_features @ target_features.T

                    ground_truth = torch.arange(images_in_batch, dtype=torch.long, device=device)
                    loss = crossentropy_criterion(logits, ground_truth)

                # 反向传播并更新权重
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                experiment.log_metric('step_loss', loss.detach().cpu().item(), step=step)
                update_train_running_results(train_running_results, loss, images_in_batch)
                set_train_bar_description(train_bar, epoch, num_epochs, train_running_results)

            train_epoch_loss = float(
                train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch'])
            experiment.log_metric('epoch_loss', train_epoch_loss, epoch=epoch)

            # 训练CSV日志记录
            training_log_frame = pd.concat(
                [training_log_frame,
                 pd.DataFrame(data={'epoch': epoch, 'train_epoch_loss': train_epoch_loss}, index=[0])])
            training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        if epoch % validation_frequency == 0:
            with experiment.validate():
                # 为病理图像数据集实现简单的验证指标
                # 这里我们计算top-1, top-5, top-10的准确率
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                                       num_workers=multiprocessing.cpu_count())
                
                correct_1 = 0
                correct_5 = 0
                correct_10 = 0
                total = 0
                
                for reference_images, target_images, captions in tqdm(val_loader, desc="Validating"):
                    batch_size = reference_images.size(0)
                    total += batch_size
                    
                    reference_images = reference_images.to(device)
                    target_images = target_images.to(device)
                    text_inputs = clip.tokenize(captions, context_length=77).to(device)
                    
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        reference_features = clip_model.encode_image(reference_images)
                        text_features = clip_model.encode_text(text_inputs)
                        
                        # 使用组合函数
                        predicted_features = combining_function(reference_features, text_features)
                        
                        # 计算所有目标图像的特征
                        all_target_features = F.normalize(clip_model.encode_image(target_images), dim=-1)
                        
                        # 计算相似度
                        similarity = predicted_features @ all_target_features.T
                        
                        # 计算top-k准确率
                        _, indices = similarity.topk(10, dim=1)
                        targets = torch.arange(batch_size, device=device)
                        
                        correct_1 += (indices[:, 0] == targets).sum().item()
                        correct_5 += (indices[:, :5] == targets.view(-1, 1)).any(dim=1).sum().item()
                        correct_10 += (indices[:, :10] == targets.view(-1, 1)).any(dim=1).sum().item()
                
                accuracy_1 = correct_1 / total
                accuracy_5 = correct_5 / total
                accuracy_10 = correct_10 / total
                
                results_dict = {
                    'top1_accuracy': accuracy_1,
                    'top5_accuracy': accuracy_5,
                    'top10_accuracy': accuracy_10,
                    'average_accuracy': (accuracy_1 + accuracy_5 + accuracy_10) / 3
                }
                
                print(json.dumps(results_dict, indent=4))
                experiment.log_metrics(results_dict, epoch=epoch)
                
                # 验证CSV日志记录
                log_dict = {'epoch': epoch}
                log_dict.update(results_dict)
                validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
                validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)
                
                if save_training:
                    if save_best and results_dict['average_accuracy'] > best_recall:
                        best_recall = results_dict['average_accuracy']
                        save_model('tuned_clip_best', epoch, clip_model, training_path)
                    elif not save_best:
                        save_model(f'tuned_clip_{epoch}', epoch, clip_model, training_path)

# ... 现有代码保持不变 ...

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="should be either 'CIRR', 'fashionIQ' or 'pathology'")
    parser.add_argument("--data-root", type=str, help="root directory of the pathology dataset")
    parser.add_argument("--api-key", type=str, help="api for Comet logging")
    parser.add_argument("--workspace", type=str, help="workspace of Comet logging")
    parser.add_argument("--experiment-name", type=str, help="name of the experiment on Comet")
    parser.add_argument("--num-epochs", default=300, type=int, help="number training epochs")
    parser.add_argument("--clip-model-name", default="RN50x4", type=str, help="CLIP model to use, e.g 'RN50', 'RN50x4'")
    parser.add_argument("--encoder", default='both', type=str,
                        help="Which CLIP encoder to fine-tune, should be in ['both', 'text', 'image']")
    parser.add_argument("--learning-rate", default=2e-6, type=float, help="Learning rate")
    parser.add_argument("--batch-size", default=512, type=int, help="Batch size")
    parser.add_argument("--validation-frequency", default=1, type=int, help="Validation frequency expressed in epochs")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")
    parser.add_argument("--save-training", dest="save_training", action='store_true',
                        help="Whether save the training model")
    parser.add_argument("--save-best", dest="save_best", action='store_true',
                        help="Save only the best model during training")

    args = parser.parse_args()
    if args.dataset.lower() not in ['fashioniq', 'cirr', 'pathology']:
        raise ValueError("Dataset should be either 'CIRR', 'FashionIQ' or 'pathology'")
    
    if args.dataset.lower() == 'pathology' and not args.data_root:
        raise ValueError("For pathology dataset, --data-root must be provided")

    training_hyper_params = {
        "num_epochs": args.num_epochs,
        "clip_model_name": args.clip_model_name,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "validation_frequency": args.validation_frequency,
        "transform": args.transform,
        "target_ratio": args.target_ratio,
        "save_training": args.save_training,
        "encoder": args.encoder,
        "save_best": args.save_best
    }

    if args.api_key and args.workspace:
        print("Comet logging ENABLED")
        experiment = Experiment(
            api_key=args.api_key,
            project_name=f"{args.dataset} clip fine-tuning",
            workspace=args.workspace,
            disabled=False
        )
        if args.experiment_name:
            experiment.set_name(args.experiment_name)
    else:
        print("Comet loging DISABLED, in order to enable it you need to provide an api key and a workspace")
        experiment = Experiment(
            api_key="",
            project_name="",
            workspace="",
            disabled=True
        )

    experiment.log_code(folder=str(base_path / 'src'))
    experiment.log_parameters(training_hyper_params)

    if args.dataset.lower() == 'cirr':
        clip_finetune_cirr(**training_hyper_params)
    elif args.dataset.lower() == 'fashioniq':
        training_hyper_params.update(
            {'train_dress_types': ['dress', 'toptee', 'shirt'], 'val_dress_types': ['dress', 'toptee', 'shirt']})
        clip_finetune_fiq(**training_hyper_params)
    elif args.dataset.lower() == 'pathology':
        training_hyper_params.update({'data_root': args.data_root})
        clip_finetune_pathology(**training_hyper_params)