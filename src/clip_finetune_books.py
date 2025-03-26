from comet_ml import Experiment
import json
import multiprocessing
import os
import random
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import List, Dict, Tuple

import clip
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image

from data_utils import base_path, squarepad_transform, targetpad_transform
from utils import collate_fn, update_train_running_results, set_train_bar_description, extract_index_features, \
    save_model, element_wise_sum, device

"""
python src/clip_finetune_books.py --dataset books --data-root E:/Github/CLIP4Cir/books_set/books_set --clip-model-name RN50x4 --batch-size 32 --save-training --save-best
"""


class BooksDataset(Dataset):
    """
    用于加载病理图像书籍数据集和对应文本描述的数据集
    """
    def __init__(self, data_root: str, split: str = 'train', transform=None):
        """
        初始化病理图像书籍数据集
        :param data_root: 数据集根目录
        :param split: 数据集划分，'train'或'val'
        :param transform: 图像预处理转换
        """
        self.data_root = data_root
        self.split = split
        self.transform = transform
        
        # 加载captions.json文件
        captions_file = os.path.join(data_root, "captions.json")
        with open(captions_file, 'r', encoding='utf-8') as f:
            self.captions_data = json.load(f)
        
        # 获取所有图像文件
        self.images_dir = os.path.join(data_root, "images")
        self.image_files = [f for f in os.listdir(self.images_dir) if f.endswith('.png')]
        
        # 创建uuid到caption的映射
        self.uuid_to_caption = {}
        for _, item in self.captions_data.items():
            self.uuid_to_caption[item['uuid']] = item['caption']
        
        # 创建训练和验证集
        random.seed(42)  # 设置随机种子以确保可重复性
        random.shuffle(self.image_files)
        
        # 划分数据集 - 80%训练，20%验证
        split_idx = int(len(self.image_files) * 0.8)
        if split == 'train':
            self.image_files = self.image_files[:split_idx]
        else:  # 'val'
            self.image_files = self.image_files[split_idx:]
        
        # 创建图像对和对应的描述
        self.image_pairs = []
        self.captions = []
        
        # 为每个图像找到一个随机的配对图像和对应的描述
        for i, img_file in enumerate(self.image_files):
            uuid = img_file.split('.')[0]  # 从文件名获取uuid
            if uuid in self.uuid_to_caption:
                # 为每个图像随机选择另一个图像作为目标
                target_idx = (i + 1) % len(self.image_files)  # 简单地选择下一个图像
                target_img = self.image_files[target_idx]
                
                self.image_pairs.append((img_file, target_img))
                self.captions.append(self.uuid_to_caption[uuid])
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        ref_img_file, target_img_file = self.image_pairs[idx]
        caption = self.captions[idx]
        
        # 加载图像
        ref_img_path = os.path.join(self.images_dir, ref_img_file)
        target_img_path = os.path.join(self.images_dir, target_img_file)
        
        ref_img = Image.open(ref_img_path).convert('RGB')
        target_img = Image.open(target_img_path).convert('RGB')
        
        # 应用转换
        if self.transform:
            ref_img = self.transform(ref_img)
            target_img = self.transform(target_img)
        
        return ref_img, target_img, caption


def compute_books_val_metrics(val_loader: DataLoader, clip_model, combining_function) -> Dict[str, float]:
    """
    计算Books数据集上的验证指标
    :param val_loader: 验证数据加载器
    :param clip_model: CLIP模型
    :param combining_function: 组合函数
    :return: 包含验证指标的字典
    """
    correct_1 = 0
    correct_5 = 0
    correct_10 = 0
    total = 0
    
    for reference_images, target_images, captions in tqdm(val_loader, desc="Validating"):
        batch_size = reference_images.size(0)
        total += batch_size
        
        reference_images = reference_images.to(device)
        target_images = target_images.to(device)
        text_inputs = clip.tokenize(captions, context_length=77, truncate=True).to(device)
        
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
    
    return results_dict


def clip_finetune_books(data_root: str, num_epochs: int, clip_model_name: str, learning_rate: float, batch_size: int,
                       validation_frequency: int, transform: str, save_training: bool, encoder: str, save_best: bool,
                       **kwargs):
    """
    在病理图像书籍数据集上微调CLIP模型，使用图像-文本元素级求和作为组合函数
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
        base_path / f"models/clip_finetuned_on_books_{clip_model_name}_{training_start}")
    training_path.mkdir(exist_ok=False, parents=True)

    # 保存所有超参数到文件
    training_hyper_params = {
        'data_root': str(data_root),
        'num_epochs': num_epochs,
        'clip_model_name': clip_model_name,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'validation_frequency': validation_frequency,
        'transform': transform,
        'save_training': save_training,
        'encoder': encoder,
        'save_best': save_best,
    }
    if transform == "targetpad":
        training_hyper_params['target_ratio'] = kwargs['target_ratio']
        
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

    # 定义训练和验证数据集
    train_dataset = BooksDataset(data_root, 'train', preprocess)
    val_dataset = BooksDataset(data_root, 'val', preprocess)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                             num_workers=multiprocessing.cpu_count(), pin_memory=False, collate_fn=collate_fn,
                             drop_last=True, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, 
                           num_workers=multiprocessing.cpu_count(), pin_memory=False, collate_fn=collate_fn,
                           shuffle=False)
    
    combining_function = element_wise_sum

    # 定义优化器、损失函数和梯度缩放器
    optimizer = optim.AdamW(
        [{'params': filter(lambda p: p.requires_grad, clip_model.parameters()), 'lr': learning_rate,
          'betas': (0.9, 0.999), 'eps': 1e-7}])
    crossentropy_criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler()

    # 当save_best == True时初始化最佳结果为零
    if save_best:
        best_accuracy = 0

    # 定义用于CSV日志记录的数据框
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    # 设置comet.ml实验
    experiment = Experiment(
        api_key="ZcFfJJXNb04CeIGTfbwOFRr1i",
        project_name="clip-finetune-books",
        workspace="test_finetune_0326",
    )
    experiment.log_parameters(training_hyper_params)

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
                # 计算验证指标
                results_dict = compute_books_val_metrics(val_loader, clip_model, combining_function)
                
                print(json.dumps(results_dict, indent=4))
                experiment.log_metrics(results_dict, epoch=epoch)
                
                # 验证CSV日志记录
                log_dict = {'epoch': epoch}
                log_dict.update(results_dict)
                validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
                validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)
                
                if save_training:
                    if save_best:
                        if results_dict['average_accuracy'] > best_accuracy:
                            best_accuracy = results_dict['average_accuracy']
                            save_model(f"best_clip_books_{clip_model_name}", epoch, clip_model, training_path)
                    else:
                        save_model(f"clip_books_{clip_model_name}_epoch_{epoch}", epoch, clip_model, training_path)

    # 如果不是只保存最佳模型，则保存最终模型
    if save_training and not save_best:
        save_model(f"final_clip_books_{clip_model_name}", num_epochs - 1, clip_model, training_path)

    print(f"训练完成，模型和日志保存在 {training_path}")
    return clip_model


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="books", help="数据集名称")
    parser.add_argument("--data-root", type=str, required=True, help="数据集根目录")
    parser.add_argument("--num-epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--clip-model-name", type=str, default="RN50", help="CLIP模型名称")
    parser.add_argument("--learning-rate", type=float, default=1e-6, help="学习率")
    parser.add_argument("--batch-size", type=int, default=32, help="批量大小")
    parser.add_argument("--validation-frequency", type=int, default=1, help="验证频率（以epoch为单位）")
    parser.add_argument("--transform", type=str, default="clip", help="预处理转换类型")
    parser.add_argument("--target-ratio", type=float, default=1.25, help="目标填充比例")
    parser.add_argument("--save-training", action="store_true", help="是否保存训练模型")
    parser.add_argument("--encoder", type=str, default="both", help="要微调的编码器")
    parser.add_argument("--save-best", action="store_true", help="是否只保存最佳模型")
    args = parser.parse_args()

    if args.transform == "targetpad":
        clip_finetune_books(args.data_root, args.num_epochs, args.clip_model_name, args.learning_rate, args.batch_size,
                          args.validation_frequency, args.transform, args.save_training, args.encoder, args.save_best,
                          target_ratio=args.target_ratio)
    else:
        clip_finetune_books(args.data_root, args.num_epochs, args.clip_model_name, args.learning_rate, args.batch_size,
                          args.validation_frequency, args.transform, args.save_training, args.encoder, args.save_best)