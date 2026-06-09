import torch
import os
import torch
import numpy as np
import pandas as pd
import argparse
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score

class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            # 移除 BatchNorm1d，在小型推荐任务中它可能导致不稳定
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

def get_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser()
    # 数据集路径
    parser.add_argument('--train_path', default='./datahub/taobaoAd/process/train.csv')
    parser.add_argument('--test_path', default='./datahub/taobaoAd/process/test.csv')
    # 训练批大小
    parser.add_argument('--bsz', type=int, default=512)
    # 是否打乱训练集数据 (1为打乱, 0为不打乱)
    parser.add_argument('--shuffle', type=int, default=1)
    # 训练轮数
    parser.add_argument('--epoch', type=int, default=1) 
    # 学习率
    parser.add_argument('--lr', type=float, default=0.0005)
    # 权重衰减 (L2正则化)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    # 运行设备 (cpu 或 cuda)
    parser.add_argument('--device', default='cpu')
    # 随机种子，确保实验可复现
    parser.add_argument('--seed', type=int, default=2024)
    args = parser.parse_args()
    return args
    
    class DINDataset(Dataset):
    """
    自定义数据集类，用于将 Pandas DataFrame 转换为 PyTorch 张量
    """
    def __init__(self, df, description, encoders, device):
        """
        初始化数据集
        参数:
            df: 包含原始数据的 Pandas DataFrame。
            description: 特征描述列表，格式为 [(name, vocab_size, type), ...]。
                         它告诉 Dataset 哪些列是离散特征(spr)、连续特征(ctn)或序列特征(seq)。
            encoders: 编码字典映射。这是最关键的部分：
                      它是一个大字典，Key 是特征名（如 'user_id'），Value 是一个内部字典。
                      内部字典负责将原始值（如字符串 'user_123' 或 ID 1001）映射为从 0 开始的连续整数索引。
                      例如：self.encoders['user_id'] = {1001: 0, 1002: 1, ...}
                      这样模型才能通过这个索引在 Embedding 层中查找对应的向量。
            device: 数据存放的设备 (cpu 或 cuda)。
        """
        # 存储原始数据和配置
        self.df = df
        self.description = description
        self.encoders = encoders
        self.device = device
        
        # 1. 提取所有特征列的名称（排除标签列）
        self.features = [name for name, _, t in description if t != 'label']
        # 2. 找到所有标签列的名称
        self.label_names = [name for name, _, t in description if t == 'label']
        
        # 3. 将特征按类型分类，方便后续分别处理
        # seq_names: 历史行为序列特征（如 'hist_seq'）
        self.seq_names = [name for name, _, t in description if t == 'seq']
        # spr_names: 离散/稀疏特征（如 'user_id', 'item_id', 'brand' 等）
        self.spr_names = [name for name, _, t in description if t == 'spr']
        # ctn_names: 连续特征（如 'price'）
        self.ctn_names = [name for name, _, t in description if t == 'ctn']
        
        # 4. 调用内部方法，将 DataFrame 中的所有数据预先转换为内存中的 Tensor，提高读取速度
        self._build_tensors()

    def _build_tensors(self):
        """
        核心预处理逻辑：根据特征类型，将 DataFrame 中的原始数据转换为 PyTorch 张量并存入 self.name2array 字典。
        """
        # name2array 用于存放处理好的张量，Key 是特征名，Value 是对应的 Tensor
        self.name2array = {}
        
        # --- A. 处理离散(稀疏)特征 (Sparse Features) ---
        for name in self.spr_names:
            # 获取该特征对应的编码字典（翻译字典）
            enc = self.encoders[name]
            # 遍历 DataFrame 中的原始值，通过编码字典转换为整数索引
            # .get(v, 0): 如果遇到没见过的原始值，统一映射为索引 0 (Padding/Unknown)
            vals = [enc.get(v, 0) for v in self.df[name].tolist()]
            # 将索引列表转换为形状为 [N, 1] 的长整型 Tensor (N 为样本数)
            arr = torch.from_numpy(np.array(vals).reshape([-1, 1])).to(self.device).to(torch.long)
            self.name2array[name] = arr
            
        # --- B. 处理连续特征 (Continuous Features) ---
        for name in self.ctn_names:
            # 连续特征不需要编码，直接取原始数值（如价格）
            # .fillna(0.0): 缺失值补 0.0
            vals = self.df[name].astype(float).fillna(0.0).tolist()
            # 将数值列表转换为形状为 [N, 1] 的浮点型 Tensor
            arr = torch.from_numpy(np.array(vals).reshape([-1, 1])).to(self.device).to(torch.float32)
            self.name2array[name] = arr
            
        # --- C. 处理序列特征 (Sequence Features，如用户点击历史) ---
        for name in self.seq_names:
            # 在 DataFrame 中寻找该序列特征对应的具体列名（如 'hist_seq_1', 'hist_seq_2' ...）
            seq_cols = [c for c in self.df.columns if c.startswith(name + '_')]
            # 序列特征（如点击历史）通常与对应的离散特征（如 item_id）共享同一个编码字典
            enc = self.encoders[name]
            # 取出这些列的原始值，结果是一个二维 NumPy 数组 [N, T]
            seq_vals = self.df[seq_cols].values
            idx_vals = []
            # 对每一行的每一个历史点击进行编码转换
            for row in seq_vals:
                idx_row = [enc.get(v, 0) for v in row]
                idx_vals.append(idx_row)
            # 将二维索引列表转换为形状为 [N, T] 的长整型 Tensor (T 为序列长度)
            arr = torch.from_numpy(np.array(idx_vals)).to(self.device).to(torch.long)
            self.name2array[name] = arr
            
        # --- D. 处理标签 (Labels) ---
        for label_name in self.label_names:
            labels = self.df[label_name].astype(int).tolist()
            # 标签 Tensor 形状为 [N, 1]
            self.name2array[label_name] = torch.from_numpy(np.array(labels).reshape([-1, 1])).to(self.device).to(torch.float32)
        
        self.length = len(self.df)

    def __getitem__(self, index):
        """
        DataLoader 在训练时会调用此方法，根据索引获取一个样本。
        参数:
            index: 样本在数据集中的索引位置。
        返回:
            一个元组 (features_dict, labels_dict)
            features_dict: 字典，包含该样本所有特征的 Tensor。
            labels_dict: 字典，包含 clk_label 和 pay_label。
        """
        feat_dict = {name: self.name2array[name][index] for name in self.features}
        label_dict = {name: self.name2array[name][index] for name in self.label_names}
        return feat_dict, label_dict

    def __len__(self):
        """
        返回数据集中的样本总数，用于 DataLoader 确定迭代次数。
        """
        return self.length

class DINDataLoaders(object):
    """
    数据加载管理类，负责读取 train.csv 和 test.csv、编码映射以及创建 DataLoader
    """
    def __init__(self, train_path, test_path, device, bsz=2048, shuffle=True):
        # 1. 读取训练集和测试集
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # 清除列名中的空格并确保标签类型正确
        for df in [train_df, test_df]:
            df.columns = [c.strip() for c in df.columns]
            if 'clk_label' in df.columns:
                df['clk_label'] = df['clk_label'].astype(int)
            if 'pay_label' in df.columns:
                df['pay_label'] = df['pay_label'].astype(int)
        
        # 2. 定义特征列
        cat_cols = ['user_id','item_id','cms_segid','cms_group_id','final_gender_code','age_level','pvalue_level','shopping_level','occupation','new_user_class_level','cate_id','campaign_id','customer','brand']
        cat_cols = [c for c in cat_cols if c in train_df.columns]
        ctn_cols = ['price']
        ctn_cols = [c for c in ctn_cols if c in train_df.columns]
        
        # 对连续特征进行归一化 (Z-score)
        for c in ctn_cols:
            mean = train_df[c].mean()
            std = train_df[c].std() + 1e-9
            train_df[c] = (train_df[c] - mean) / std
            test_df[c] = (test_df[c] - mean) / std
        
        # 3. 处理历史行为序列
        seq_base = 'hist_seq'
        hist_cols = [c for c in ['hist_item_1','hist_item_2','hist_item_3','hist_item_4','hist_item_5'] if c in train_df.columns]
        seq_cols = []
        if len(hist_cols) > 0:
            seq_cols = [seq_base + '_' + str(i+1) for i in range(len(hist_cols))]
            for df in [train_df, test_df]:
                for i, hc in enumerate(hist_cols):
                    df[seq_cols[i]] = df[hc]
        
        # 4. 构建统一的 item_id 编码器（基于训练集和测试集的所有 item_id）
        all_item_ids = pd.concat([train_df['item_id'], test_df['item_id']]).fillna(-1).astype(float).astype(int)
        item_unique = sorted(all_item_ids.unique().tolist())
        item_map = {v: i+1 for i, v in enumerate(item_unique)} # 0 留给 padding
        
        # 5. 构建其他分类特征的编码器（统一映射）
        encoders = {}
        description = []
        for c in cat_cols:
            if c == 'item_id':
                encoders[c] = item_map
                size = len(item_map) + 1
            else:
                # 基于全量数据构建编码器，确保 test 集不出现 key error
                all_vals = pd.concat([train_df[c], test_df[c]]).fillna(-1).astype(float).astype(int)
                uniq = sorted(list(all_vals.unique()))
                enc = {v: i for i, v in enumerate(uniq)}
                encoders[c] = enc
                size = len(enc)
            description.append((c, size, 'spr'))
            
        for c in ctn_cols:
            description.append((c, 1, 'ctn'))
            
        if len(hist_cols) > 0:
            description.append((seq_base, len(item_map)+1, 'seq'))
            
        description.append(('clk_label', 2, 'label'))
        description.append(('pay_label', 2, 'label'))
        self.description = description
        
        # 6. 构建最终的编码字典
        all_encoders = dict(encoders)
        if len(seq_cols) > 0:
            all_encoders[seq_base] = encoders['item_id']
            
        # 7. 创建 Dataset 和 DataLoader
        train_dataset = DINDataset(train_df, description, all_encoders, device)
        test_dataset = DINDataset(test_df, description, all_encoders, device)
        
        self.dataloaders = {
            'train': DataLoader(train_dataset, batch_size=bsz, shuffle=shuffle),
            'test': DataLoader(test_dataset, batch_size=bsz, shuffle=False)
        }

    def __getitem__(self, name):
        """
        获取指定的 dataloader ('train' 或 'test')
        """
        return self.dataloaders[name]

class AttentionUnit(nn.Module):
    """
    DIN 核心注意力单元
    """
    def __init__(self, embed_dim, hidden_dims=(64, 32), dropout=0.2):
        super().__init__()
        # 输入包括：target, history, target-history, target*history
        self.mlp = MultiLayerPerceptron(4 * embed_dim, hidden_dims, dropout, output_layer=True)

    def forward(self, target, history):
        B, T, D = history.shape
        # 扩展目标向量以匹配历史序列长度
        target_expanded = target.unsqueeze(1).expand(-1, T, -1) # [B, T, D]
        
        # 构造特征：[q, h, q-h, q*h]
        combined = torch.cat([
            target_expanded, 
            history, 
            target_expanded - history, 
            target_expanded * history
        ], dim=-1) # [B, T, 4*D]
        
        # 计算每个位置的分数
        # DIN 标准做法：不使用 Softmax 归一化，保留兴趣强度
        scores = self.mlp(combined.view(-1, 4 * D)).view(B, T)
        return scores

class DINModel(nn.Module):
    def __init__(self, description, embed_dim=4, mlp_dims=(64, 32), dropout=0.2, item_id_name='item_id', seq_name='hist_seq'):
        super().__init__()
        self.raw_description = description
        self.item_id_name = item_id_name
        self.seq_name = seq_name
        
        # 1. 共享 Embedding 层
        item_vocab_size = None
        for name, size, type in description:
            if name == item_id_name:
                item_vocab_size = size
                break
        self.shared_item_embed = nn.Embedding(item_vocab_size, embed_dim)
        
        # 2. 其他特征 Embedding
        self.embed_layers = nn.ModuleDict()
        input_dim = 0
        for name, size, type in description:
            if type == 'spr':
                if name == item_id_name:
                    input_dim += embed_dim
                else:
                    self.embed_layers[name] = nn.Embedding(size, embed_dim)
                    input_dim += embed_dim
            elif type == 'ctn':
                input_dim += 1
            elif type == 'seq':
                input_dim += embed_dim # 兴趣向量维度
        
        # 3. 注意力单元
        self.attention_unit = AttentionUnit(embed_dim, hidden_dims=(32, 16), dropout=dropout)
        
        # 4. 双头预测 MLP (点击头和购买头)
        self.clk_mlp = MultiLayerPerceptron(input_dim, mlp_dims, dropout)
        self.pay_mlp = MultiLayerPerceptron(input_dim, mlp_dims, dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.1, 0.1)

    def forward(self, x_dict):
        all_features = []
        qi = None 
        H = None  
        
        for name, size, type in self.raw_description:
            if type == 'spr':
                x = x_dict[name].squeeze(-1) 
                if name == self.item_id_name:
                    emb = self.shared_item_embed(x) 
                    qi = emb
                else:
                    emb = self.embed_layers[name](x)
                all_features.append(emb)
            elif type == 'ctn':
                all_features.append(x_dict[name])
            elif type == 'seq':
                seq = x_dict[name] 
                H = self.shared_item_embed(seq) 
                
        # 2. 注意力计算 (标准 DIN：加权求和但不做 Softmax)
        scores = self.attention_unit(qi, H) # [B, T]
        mask = (x_dict[self.seq_name] > 0).float()
        # 将 Mask 作用于分数（补零位置权重设为 0）
        scores = scores * mask
        
        # 加权求和得到兴趣向量
        interest = (scores.unsqueeze(-1) * H).sum(dim=1) # [B, embed_dim]
        
        # 3. 拼接并预测
        all_features.append(interest)
        x = torch.cat(all_features, dim=1) 
        
        # 4. 双头输出
        y_clk = torch.sigmoid(self.clk_mlp(x).squeeze(1))
        y_pay = torch.sigmoid(self.pay_mlp(x).squeeze(1))
        
        return y_clk, y_pay

def test(model, data_loader, device):
    """
    在测试集上评估模型性能 (AUC)
    """
    model.eval()
    clk_labels, clk_scores = list(), list()
    pay_labels, pay_scores = list(), list()
    with torch.no_grad():
        for _, (features, label_dict) in enumerate(data_loader):
            features = {key: value.to(device) for key, value in features.items()}
            y_clk, y_pay = model(features)
            
            clk_labels.extend(label_dict['clk_label'].tolist())
            clk_scores.extend(y_clk.tolist())
            pay_labels.extend(label_dict['pay_label'].tolist())
            pay_scores.extend(y_pay.tolist())
            
    clk_auc = roc_auc_score(clk_labels, clk_scores)
    pay_auc = roc_auc_score(pay_labels, pay_scores)
    return clk_auc, pay_auc

def train(model, data_loader, device, epoch, lr, weight_decay):
    """
    模型训练主循环
    """
    model.train()
    # 使用标准二元交叉熵损失
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    
    for epoch_i in range(1, epoch + 1):
        epoch_loss = 0.0
        total_iters = len(data_loader)
        for i, (features, label_dict) in enumerate(data_loader):
            # 明确将特征和标签移动到指定设备
            features = {key: value.to(device) for key, value in features.items()}
            clk_label = label_dict['clk_label'].to(device).squeeze(1) # [B]
            pay_label = label_dict['pay_label'].to(device).squeeze(1) # [B]
            
            # --- 1. 点击任务更新 (CTR Update) ---
            # 只更新共享参数和 clk_mlp
            optimizer.zero_grad()
            y_clk, _ = model(features)
            loss_clk = criterion(y_clk, clk_label)
            loss_clk.backward()
            optimizer.step()
            
            # --- 2. 购买任务更新 (CVR Update) ---
            # 只更新共享参数和 pay_mlp
            optimizer.zero_grad()
            # 重新前向传播以应用 CTR 更新后的共享参数
            _, y_pay = model(features)
            loss_pay = criterion(y_pay, pay_label)
            loss_pay.backward()
            optimizer.step()
            
            loss = loss_clk + loss_pay
            epoch_loss += loss.item()
            # 打印训练进度
            if (i + 1) % 10 == 0:
                print(f"    iters {i+1}/{total_iters} loss: {loss.item():.4f} (clk: {loss_clk.item():.4f}, pay: {loss_pay.item():.4f})", end='\r')
        
        print(f"Epoch {epoch_i}/{epoch} 平均训练损失: {epoch_loss/total_iters:.4f}")

if __name__ == '__main__':
    # 1. 获取参数和设置随机种子
    args = get_args()
    if args.seed > -1:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 2. 加载数据并初始化模型
    print("正在加载数据...")
    dataloaders = DINDataLoaders(args.train_path, args.test_path, device, bsz=args.bsz, shuffle=(args.shuffle == 1))
    
    print("正在构建模型...")
    model = DINModel(dataloaders.description).to(device)
    
    # 3. 初始评估 (未训练状态)
    clk_auc, pay_auc = test(model, dataloaders['test'], device)
    print(f"训练前测试集评估 -> Click AUC: {clk_auc:.4f}, Pay AUC: {pay_auc:.4f}")
    
    # 4. 开始训练
    print("开始模型训练...")
    train(model, dataloaders['train'], device, args.epoch, args.lr, args.weight_decay)
    
    # 5. 最终评估
    clk_auc, pay_auc = test(model, dataloaders['test'], device)
    print(f"训练后测试集评估 -> Click AUC: {clk_auc:.4f}, Pay AUC: {pay_auc:.4f}")

























```


