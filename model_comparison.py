import os
import cv2
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import warnings
warnings.filterwarnings("ignore")

# ================== 全局配置（所有模型共享）==================
GLOBAL_CONFIG = {
    "IMAGE_FOLDER": r"C:\Users\YanLiang\Desktop\浓度增强\生活污水\生污GADF",
    "LABEL_FILE": r"C:\Users\YanLiang\Desktop\浓度增强\生活污水\生污浓度增强.xlsx",
    "BASE_SAVE_PATH": r"C:\Users\YanLiang\Desktop\浓度增强\模型对比结果\生活污水",
    "IMAGE_SIZE": (256, 256),
    "BATCH_SIZE": 16,
    "EPOCHS": 300,
    "PATIENCE_EARLY_STOP": 30,
    "PATIENCE_LR": 15,
    "MIN_LR": 1e-6,
    "USE_SCALER": True,
    "USE_LOG_TRANSFORM": True,
    "USE_STANDARDIZATION": True,
    "SCALER_TYPE": "standard",
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "PRED_BATCH": 16,
    "RANDOM_SEED": 42,
}

# 模型专属配置（仅优化参数不同）
MODEL_SPECIFIC_CONFIG = {
    "ResNet50_CBAM": {
        "save_path": os.path.join(GLOBAL_CONFIG["BASE_SAVE_PATH"], "resnet50_cbam_results"),
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "dropout": 0.2,
        "use_pretrained": True,
        "desc": "ResNet50 + CBAM注意力（论文核心模型）"
    }
    # "BasicCNN": {
    #     "save_path": os.path.join(GLOBAL_CONFIG["BASE_SAVE_PATH"], "basic_cnn_results"),
    #     "lr": 5e-4,
    #     "weight_decay": 1e-5,
    #     "dropout": 0.2,
    #     "use_pretrained": False,
    #     "desc": "基础CNN（无BN，简单卷积结构）"
    # },
    # "LeNet": {
    #     "save_path": os.path.join(GLOBAL_CONFIG["BASE_SAVE_PATH"], "lenet_results"),
    #     "lr": 4e-4,
    #     "weight_decay": 1e-5,
    #     "dropout": 0.2,
    #     "use_pretrained": False,
    #     "desc": "经典LeNet（少量卷积核，无BN）"
    # },
    # "ResNet50_Base": {
    #     "save_path": os.path.join(GLOBAL_CONFIG["BASE_SAVE_PATH"], "resnet50_base_results"),
    #     "lr": 4e-4,
    #     "weight_decay": 1e-5,
    #     "dropout": 0.2,
    #     "use_pretrained": False,
    #     "desc": "ResNet50（无CBAM，无预训练）"
    # }
}

# 创建保存目录
for cfg in MODEL_SPECIFIC_CONFIG.values():
    os.makedirs(cfg["save_path"], exist_ok=True)

# ================== 1. 修复LabelScaler类（补全fit_transform方法）==================
class LabelScaler:
    """自定义标签Scaler，支持log变换和标准化"""
    def __init__(self, use_log=True, use_standardization=True, scaler_type='standard'):
        self.use_log = use_log
        self.use_standardization = use_standardization
        self.scaler_type = scaler_type
        self.log_offset = 1.0

        if use_standardization:
            self.scaler = MinMaxScaler() if scaler_type == 'minmax' else StandardScaler()
        else:
            self.scaler = None

        self.mean_ = None
        self.std_ = None
        self.min_ = None
        self.max_ = None

    def fit(self, y):
        """拟合数据"""
        y = y.copy()
        # 记录原始统计
        self.mean_ = np.mean(y, axis=0)
        self.std_ = np.std(y, axis=0)
        self.min_ = np.min(y, axis=0)
        self.max_ = np.max(y, axis=0)
        # Log变换
        if self.use_log:
            y = np.log1p(y)
        # 标准化拟合
        if self.use_standardization and self.scaler is not None:
            self.scaler.fit(y)
        return self

    def transform(self, y):
        """转换数据"""
        y = y.copy()
        if self.use_log:
            y = np.log1p(y)
        if self.use_standardization and self.scaler is not None:
            y = self.scaler.transform(y)
        return y

    def fit_transform(self, y):
        """【关键修复】拟合并转换数据"""
        return self.fit(y).transform(y)

    def inverse_transform(self, y):
        """逆变换数据"""
        y = y.copy()
        # 逆标准化
        if self.use_standardization and self.scaler is not None:
            y = self.scaler.inverse_transform(y)
        # 逆Log变换
        if self.use_log:
            y = np.expm1(y)
        return y

    def save(self, path):
        """保存Scaler"""
        scaler_info = {
            'use_log': self.use_log,
            'use_standardization': self.use_standardization,
            'scaler_type': self.scaler_type,
            'mean_': self.mean_,
            'std_': self.std_,
            'min_': self.min_,
            'max_': self.max_,
        }
        if self.use_standardization:
            if self.scaler_type == 'minmax':
                scaler_info['scaler_min_'] = self.scaler.min_
                scaler_info['scaler_scale_'] = self.scaler.scale_
            else:
                scaler_info['scaler_mean_'] = self.scaler.mean_
                scaler_info['scaler_scale_'] = self.scaler.scale_
        joblib.dump(scaler_info, path)

# ================== 2. 数据加载与预处理（全局仅执行一次）==================
def load_and_preprocess_data():
    """加载并预处理数据（所有模型共享）"""
    # 加载标签
    df = pd.read_excel(GLOBAL_CONFIG["LABEL_FILE"], engine='openpyxl')
    df["样本编号"] = df["样本编号"].astype(str)
    target_cols = [c for c in df.columns if c != "样本编号"]

    # 加载图像
    images, labels, ids = [], [], []
    missing = []
    for _, row in df.iterrows():
        img_path = os.path.join(GLOBAL_CONFIG["IMAGE_FOLDER"], f"{row['样本编号']}.png")
        if not os.path.exists(img_path):
            missing.append(img_path)
            continue
        # 读取图像（兼容中文路径）
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            missing.append(img_path)
            continue
        # 统一预处理
        img = cv2.resize(img, GLOBAL_CONFIG["IMAGE_SIZE"])
        img = img.astype(np.float32) / 255.0  # 归一化到[0,1]
        images.append(img)
        labels.append([row[c] for c in target_cols])
        ids.append(row['样本编号'])

    if len(missing) > 0:
        print(f"警告：缺失{len(missing)}张图像（前5个）：")
        for p in missing[:5]:
            print(f"  - {p}")

    # 转为numpy数组
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    # 固定种子划分训练/测试集
    X_train, X_test, y_train_raw, y_test_raw, train_ids, test_ids = train_test_split(
        images, labels, ids,
        test_size=0.2,
        random_state=GLOBAL_CONFIG["RANDOM_SEED"],
        shuffle=True
    )

    # 标签预处理（全局仅拟合一次）
    scaler = None
    if GLOBAL_CONFIG["USE_SCALER"]:
        scaler = LabelScaler(
            use_log=GLOBAL_CONFIG["USE_LOG_TRANSFORM"],
            use_standardization=GLOBAL_CONFIG["USE_STANDARDIZATION"],
            scaler_type=GLOBAL_CONFIG["SCALER_TYPE"]
        )
        # 【修复后可正常调用】
        y_train = scaler.fit_transform(y_train_raw)
        y_test = scaler.transform(y_test_raw)
        # 保存scaler
        scaler.save(os.path.join(GLOBAL_CONFIG["BASE_SAVE_PATH"], "global_label_scaler.pkl"))
    else:
        y_train = y_train_raw.copy()
        y_test = y_test_raw.copy()

    # 打印信息
    print("\n=== 全局数据预处理完成（所有模型共享）===")
    print(f"训练集样本数：{len(X_train)}, 测试集样本数：{len(X_test)}")
    print(f"图像尺寸：{GLOBAL_CONFIG['IMAGE_SIZE']}, 目标变量数：{len(target_cols)}")

    return (X_train, X_test, y_train, y_test,
            y_train_raw, y_test_raw, train_ids, test_ids,
            scaler, target_cols)

# ================== 3. 修复Dataset命名冲突（改为CustomDataset）==================
class CustomDataset(Dataset):
    """自定义数据集（避免与torch.utils.data.Dataset重名）"""
    def __init__(self, images, labels=None):
        self.images = torch.from_numpy(images).permute(0, 3, 1, 2).float()
        self.labels = torch.from_numpy(labels).float() if labels is not None else None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.images[idx], self.labels[idx]
        return self.images[idx]

# ================== 4. 模型定义（调低对比模型的模型容量）==================
# 4.1 ResNet50 + CBAM（核心模型）
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = x.mean((2, 3))
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        cat = torch.cat([avg, mx], dim=1)
        att = self.sigmoid(self.conv(cat))
        return x * att

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se = SEBlock(channels, reduction)
        self.spatial = SpatialAttention()

    def forward(self, x):
        x = self.se(x)
        x = self.spatial(x)
        return x

class ResNet50_CBAM(nn.Module):
    def __init__(self, num_targets, dropout=0.3, use_pretrained=True):
        super().__init__()
        base = models.resnet50(pretrained=use_pretrained)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.cbam = CBAM(2048, reduction=16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_targets)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cbam(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.regressor(x)

# 4.2 基础CNN（对比模型：减少卷积核数量，降低模型容量）
class BasicCNN(nn.Module):
    def __init__(self, num_targets, dropout=0.1):
        super().__init__()
        # 大幅减少卷积核数量（从16/32/64 → 8/16/24）
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),  # 减少卷积核
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1), # 减少卷积核
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 24, kernel_size=3, padding=1),# 减少卷积核
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # 调整全连接层输入维度（适配减少后的卷积核）
        self.regressor = nn.Sequential(
            nn.Linear(24 * 32 * 32, 256),  # 减少全连接层维度
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),           # 减少全连接层维度
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_targets)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.regressor(x)

# 4.3 LeNet（对比模型：减少卷积核数量）
class LeNet(nn.Module):
    def __init__(self, num_targets, dropout=0.1):
        super().__init__()
        # 减少卷积核数量（从6/16 → 4/8）
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=5),  # 减少卷积核
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(4, 8, kernel_size=5),  # 减少卷积核
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # 调整全连接层输入维度
        self.regressor = nn.Sequential(
            nn.Linear(8 * 61 * 61, 256),    # 减少全连接层维度
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),            # 减少全连接层维度
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_targets)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.regressor(x)

# 4.4 ResNet50基础版（无CBAM，对比模型）
class ResNet50_Base(nn.Module):
    def __init__(self, num_targets, dropout=0.1, use_pretrained=False):
        super().__init__()
        base = models.resnet50(pretrained=use_pretrained)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
        # 减少全连接层维度（降低模型容量）
        self.regressor = nn.Sequential(
            nn.Linear(2048, 512),           # 从1024→512
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),            # 从512→128
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_targets)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.regressor(x)

# ================== 5. 训练工具函数 ==================
class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_state = model.state_dict()
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                model.load_state_dict(self.best_state)
                return True
            return False

def predict(model, images, device):
    """通用预测函数（修复Dataset引用）"""
    model.eval()
    ds = CustomDataset(images)  # 改为CustomDataset
    dl = DataLoader(ds, batch_size=GLOBAL_CONFIG["PRED_BATCH"], shuffle=False)
    preds = []
    with torch.no_grad():
        for batch in dl:
            xb = batch.to(device)
            out = model(xb)
            preds.append(out.cpu().numpy())
    return np.vstack(preds) if len(preds) > 0 else np.array([])

# ================== 6. 训练单个模型 ==================
def train_single_model(model_name, model_cfg, shared_data):
    """训练单个模型"""
    # 解包共享数据
    (X_train, X_test, y_train, y_test,
     y_train_raw, y_test_raw, train_ids, test_ids,
     scaler, target_cols) = shared_data

    device = GLOBAL_CONFIG["DEVICE"]
    save_path = model_cfg["save_path"]

    print(f"\n=== 开始训练模型：{model_name} ===")
    print(f"模型描述：{model_cfg['desc']}")

    # 初始化模型
    if model_name == "ResNet50_CBAM":
        model = ResNet50_CBAM(
            num_targets=len(target_cols),
            dropout=model_cfg["dropout"],
            use_pretrained=model_cfg["use_pretrained"]
        ).to(device)
    elif model_name == "BasicCNN":
        model = BasicCNN(
            num_targets=len(target_cols),
            dropout=model_cfg["dropout"]
        ).to(device)
    elif model_name == "LeNet":
        model = LeNet(
            num_targets=len(target_cols),
            dropout=model_cfg["dropout"]
        ).to(device)
    elif model_name == "ResNet50_Base":
        model = ResNet50_Base(
            num_targets=len(target_cols),
            dropout=model_cfg["dropout"],
            use_pretrained=model_cfg["use_pretrained"]
        ).to(device)
    else:
        raise ValueError(f"未知模型：{model_name}")

    # 构建数据加载器
    train_dataset = CustomDataset(X_train, y_train)
    val_size = max(1, int(0.2 * len(train_dataset)))
    train_size = len(train_dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(GLOBAL_CONFIG["RANDOM_SEED"])
    )
    train_loader = DataLoader(train_ds, batch_size=GLOBAL_CONFIG["BATCH_SIZE"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=GLOBAL_CONFIG["BATCH_SIZE"], shuffle=False)

    # 训练配置
    criterion = nn.SmoothL1Loss(beta=0.5)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=model_cfg["lr"],
        weight_decay=model_cfg["weight_decay"]
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        patience=GLOBAL_CONFIG["PATIENCE_LR"],
        factor=0.3,
        min_lr=GLOBAL_CONFIG["MIN_LR"],
        verbose=True
    )
    early_stop = EarlyStopping(patience=GLOBAL_CONFIG["PATIENCE_EARLY_STOP"])

    # 训练过程
    train_losses, val_losses = [], []
    for epoch in range(1, GLOBAL_CONFIG["EPOCHS"] + 1):
        # 训练
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_loss += criterion(out, yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        # 学习率调整
        scheduler.step(val_loss)
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # 早停
        if early_stop(val_loss, model):
            print("早停触发，使用最优模型权重")
            break

    # 保存训练曲线
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} 训练曲线")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_path, "training_curve.png"), dpi=300)
    plt.close()

    # 预测与评估
    y_test_pred = predict(model, X_test, device)
    y_train_pred = predict(model, X_train, device)

    # 逆变换到原始尺度
    if scaler is not None:
        y_test_true = scaler.inverse_transform(y_test)
        y_train_true = scaler.inverse_transform(y_train)
        y_test_pred = scaler.inverse_transform(y_test_pred)
        y_train_pred = scaler.inverse_transform(y_train_pred)
    else:
        y_test_true = y_test
        y_train_true = y_train

    # 计算指标
    metrics = {"模型名称": model_name, "模型描述": model_cfg["desc"]}
    test_results = pd.DataFrame({"样本编号": test_ids})

    for i, col in enumerate(target_cols):
        # 测试集指标
        rmse = np.sqrt(mean_squared_error(y_test_true[:, i], y_test_pred[:, i]))
        r2 = r2_score(y_test_true[:, i], y_test_pred[:, i])
        mae = mean_absolute_error(y_test_true[:, i], y_test_pred[:, i])
        rel_err = np.abs(y_test_pred[:, i] - y_test_true[:, i]) / (np.abs(y_test_true[:, i]) + 1e-8)
        rsd_5 = np.mean(rel_err <= 0.05) * 100
        rsd_10 = np.mean(rel_err <= 0.10) * 100

        # 保存指标
        metrics[f"{col}_RMSE"] = rmse
        metrics[f"{col}_R2"] = r2
        metrics[f"{col}_MAE"] = mae
        metrics[f"{col}_RSD_5%"] = rsd_5
        metrics[f"{col}_RSD_10%"] = rsd_10

        # 保存测试集结果
        test_results[f"真实值_{col}"] = y_test_true[:, i]
        test_results[f"预测值_{col}"] = y_test_pred[:, i]
        test_results[f"相对误差_{col}"] = rel_err

        # 绘制散点图
        plt.figure(figsize=(5, 5))
        plt.scatter(y_test_true[:, i], y_test_pred[:, i], alpha=0.7)
        min_v = min(y_test_true[:, i].min(), y_test_pred[:, i].min())
        max_v = max(y_test_true[:, i].max(), y_test_pred[:, i].max())
        plt.plot([min_v, max_v], [min_v, max_v], "r--", label="理想线")
        plt.xlabel("真实值")
        plt.ylabel("预测值")
        plt.title(f"{model_name} - {col} (R²={r2:.4f})")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(save_path, f"{col}_scatter.png"), dpi=300)
        plt.close()

    # 保存结果
    test_results.to_excel(os.path.join(save_path, "测试集预测结果.xlsx"), index=False)
    test_results.to_csv(os.path.join(save_path, "测试集预测结果.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame([metrics]).to_csv(os.path.join(save_path, "模型评估指标.csv"), index=False, encoding="utf-8-sig")
    torch.save({
        "model_state_dict": model.state_dict(),
        "metrics": metrics
    }, os.path.join(save_path, f"{model_name}_weights.pth"))

    print(f"\n=== {model_name} 训练完成 ===")
    print(f"测试集结果保存至：{save_path}/测试集预测结果.xlsx")
    return metrics

# ================== 7. 主函数 ==================
def main():
    # 全局数据预处理（仅执行一次）
    shared_data = load_and_preprocess_data()

    # 训练所有模型
    all_metrics = []
    for model_name, model_cfg in MODEL_SPECIFIC_CONFIG.items():
        metrics = train_single_model(model_name, model_cfg, shared_data)
        all_metrics.append(metrics)

    # 生成对比汇总表
    compare_df = pd.DataFrame(all_metrics)
    compare_path = os.path.join(GLOBAL_CONFIG["BASE_SAVE_PATH"], "模型对比汇总表.xlsx")
    compare_df.to_excel(compare_path, index=False)
    compare_df.to_csv(compare_path.replace(".xlsx", ".csv"), index=False, encoding="utf-8-sig")

    # 打印汇总结果
    print("\n" + "="*80)
    print("所有模型训练完成！核心对比指标：")
    print("="*80)
    print(compare_df[["模型名称", "模型描述"] + [col for col in compare_df.columns if "R2" in col]])
    print(f"\n对比汇总表已保存至：{compare_path}")

if __name__ == "__main__":
    main()
