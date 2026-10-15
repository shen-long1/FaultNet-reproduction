import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader

# ====================== 环境修复 ======================
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

# ====================== Dataset ======================
from dataset import FaultDataset

# ====================== Regression（训练 + 验证） ======================
from regression import train_model, load_and_validate_model

# ====================== 可选模型 ======================
from fault_detection.models.Inception import InceptionTime
from models.MLP import MLP
from models.LSTM import LSTMModel
from models.Transformer import TransformerModel
from models.Crossformer import Model as Crossformer
from models.PatchTST import Model as PatchTST
from models.LightTS import Model as LightTS
from models.TCN import TCN
from models.FaultNet import FaultNet
from models.iTransformer import Model as iTransformer
from models.FaultNet1 import FaultNet1


# =======================================================================
# 🧨 主程序入口
# =======================================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="FaultDetectionConfigs")

    # ============================================================
    # 基本运行控制
    # ============================================================
    parser.add_argument("--model_name", type=str, default="FaultNet")
    parser.add_argument("--is_train", type=int, default=0)  # 1=train, 0=test
    parser.add_argument("--is_resume", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--loss_func", type=str, default="MSE")
    parser.add_argument("--use_norm", type=int, default=1)

    # GPU
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--device", type=str, default="0")

    # ============================================================
    # 数据维度/序列长度
    # ============================================================
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--label_len", type=int, default=6)
    parser.add_argument("--pred_len", type=int, default=1)

    # 🔥 必须有（Crossformer 需要）
    parser.add_argument("--task_name", type=str, default="classification")

    parser.add_argument("--num_class", type=int, default=6)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--d_conv", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--num_kernels", type=int, default=6)

    parser.add_argument("--enc_in", type=int, default=6)
    parser.add_argument("--dec_in", type=int, default=6)
    parser.add_argument("--c_out", type=int, default=6)

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--e_layers", type=int, default=1)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--d_ff", type=int, default=64)

    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--distil", action="store_false", default=True)
    parser.add_argument("--freq", type=str, default="s")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--activation", type=str, default="gelu")

    # 🔥 Crossformer 必须
    parser.add_argument("--decomp_method", type=str, default='moving_avg')
    parser.add_argument("--down_sampling_layers", type=int, default=0)
    parser.add_argument("--down_sampling_window", type=int, default=1)
    parser.add_argument("--down_sampling_method", type=str, default=None)
    parser.add_argument("--seg_len", type=int, default=48)

    args = parser.parse_args()

    # ============================================================
    # GPU 设置
    # ============================================================
    if args.use_gpu and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        print(f"🟢 Using GPU {args.device}")
    else:
        args.use_gpu = False
        print("⚪ Using CPU Only")

    # ============================================================
    # 加载数据
    # ============================================================
    print("📌 Loading dataset...")
    train_dataset = FaultDataset("dataset/train_data.npz", normalize=args.use_norm)
    test_dataset = FaultDataset("dataset/test_data.npz", normalize=args.use_norm)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

    # ============================================================
    # 模型字典（保持统一接口）
    # ============================================================
    model_hash = {
        "MLP": MLP(args),
        "LSTM": LSTMModel(args),
        "TCN": TCN(args),
        "Transformer": TransformerModel(args),
        "Crossformer": Crossformer(args),
        "PatchTST": PatchTST(args),
        "iTransformer": iTransformer(args),
        "LightTS": LightTS(args),
        "Inception": InceptionTime(args),
        "FaultNet": FaultNet(args),
        "FaultNet1": FaultNet1(args),
    }

    if args.model_name not in model_hash:
        raise ValueError(f"Unknown model: {args.model_name}")

    model = model_hash[args.model_name]
    print(f"🚀 Selected Model: {args.model_name}")

    # ============================================================
    # 是否恢复训练
    # ============================================================
    if args.is_resume:
        resume_path = r"E:\电子科大研究生\2025-2026\容错论文1\fault-tolerant-control-main\fault_detection\checkpoints\best_model.pth"
        print(f"⏳ Loading checkpoint from: {resume_path}")
        model.load_state_dict(torch.load(resume_path, map_location="cpu"))

    # ============================================================
    # 训练模式
    # ============================================================
    if args.is_train == 1:
        print("🟦 Training Mode Enabled")
        train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=args.epochs,
            learning_rate=args.lr,
            loss_func=args.loss_func,
        )

    # ============================================================
    # 验证模式（自动生成健康指数 H(t) 图）
    # ============================================================
    else:
        print("🟩 Validation Mode Enabled")

        load_and_validate_model(
            model=model,
            model_path=r"E:\电子科大研究生\A容错控制\A容错1论文实验结果\故障检测\消融实验\完整mamba\exp38\best_model.pth",
            test_loader=test_loader,
            base_save_dir="results",
            loss_func=args.loss_func,
        )

