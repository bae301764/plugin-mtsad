import os
import argparse
import random
import numpy as np
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
from src.dataset import TimeDataset, get_loaders
from src.model import GDN
from src.anomaly_detection import *
import sys
import warnings
warnings.filterwarnings(action='ignore')
from train import *

class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding="utf-8")  # append 모드

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
        
        
def main():
    parser = argparse.ArgumentParser()
    # dataset configurations
    parser.add_argument("--dataset", type=str, default="swat", help="Dataset name.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio.")
    parser.add_argument("--slide_win", type=int, default=5, help="Slide window size.")
    parser.add_argument("--slide_stride", type=int, default=1, help="Slide window stride.")
    
    # model configurations
    parser.add_argument("--hid_dim", type=int, default=64, help="Hidden dimension.")
    parser.add_argument("--attn_hidden", type=int, default=256, help="Hidden dimension.")
    parser.add_argument("--out_layer_num", type=int, default=1, help="Number of out layers.")
    parser.add_argument("--out_layer_hid_dim", type=int, default=128, help="Out layer hidden dimension.")
    parser.add_argument("--heads", type=int, default=1, help="Number of heads.")
    parser.add_argument("--topk", type=int, default=15, help="The knn graph topk.")
    
    # training configurations
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    
    parser.add_argument("--device", type=int, default=0, help="Training cuda device, -1 for cpu.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--early_stop_patience", type=int, default=30, help="Early stop patience.")
    parser.add_argument("--model_checkpoint_path", type=str, default="./checkpoint_temp/", help="Model checkpoint path.")
    parser.add_argument("--lambda_attn", type=float, default=0.1, help="lambda of loss")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay.")
    
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    
    # config
    args = parser.parse_args()
    args.model_checkpoint_path = args.model_checkpoint_path + f'k{args.topk}/{args.lambda_attn}/'+ f'seed{args.random_seed}/'
    os.makedirs(args.model_checkpoint_path, exist_ok=True)
    log_dir = args.model_checkpoint_path
    
    # set random seed
    seed_everything(args.random_seed)
    
    device = torch.device("cpu") if args.device < 0 else torch.device("cuda:{}".format(args.device))
    
    print("Loading datasets...")
    train_dataset = TimeDataset(dataset_name=args.dataset,
                                mode='train',
                                slide_win=args.slide_win,
                                slide_stride=args.slide_stride)
    test_dataset = TimeDataset(dataset_name=args.dataset,
                               mode='test',
                               slide_win=args.slide_win,
                               slide_stride=args.slide_stride)

    train_loader, val_loader, test_loader = get_loaders(train_dataset,
                                                        test_dataset,
                                                        args.batch_size,
                                                        val_ratio=args.val_ratio)
    
    # build model   
    model = GDN(number_nodes=train_dataset.number_nodes,
                in_dim = train_dataset.input_dim,
                hid_dim = args.hid_dim,
                out_layer_hid_dim = args.out_layer_hid_dim,
                out_layer_num = args.out_layer_num,
                topk = args.topk,
                heads = args.heads,
                input_length=args.slide_win,
                attn_hidden=args.attn_hidden,
                device=device)
    model = model.to(device)
    
    # train and test
    if args.train:
        print('start train')
        train(args, model, train_loader, val_loader, device)
    if args.test:
        print('start test')
        log_path = os.path.join(log_dir, f"result.txt")
        sys.stdout = Logger(log_path)  
        model.eval()
        model.load_state_dict(torch.load(os.path.join(args.model_checkpoint_path, f'best_model.pth')), strict=False)
        test_results = test(model, test_loader, device, mode='test', args=args)
        test_performence(test_results)
    
    
if __name__ == '__main__':
    main()
        