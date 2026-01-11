import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from create_dataset import dataset_format
from extract_ifg_from_lrm import load_graph_adj

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# ============ 超参数配置 ============
RANDOM_SEED = 42
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 30
NUM_LAYERS = 1
DROPOUT = 0.2
TRAIN_SIZES = [15000]


# ============ 数据处理 ============
class GraphDataset(Dataset):
    def __init__(self, mask_ids, input_x, labels):
        self.mask_ids = mask_ids
        self.input_x = input_x
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'mask_id': torch.tensor(self.mask_ids[idx][0], dtype=torch.long),
            'input_x': torch.tensor(self.input_x[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx][0], dtype=torch.long)
        }


# ============ 模型定义 ============
class MultiRelationGraphAttention(nn.Module):
    def __init__(self, d_model, feature_len, num_relations=2, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.feature_len = feature_len
        self.num_relations = num_relations

        self.WQ_node = nn.Linear(d_model, d_model)
        self.WQ_input = nn.Linear(self.feature_len, self.feature_len)
        self.WQ = nn.Linear(d_model + self.feature_len, d_model)

        self.WK = nn.ModuleList([nn.Linear(d_model + self.feature_len, d_model) for _ in range(num_relations)])
        self.WV = nn.ModuleList([nn.Linear(d_model + self.feature_len, d_model) for _ in range(num_relations)])

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(d_model)
        self.WF = nn.Linear(num_relations * d_model, d_model)

    def forward(self, node_embed, input_x, adj_list):
        """
        Args:
            node_embed: (batch_size, num_nodes, d_model)
            input_x: (batch_size, feature_len)
            adj_list: list of (batch_size, num_nodes, num_nodes)

        Returns:
            hidden: (batch_size, num_nodes, d_model)
        """
        batch_size, num_nodes, _ = node_embed.shape

        # 动态特征投影并扩展
        x1 = self.WQ_input(input_x)  # (batch_size, feature_len)
        x1 = x1.unsqueeze(1).expand(-1, num_nodes, -1)  # (batch_size, num_nodes, feature_len)

        q_concat = torch.cat([node_embed, x1], dim=-1)  # (batch_size, num_nodes, d_model+feature_len)
        # 共享查询投影
        q = self.WQ(q_concat)  # (batch_size, num_nodes, d_model)

        # 多关系编码
        all_h = []
        for r in range(self.num_relations):
            adj = adj_list[r]  # (batch_size, num_nodes, num_nodes)

            k = self.WK[r](q_concat)  # (batch_size, num_nodes, d_model)
            v = self.WV[r](q_concat)  # (batch_size, num_nodes, d_model)

            scores = torch.matmul(q, k.transpose(-1, -2)) / (self.d_model ** 0.5)

            scores = scores.masked_fill(adj == 0, -1e9)
            alpha = F.softmax(scores, dim=-1)

            m_r = torch.matmul(alpha, v)
            m_r = self.dropout(m_r)
            all_h.append(m_r)

        h_concat = torch.cat(all_h, dim=-1)
        h_fused = self.activation(self.WF(h_concat))

        hidden = self.layer_norm(node_embed + h_fused)

        return hidden


class MultiRelationGraphTransformer(nn.Module):

    def __init__(self, d_model, feature_len, num_layers=2, num_relations=2,
                 num_classes=3, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.layers = nn.ModuleList([
            MultiRelationGraphAttention(d_model, feature_len, num_relations, dropout)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, node_embed, input_x, adj_list):
        """
        Args:
            node_embed: (batch_size, num_nodes, d_model)
            input_x: (batch_size, feature_len)
            adj_list: list of (batch_size, num_nodes, num_nodes)

        Returns:
            hidden: (batch_size, num_nodes, d_model)
        """
        hidden = node_embed
        for layer in self.layers:
            hidden = layer(hidden, input_x, adj_list)
        return hidden

    def predict(self, node_embed, input_x, adj_list, mask_id):
        """
        Args:
            node_embed: (batch_size, num_nodes, d_model)
            input_x: (batch_size, feature_len)
            adj_list: list of (batch_size, num_nodes, num_nodes)
            mask_id: (batch_size,)

        Returns:
            logits: (batch_size, num_classes)
        """
        hidden = self.forward(node_embed, input_x, adj_list)

        batch_size = hidden.size(0)
        batch_idx = torch.arange(batch_size, device=hidden.device)
        node_hidden = hidden[batch_idx, mask_id]  # (batch_size, d_model)

        logits = self.classifier(node_hidden)  # (batch_size, num_classes)

        return logits


# ============ 训练器 ============
class Trainer:

    def __init__(self, model, device, save_dir='../checkpoints'):
        self.model = model
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.best_val_metric = 0
        self.best_epoch = 0
        self.best_metrics = {}

        self.best_test1_acc = 0
        self.best_test1_mf1 = 0
        self.best_test1_epoch = 0

        self.best_test2_acc = 0
        self.best_test2_mf1 = 0
        self.best_test2_epoch = 0

    def train_epoch(self, train_loader, optimizer, node_embed, adj_list):
        self.model.train()
        total_loss = 0

        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            optimizer.zero_grad()

            mask_id = batch['mask_id'].to(self.device)
            input_x = batch['input_x'].to(self.device)
            label = batch['label'].to(self.device)

            batch_size = mask_id.size(0)
            batch_node_embed = node_embed.unsqueeze(0).expand(batch_size, -1, -1)

            logits = self.model.predict(batch_node_embed, input_x, adj_list, mask_id)

            loss = F.cross_entropy(logits, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        return total_loss / len(train_loader)

    @torch.no_grad()
    def evaluate(self, data_loader, node_embed, adj_list, desc="Evaluating"):
        self.model.eval()
        all_preds = []
        all_labels = []

        progress_bar = tqdm(data_loader, desc=desc)
        for batch in progress_bar:
            mask_id = batch['mask_id'].to(self.device)
            input_x = batch['input_x'].to(self.device)
            label = batch['label'].to(self.device)

            batch_size = mask_id.size(0)
            batch_node_embed = node_embed.unsqueeze(0).expand(batch_size, -1, -1)

            logits = self.model.predict(batch_node_embed, input_x, adj_list, mask_id)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        mf1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        return acc, mf1

    def train(self, train_loader, val_loader, test_loaders, node_embed, adj_list,
              num_epochs=100, lr=0.001):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-6
        )

        print(f"\n{'=' * 80}")
        for epoch in range(num_epochs):
            # train
            train_loss = self.train_epoch(train_loader, optimizer, node_embed, adj_list)

            # valid
            val_acc, val_mf1 = self.evaluate(val_loader, node_embed, adj_list, desc="Validating")

            print(f"\n{'=' * 80}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'=' * 80}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val - Acc: {val_acc:.4f}, MF1: {val_mf1:.4f}")

            scheduler.step(val_acc)
            val_metric = val_acc + val_mf1
            if val_metric > self.best_val_metric:
                self.best_val_metric = val_metric
                self.best_epoch = epoch + 1
                self.best_metrics = {
                    'val_acc': val_acc,
                    'val_mf1': val_mf1
                }

                model_path = os.path.join(self.save_dir, 'best_model.pt')
                torch.save(self.model.state_dict(), model_path)
                print(f"✓ Saved best model (based on val) at epoch {epoch + 1}\n")

    def load_best_model(self, model_type='val'):
        model_names = {
            'val': 'best_model.pt'
        }
        model_path = os.path.join(self.save_dir, model_names.get(model_type, 'best_model.pt'))
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            print(f"✓ Loaded best model ({model_type}) from {model_path}")
        else:
            print(f"✗ Best model ({model_type}) not found at {model_path}")


# ============ 数据加载和预处理 ============
def load_data(program_name, train_size=15000, load_test=True):
    train_path = f'../datasets/{program_name}/dataset_train.csv'
    test1_path = f'../datasets/{program_name}/dataset_val_cleaned.csv'
    test2_path = f'../datasets/{program_name}/dataset_test_cleaned.csv'
    instruction_embeddings_path = f'../datasets/{program_name}/instr_embeddings.npy'

    loaded_embeddings = np.load(instruction_embeddings_path)
    node_nums = len(loaded_embeddings)

    graph_cfg, graph_rdg = load_graph_adj(program_name, node_nums)

    result = {
        'embeddings': loaded_embeddings,
        'graph_cfg': graph_cfg,
        'graph_rdg': graph_rdg
    }

    train_data_all = pd.read_csv(train_path, keep_default_na=False, low_memory=False)

    actual_train_size = min(train_size, len(train_data_all))
    train_data = train_data_all.sample(actual_train_size, random_state=RANDOM_SEED, axis=0)

    remaining_data = train_data_all.loc[list(set(train_data_all.index) - set(train_data.index))]
    val_size = min(15000, len(remaining_data))
    val_data = remaining_data.sample(val_size, random_state=RANDOM_SEED, axis=0)

    print(f"training set: {len(train_data)}, valid set: {len(val_data)}")

    mask_id_list_train, input_x_list_train, label_list_train = dataset_format(
        program_name,
        train_data,
        f'../datasets/{program_name}/encode_setting.json',
        f'../datasets/{program_name}/node_mapping.json',
        f'../datasets/{program_name}/inputs_di_count.csv')

    mask_id_list_val, input_x_list_val, label_list_val = dataset_format(
        program_name,
        val_data,
        f'../datasets/{program_name}/encode_setting.json',
        f'../datasets/{program_name}/node_mapping.json',
        f'../datasets/{program_name}/inputs_di_count.csv')

    feature_len = len(input_x_list_train[0])
    result['feature_len'] = feature_len
    result['train'] = (mask_id_list_train, input_x_list_train, label_list_train)
    result['val'] = (mask_id_list_val, input_x_list_val, label_list_val)

    if load_test:
        test_data = {}

        # load test1
        if os.path.exists(test1_path):
            test1_data = pd.read_csv(test1_path, keep_default_na=False, low_memory=False)
            print(f"test1: {len(test1_data)}")
            mask_id_list_test1, input_x_list_test1, label_list_test1 = dataset_format(
                program_name,
                test1_data,
                f'../datasets/{program_name}/encode_setting.json',
                f'../datasets/{program_name}/node_mapping.json',
                f'../datasets/{program_name}/inputs_di_count.csv')
            test_data['test1'] = (mask_id_list_test1, input_x_list_test1, label_list_test1)
        else:
            print(f"✗ Test1 file not found: {test1_path}")

        # load test2
        if os.path.exists(test2_path):
            test2_data = pd.read_csv(test2_path, keep_default_na=False, low_memory=False)
            print(f"test2: {len(test2_data)}")
            mask_id_list_test2, input_x_list_test2, label_list_test2 = dataset_format(
                program_name,
                test2_data,
                f'../datasets/{program_name}/encode_setting.json',
                f'../datasets/{program_name}/node_mapping.json',
                f'../datasets/{program_name}/inputs_di_count.csv')
            test_data['test2'] = (mask_id_list_test2, input_x_list_test2, label_list_test2)
        else:
            print(f"✗ Test2 file not found: {test2_path}")

        result['test_data'] = test_data

    return result


def append_to_txt(file_path, text):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(text + '\n')
        print(f"✓: {file_path}")
    except Exception as e:
        print(f"✗: {e}")


# ============ 主函数 ============
def main(num_epochs=10, num_layers=2, train_size=15000, PROGRAM_NAME="2mm"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"\n{'=' * 60}")
    print(f"Training program: {PROGRAM_NAME}")
    print(f"Training size: {train_size}")
    print(f"{'=' * 60}\n")

    data = load_data(PROGRAM_NAME, train_size=train_size, load_test=True)
    loaded_embeddings = data['embeddings']
    graph_cfg = data['graph_cfg']
    graph_rdg = data['graph_rdg']
    feature_len = data['feature_len']

    train_dataset = GraphDataset(*data['train'])
    val_dataset = GraphDataset(*data['val'])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    test_loaders = {}
    test_data_dict = data.get('test_data', {})
    for test_name, test_tuple in test_data_dict.items():
        test_dataset = GraphDataset(*test_tuple)
        test_loaders[test_name] = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    adj_cfg = torch.tensor(graph_cfg, dtype=torch.float32).to(device)
    adj_rdg = torch.tensor(graph_rdg, dtype=torch.float32).to(device)
    adj_list = [adj_cfg, adj_rdg]

    node_embed = torch.tensor(loaded_embeddings, dtype=torch.float32).to(device)

    d_model = loaded_embeddings.shape[1]
    if PROGRAM_NAME == "lud":
        NUM_CLASSES = 2
    else:
        NUM_CLASSES = 3

    model = MultiRelationGraphTransformer(
        d_model=d_model,
        feature_len=feature_len,
        num_layers=num_layers,
        num_relations=2,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT
    ).to(device)

    print(f"✓ IASCM model created with {num_layers} layers")
    print(f"  - d_model: {d_model}")
    print(f"  - feature_len: {feature_len}")
    print(f"  - num_classes: {NUM_CLASSES}\n")

    trainer = Trainer(model, device, save_dir=f'./checkpoints/{PROGRAM_NAME}')
    trainer.train(
        train_loader, val_loader, test_loaders,
        node_embed, adj_list,
        num_epochs=num_epochs,
        lr=LEARNING_RATE
    )

    results_file = './test_results.txt'

    append_to_txt(results_file, f"\n{'=' * 80}")
    append_to_txt(results_file, f"Program: {PROGRAM_NAME}, TrainSize: {train_size}")
    append_to_txt(results_file, f"{'=' * 80}")

    append_to_txt(results_file, f"Val - Acc: {trainer.best_metrics['val_acc']:.4f}, MF1: {trainer.best_metrics['val_mf1']:.4f}")

    if 'test_results' in trainer.best_metrics:
        for test_name, metrics in trainer.best_metrics['test_results'].items():
            result_text = f"{test_name} - Acc: {metrics['acc']:.4f}, MF1: {metrics['mf1']:.4f}"
            append_to_txt(results_file, result_text)

    append_to_txt(results_file, f"\n【Test1 best - Epoch {trainer.best_test1_epoch}】")
    append_to_txt(results_file, f"Test1 - Acc: {trainer.best_test1_acc:.4f}, MF1: {trainer.best_test1_mf1:.4f}")

    append_to_txt(results_file, f"\n【Test2 best - Epoch {trainer.best_test2_epoch}】")
    append_to_txt(results_file, f"Test2 - Acc: {trainer.best_test2_acc:.4f}, MF1: {trainer.best_test2_mf1:.4f}")

    append_to_txt(results_file, "=" * 80 + "\n")

    print(f"\n{'=' * 60}")
    print(f"Results saved to {results_file}")
    print(f"{'=' * 60}\n")

    test_results = {
        'best_val_model': {
            'epoch': trainer.best_epoch,
            'val_acc': trainer.best_metrics['val_acc'],
            'val_mf1': trainer.best_metrics['val_mf1'],
            'test_results': trainer.best_metrics.get('test_results', {})
        },
        'best_test1': {
            'epoch': trainer.best_test1_epoch,
            'acc': trainer.best_test1_acc,
            'mf1': trainer.best_test1_mf1
        },
        'best_test2': {
            'epoch': trainer.best_test2_epoch,
            'acc': trainer.best_test2_acc,
            'mf1': trainer.best_test2_mf1
        }
    }

    return model, trainer, test_results


if __name__ == '__main__':
    p_list = ["pathfinder", "nw", "nn", "mvt", "lud", "lavaMD", "gemm", "gaussian", "conv2d", "backprop", "atax", "2mm"]
    for program_name in p_list:
        print(f"\n\n{'#' * 80}")
        print(f"{'#' * 80}")
        print(f"Processing program: {program_name}")
        print(f"{'#' * 80}")
        print(f"{'#' * 80}\n")

        for train_size in TRAIN_SIZES:
            try:
                print(f"\n{'*' * 60}")
                print(f"Training with size: {train_size}")
                print(f"{'*' * 60}\n")

                model, trainer, test_results = main(
                    num_epochs=EPOCHS,
                    num_layers=NUM_LAYERS,
                    train_size=train_size,
                    PROGRAM_NAME=program_name
                )

                print(f"\n✓ Completed training with size {train_size}")

            except Exception as e:
                print(f"\n✗ Error processing {program_name} with train_size {train_size}: {str(e)}")
                import traceback

                traceback.print_exc()
                continue

    print(f"\n\n{'=' * 80}")
    print("All programs and training sizes processed!")
    print(f"{'=' * 80}\n")