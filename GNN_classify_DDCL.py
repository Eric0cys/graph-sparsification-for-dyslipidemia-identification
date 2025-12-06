from os import device_encoding
import pickle
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import networkx as nx

import sparse_learning
import time
import architecture


path = 'your base path'
folder_name = "results_DDCL"




def dataset2loader(dataset, num_total, ratio_list=None, batch_size=32):
    if ratio_list is None:
        ratio_list = [0.7, 0.15, 0.15]
    num_train = int(num_total * ratio_list[0])
    num_val = int(num_total * ratio_list[1])
    num_test = num_total - num_train - num_val
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [num_train, num_val, num_test],
    )

    # train_ind = list(pd.read_excel(os.path.join(path, "train_ind.xlsx")).to_numpy()[:, 1])
    # val_ind = list(pd.read_excel(os.path.join(path, "val_ind.xlsx")).to_numpy()[:, 1])
    # test_ind = list(pd.read_excel(os.path.join(path, "test_ind.xlsx")).to_numpy()[:, 1])
    train_dataset = dataset[train_ind]
    val_dataset = dataset[val_ind]
    test_dataset = dataset[test_ind]

    # Create DataLoaders for each set
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training set
        pin_memory=True  # Accelerate GPU data transfer
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


def data2dataset(X, Y, weighted_A):
    rows, cols = np.where(weighted_A != 0)
    weights = weighted_A[rows, cols]
    row = rows.tolist()
    col = cols.tolist()
    weight = weights.tolist()
    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_weight = torch.tensor(weight, dtype=torch.long)

    # Added in G_level_classify function
    print("number of nodes:", X[0, :].shape[0])

    graph_list = []
    for i in range(X.shape[0]):
        graph = Data(
            x=X[i, :].unsqueeze(1).float(),
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=Y[i]
        )
        graph_list.append(graph)
    dataset = architecture.MyDataset(graph_list)
    return dataset, edge_index
#


# # Training function
# def train(model, train_loader):
#     model.train()
#     total_loss = 0
#     for data in train_loader:
#         data = data.to(device)
#         optimizer.zero_grad()
#         out = model(data.x, data.edge_index, data.batch)
#         loss = criterion(out, data.y)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * data.num_graphs
#     return total_loss / len(train_dataset)


# 测试函数
def test(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_weight, data.batch)

            # 计算准确率
            pred = out.argmax(1)
            y = torch.tensor(data.y).to(torch.long).to(device)
            correct += (pred == y).sum().item()

            # 收集所有预测概率和真实标签用于后续指标计算
            all_preds.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy())  # 正类的概率
            all_labels.extend(data.y.cpu().numpy())

    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    binary_preds = np.round(all_preds)  # 将概率转换为二分类预测

    # 计算各项指标
    accuracy = correct / len(all_labels)
    roc_auc = roc_auc_score(all_labels, all_preds)
    precision = precision_score(all_labels, binary_preds)
    recall = recall_score(all_labels, binary_preds)
    f1 = f1_score(all_labels, binary_preds)

    return roc_auc, accuracy, precision, recall, f1


def load_model(X_scaled, Y, weighted_adj, load_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(os.path.join(load_path, "full_model.pth"), weights_only=False).to(device)
    batch_size = 64
    # Load dataset
    dataset0, original_edge_index = data2dataset(torch.tensor(X_scaled), Y, weighted_adj)
    print(X_scaled.shape, Y.shape, len(dataset0))
    train_loader, val_loader, test_loader = dataset2loader(dataset0, len(Y), batch_size=batch_size)

    model.eval()
    _, train_acc, _, _, _, _, _, _ = sparse_learning.test_model(model, train_loader, device)
    _, val_acc, _, _, _, _, _, _ = sparse_learning.test_model(model, val_loader, device)
    print(f'Train Acc: {train_acc:.4f}')
    print(f'Val Acc: {val_acc:.4f}')
    roc_auc, test_acc, precision, recall, f1_score, auc_mean, ci_low, ci_up = (
        sparse_learning.test_model(model, test_loader, device, 1, load_path))
    print(f'The final model, ROC AUC: {roc_auc:.4f}, Test Acc: {test_acc:.4f}'
          f' Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}')
    print(f'AUC CI: {auc_mean:.4f} ( {ci_low:.4f} -- {ci_up:.4f} ) ')



def load_graph(weighted_causal, load_path, keep_ind, del_ind, node_name):
    n = weighted_causal.shape[0]
    sparse_adj = np.load(os.path.join(load_path, "sparse_adj.npy"))
    node_importance = torch.load(os.path.join(load_path, "node_importance.pt"), map_location=torch.device('cpu'))
    sparse_node = [i for i in range(len(keep_ind)) if node_importance[i] > 1e-3]
    original_index = [keep_ind[i] for i in sparse_node]
    sparse_del_ind = [i for i in range(n) if i in original_index or i in del_ind]
    # Convert DataFrame to float64 type
    weighted_causal = weighted_causal.astype(np.float64)
    weighted_causal.iloc[original_index, original_index] = sparse_adj
    final_adj = weighted_causal.iloc[sparse_del_ind, sparse_del_ind]
    sparse_DAG = nx.from_numpy_array(final_adj.to_numpy(), create_using=nx.DiGraph)
    for i in range(len(sparse_del_ind)):
        sparse_DAG.nodes[i]['name'] = node_name[sparse_del_ind[i]]
    nx.write_gexf(sparse_DAG, os.path.join(load_path, "full_graph.gexf"))



def G_level_classify(X_scaled, Y, weighted_adj, node_name, num_classes=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    num_node_features = 1
    num_nodes = X_scaled.shape[1]
    print("num_nodes: ", num_nodes)

    # Hyperparameter settings
    batch_size = 64
    hidden_dim = 1
    lr = 0.001
    epochs = 100
    dropout = 0.5
    select_ratio = 1

    # Load dataset
    dataset, original_edge_index = data2dataset(torch.tensor(X_scaled), Y, weighted_adj)
    print(X_scaled.shape, Y.shape, len(dataset))
    train_loader, val_loader, test_loader = dataset2loader(dataset, len(Y), batch_size=batch_size)

    model = sparse_learning.LearnableGraphSparsifier(
        input_dim=num_node_features,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        num_nodes=X_scaled.shape[1],
        num_edges=np.count_nonzero(weighted_adj),
        node_names=range(0, X_scaled.shape[1]),
        num_classes=num_classes,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train model
    for epoch in range(1, 61):
        loss = sparse_learning.train_model(model, train_loader, optimizer, device)
        if epoch % 1 == 0:
            _, train_acc, _, _, _, _, _, _ = sparse_learning.test_model(model, train_loader, device)
            _, val_acc, _, _, _, _ , _, _ = sparse_learning.test_model(model, val_loader, device)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}')
    roc_auc, test_acc, precision, recall, f1_score, auc_mean, ci_low, ci_up = sparse_learning.test_model(model, test_loader, device)
    print(f'The final model, ROC AUC: {roc_auc:.4f}, Test Acc: {test_acc:.4f}'
          f' Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}')
    print(f'AUC CI: {auc_mean:.4f} ( {ci_low:.4f} -- {ci_up:.4f} ) ')



    sparse_adj, node_importance = model.get_sparse_subgraph(original_edge_index)
    sparse_nodes = [i for i in range(X_scaled.shape[1]) if node_importance[i] > 1e-3]
    sparse_adj = sparse_adj[np.ix_(sparse_nodes, sparse_nodes)]
    sparse_DAG = nx.from_numpy_array(sparse_adj, create_using=nx.DiGraph)
    for i in range(len(sparse_nodes)):
        sparse_DAG.nodes[i]['name'] = node_name[sparse_nodes[i]]
        # sparse_DAG.nodes[i]['weight'] = node_importance[sparse_nodes[i]]
    return test_acc, sparse_DAG, node_importance, original_edge_index, sparse_adj, model

def main():
    del_ind = [273, 274, 265, 266, 267, 268]
    data = pd.read_excel(path + '/your dataset name.xlsx')
    v_typeCVP = pd.read_excel(path + '/nodes index and nodes name.xlsx').to_numpy()
    m, n = data.shape
    weighted_causal = pd.read_excel(path + '/' + folder_name + '/weighted_causal_DDCL1.xlsx')
    Y = data.iloc[:, 273].to_numpy()

    G = nx.from_numpy_array(weighted_causal.to_numpy(), create_using=nx.Graph)
    DiG = nx.from_numpy_array(weighted_causal.to_numpy(), create_using=nx.DiGraph)
    for i in range(n):
        G.nodes[i]['name'] = v_typeCVP[i, 1]
        DiG.nodes[i]['name'] = v_typeCVP[i, 1]


    keep_ind = [i for i in range(n)]
    keep_ind = [i for i in keep_ind if i not in del_ind]
    print("len(keep_ind): ", len(keep_ind))

    X = data.iloc[:, keep_ind].to_numpy()

    weighted_adj = weighted_causal.iloc[keep_ind, keep_ind].to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    mask = np.zeros((len(keep_ind), len(keep_ind)), dtype=int)
    indices = np.random.choice(len(keep_ind) * len(keep_ind), int(len(keep_ind)*len(keep_ind)*0.02), replace=False)
    mask.flat[indices] = 1
    weighted_adj = weighted_adj ^ mask


    robustness = False
    for i in range(1):
        start_time = time.time()
        torch.manual_seed(int(start_time))
        if robustness:
            save_dir = os.path.join(path, folder_name, "random_state0train_val_test", "robustness", str(start_time))
            acc, sparse_G, node_importance, edge_index, sparse_adj, model = (
                G_level_classify(X_noise,  # X_scaled,
                                 Y,
                                 weighted_adj,
                                 v_typeCVP[keep_ind, 1],
                                 num_classes=2))
        else:
            save_dir = os.path.join(path, folder_name, "random_state0train_val_test", str(start_time))
            acc, sparse_G, node_importance, edge_index, sparse_adj, model = (
                G_level_classify(X_scaled,
                                 Y,
                                 weighted_adj,
                                 v_typeCVP[keep_ind, 1],
                                 num_classes=2))
        os.makedirs(save_dir, exist_ok=True)
        nx.write_gexf(sparse_G, os.path.join(save_dir, str(acc)[:6] + ".gexf"))
        # Save node importance tensor
        torch.save(node_importance, os.path.join(save_dir, 'node_importance.pt'))
        pd.DataFrame(node_importance.detach().cpu().numpy()).to_excel(os.path.join(save_dir, 'node_importance.xlsx'),
                                                                        index=True)
        # Save edge index list and node names
        with open(os.path.join(save_dir, 'edge_index.pkl'), 'wb') as f:
            pickle.dump(edge_index, f)
        with open(os.path.join(save_dir, 'nodes_name.pkl'), 'wb') as f:
            pickle.dump(v_typeCVP[keep_ind, 1], f)
        # Save sparse adjacency matrix
        np.save(os.path.join(save_dir, 'sparse_adj.npy'), sparse_adj)
        # Save model
        torch.save(model.state_dict(), os.path.join(save_dir, 'model_weights.pth'))
        # Can also save the complete model (including structure)
        torch.save(model, os.path.join(save_dir, 'full_model.pth'))
        pd.DataFrame(weighted_adj).to_excel(os.path.join(save_dir, "weighted_adj.xlsx"), index=False)

if __name__ == "__main__":
    main()