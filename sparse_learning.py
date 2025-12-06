import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from torch_geometric.nn import GATConv, global_add_pool, GCNConv
from torch_geometric.data import Batch
import auc_ci


class LearnableGraphSparsifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_nodes, num_edges, num_classes, node_names=None):
        super(LearnableGraphSparsifier, self).__init__()

        self.num_edges = num_edges
        self.num_nodes = num_nodes
        # 创建可学习的边权重参数（一维张量，长度为原始图的边数量）
        self.edge_weights = nn.Parameter(torch.ones(self.num_edges))

        # 节点重要性权重
        self.node_importance = nn.Parameter(torch.ones(self.num_nodes))

        # GNN层
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # GAT层
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=1)

        # 分类器
        self.nn = nn.Linear(hidden_dim * num_nodes, 32)
        self.classifier = nn.Linear(32, num_classes)  # 假设281是最大节点数

        # 节点名称（用于可视化）
        self.node_names = node_names

    def forward(self, x, edge_index, batch):
        """
        Forward propagation function

        Parameters:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices of batched graphs [2, num_edges_in_batch]
            batch: Batch indices [num_nodes]
        """

        edge_weight = torch.where(
            self.edge_weights < 0.2,
            torch.zeros_like(self.edge_weights),  # 小于0.2 → 0
            torch.clamp(self.edge_weights, max=1.0)  # 其他 → 保持或截断到1
        )
        # edge_weight = torch.sigmoid(self.edge_weights)
        # edge_weight = torch.where(
        #     self.edge_weights > 0.5,
        #     torch.ones_like(self.edge_weights),  # greater than 0.5 → 1
        #     torch.zeros_like(self.edge_weights)  # otherwise → 0
        # )
        # edge_weight = torch.clamp(self.edge_weights, min=0, max=1)
        node_importance = torch.clamp(self.node_importance, min=0, max=1)
        # node_importance = torch.where(
        #     self.node_importance < 0.2,
        #     torch.zeros_like(self.node_importance),  # less than 0.2 → 0
        #     torch.clamp(self.node_importance, max=1.0)  # others → keep or clamp to 1
        # )

        # 计算当前批次中的图数量
        batch_size = edge_index.size(1) // self.num_edges

        # 扩展边权重以匹配批次中的所有图
        expanded_edge_weight = edge_weight.repeat(batch_size)
        # 应用节点重要性
        expanded_node_importance = self.node_importance.repeat(batch_size).view(-1, 1)
        x = x * expanded_node_importance

        # 第一卷积层，使用原始图学习的边权重
        # 注意：这里假设batch中的所有图都有相同的结构
        x = self.conv1(x, edge_index, expanded_edge_weight)
        x = F.relu(x)

        # # Optional second convolutional layer
        # edge_index_filter = edge_index[:, edge_weight.repeat(batch_size) > 0.8]
        # x = self.gat1(x, edge_index_filter)
        # x = F.relu(x)

        # 将每个图的所有节点特征拼接为图级特征
        batch_size = batch.max().item() + 1
        graph_features = []

        for i in range(batch_size):
            # 获取当前图的节点索引
            graph_mask = batch == i
            graph_nodes = x[graph_mask]  # 当前图的节点特征

            # 节点数不足时填充零向量
            max_nodes = self.num_nodes
            if graph_nodes.size(0) < max_nodes:
                padding = torch.zeros(max_nodes - graph_nodes.size(0), graph_nodes.size(1), device=x.device)
                graph_nodes = torch.cat([graph_nodes, padding], dim=0)
            else:
                graph_nodes = graph_nodes[:max_nodes]  # 截断超过的节点

            # 展平为一维向量
            graph_feature = graph_nodes.view(-1)
            graph_features.append(graph_feature)

        # 合并所有图的特征
        x = torch.stack(graph_features, dim=0)

        # 分类器
        x = self.nn(x)
        out = self.classifier(x)

        # 返回分类结果、边权重和节点重要性
        return out, edge_weight, node_importance

    def get_sparse_subgraph(self, original_edge_index):
        """Get sparse subgraph structure and important nodes"""
        edge_weight = torch.clamp(self.edge_weights, min=0, max=1)
        node_importance = torch.clamp(self.node_importance, min=0, max=1)

        sparse_adj = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(edge_weight.size(0)):
            if edge_weight[i].item() > 1e-3:
                sparse_adj[original_edge_index[0, i].item(), original_edge_index[1, i].item()] = edge_weight[i].item()
        return sparse_adj, node_importance


# Training function correction
def train_model(model, train_loader, optimizer, device,
                class_reg=0.75,
                edge_reg=0.01,
                node_reg=0.03,
                target_num_edges=100,
                target_num_nodes=30,
                targe_edges_reg=1,
                target_node_reg=2,
                verbose=True):
    model.train()
    total_loss = 0
    avg_non_zero_weights = -1
    avg_weights = -1
    avg_node_importance = -1
    avg_non_zero_node_importance = -1

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        # 前向传播
        out, edge_weight, node_importance = model(data.x, data.edge_index, data.batch)

        # Calculate classification loss
        y = torch.tensor(data.y).to(torch.long).to(device)
        class_loss = F.cross_entropy(out, y)


        # Calculate sparsity regularization loss (L1 regularization)
        edge_sparsity_loss = torch.sum(torch.abs(edge_weight))
        node_sparsity_loss = torch.sum(torch.abs(node_importance))

        edge_count = edge_weight.size(0)
        non_zero_edge_count = (edge_weight > 1e-3).sum().item()
        num_nodes = node_importance.size(0)
        num_non_zero_nodes = (node_importance > 1e-3).sum().item()
        edge_target_loss = (non_zero_edge_count - target_num_edges) ** 2
        node_target_loss = (num_non_zero_nodes - target_num_nodes) ** 2

        # Total loss
        loss = (class_reg * class_loss  * data.num_graphs # classification loss
                + edge_reg * edge_sparsity_loss # edge weight sparsification loss
                + node_reg * node_sparsity_loss # node importance sparsification loss
                + targe_edges_reg * edge_target_loss # non-zero edge count control loss
                + target_node_reg * node_target_loss # non-zero node count control loss
                )

        total_loss += loss.item()

        if avg_weights == -1 and avg_non_zero_weights == -1:
            non_zero_weights = edge_weight[edge_weight > 1e-3].sum().item()
            total_weights = edge_weight.sum().item()
            avg_weights = total_weights/edge_count
            if non_zero_edge_count == 0:
                avg_non_zero_weights = 0
            else:
                avg_non_zero_weights = non_zero_weights/non_zero_edge_count

        if avg_node_importance == -1 and avg_non_zero_node_importance == -1:

            non_zero_node_importance = node_importance[node_importance > 1e-3].sum().item()
            total_node_importance = node_importance.sum().item()
            avg_node_importance = total_node_importance/num_nodes
            if num_non_zero_nodes == 0:
                avg_non_zero_node_importance = 0
            else:
                avg_non_zero_node_importance = non_zero_node_importance/num_non_zero_nodes

        # Backward propagation
        loss.backward()
        optimizer.step()



    if verbose:
        print(f"Number of edges in original graph structure: {edge_count} Total nodes: {num_nodes}")
        print(f"Number of non-zero edges: {non_zero_edge_count} ({non_zero_edge_count / edge_count * 100:.2f}%) "
              f"Number of non-zero nodes： {num_non_zero_nodes} ({num_non_zero_nodes/num_nodes * 100:.2f}%)")
        print(f"Average non-zero edge weight: {avg_non_zero_weights:.4f}, Total average edge weight: {avg_weights:.4f} "
              f"Average non-zero node importance: {avg_non_zero_node_importance:.4f}, Total average node importance: {avg_node_importance:.4f}")

    return total_loss / len(train_loader.dataset)


# Testing function
def test_model(model, loader, device, save=0, path=None):
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out, _, _ = model(data.x, data.edge_index, data.batch)

            # Calculate accuracy
            pred = out.argmax(1)
            y = data.y.to(torch.long).to(device)
            correct += (pred == y).sum().item()

            # Collect all prediction probabilities and true labels for subsequent metric calculation
            all_preds.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy())  # Probability of positive class
            all_labels.extend(y.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    binary_preds = np.round(all_preds)  # Convert probabilities to binary predictions

    # Calculate various metrics
    accuracy = correct / len(all_labels)
    roc_auc = roc_auc_score(all_labels, all_preds)
    precision = precision_score(all_labels, binary_preds)
    recall = recall_score(all_labels, binary_preds)
    f1 = f1_score(all_labels, binary_preds)

    auc_mean, ci_low, ci_up = auc_ci.bootstrap_auc_ci(all_labels, all_preds)
    if save == 1:
        df = pd.DataFrame(all_labels, columns=["labels"])
        df["preds"] = list(all_preds)
        df["binary_preds"] = list(binary_preds)
        df.to_excel(os.path.join(path, "GNN_resuls.xlsx"))
    return roc_auc, accuracy, precision, recall, f1, auc_mean, ci_low, ci_up


