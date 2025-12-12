from torch_geometric.data import Dataset

class MyDataset(Dataset):
    def __init__(self, graph_list, transform=None, pre_transform=None):
        # Pass an empty string as root
        super().__init__('', transform, pre_transform)
        self.graph_list = graph_list
        # Optional: Apply preprocessing
        if self.pre_transform is not None:
            self.graph_list = [self.pre_transform(graph) for graph in self.graph_list]

    def len(self):
        return len(self.graph_list)

    def get(self, idx):
        return self.graph_list[idx]
