from torch.utils.data import DataLoader
from esim_data import SynonymDataset
data = [[1,2],[3,4],[5,6]]
dataset = SynonymDataset(data)
def test(d):
    print(test)
data_loader = DataLoader(dataset, batch_size=2, collate_fn=test)
for batch in data_loader:
    print(batch)