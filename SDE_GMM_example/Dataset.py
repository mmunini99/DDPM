import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import torch.utils.data as data
import torch



def generate_univariate_gmm_dataset(n_rows, n_samples_per_row, means, variances, weights):

    n_components = len(means)
    dataset = np.zeros((n_rows, n_samples_per_row))  # Initialize dataset

    for row_idx in range(n_rows):
        # Step 1: Choose components for each sample in the row
        component_indices = np.random.choice(
            np.arange(n_components), size=n_samples_per_row, p=weights
        )
        
        # Step 2: Sample from the selected Gaussian components
        for sample_idx, component in enumerate(component_indices):
            mean = means[component]
            std_dev = np.sqrt(variances[component])
            dataset[row_idx, sample_idx] = np.random.normal(loc=mean, scale=std_dev)
    
    return dataset

def scaling(dataset):
    scaler = StandardScaler()
    dataset_normalized = scaler.fit_transform(dataset)

    return dataset_normalized, scaler.mean_.mean(), scaler.scale_.mean()

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

def loader(dataset, batch_size):
    dataset = CustomDataset(dataset)
    dataset = data.DataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=False,
                              collate_fn=numpy_collate)

    return dataset

def build_dataset(n_data, n_sample, means, var, mix_factor, batch_size):
    dataset = generate_univariate_gmm_dataset(
                                                    n_rows=n_data,
                                                    n_samples_per_row=n_sample,
                                                    means=means,
                                                    variances=var,
                                                    weights=mix_factor
                                                )
    dataset, m, v = scaling(dataset)
    #dataset = torch.tensor(dataset)
    dataset = loader(dataset, batch_size)

    return dataset, m, v

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    



