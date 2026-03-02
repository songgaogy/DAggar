import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class PUSequenceDataset(Dataset):
    def __init__(self, data_dir: str, history_len: int = 4):
        self.history_len = history_len
        self.samples = []
        
        expert_files = glob.glob(f"{data_dir}/expert/*.hdf5")
        unlabeled_files = glob.glob(f"{data_dir}/unlabled/*.hdf5")

        self._load_files(expert_files, label=1.0, file_category="expert")
        self._load_files(unlabeled_files, label=0.0, file_category="unlabled")

    def _load_files(self, file_paths: list, label: float, file_category: str):
        for path in tqdm(file_paths, desc=f"reading {file_category} files..."):
            with h5py.File(path, 'r') as f:
                for demo_key in f['demos'].keys():
                    demo = f['demos'][demo_key]
                    states = demo['states'][:]
                    actions = demo['actions'][:]
                    img_agent = demo['observations']['agentview']['images'][:]
                    
                    # Create sliding windows
                    seq_len = self.history_len + 1
                    for i in range(len(states) - seq_len):
                        self.samples.append({
                            'state_seq': states[i : i + seq_len],
                            'img_agent_seq': img_agent[i : i + seq_len],
                            'action': actions[i + seq_len - 1], # Current action
                            'label': label
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.from_numpy(s['img_agent_seq']).permute(0, 3, 1, 2), # B, T, C, H, W
            torch.from_numpy(s['state_seq']).float(),
            torch.from_numpy(s['action']).float(),
            torch.tensor(s['label'], dtype=torch.float32)
        )