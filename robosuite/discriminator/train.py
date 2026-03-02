import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import PUSequenceDataset
from model import SequenceFailureDetector, DetectorConfig
from metrics import evaluate_pu_metrics


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = PUSequenceDataset(data_dir="/home/dodo/Documents/DAggar/robosuite/data/PandaLift", history_len=4)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=16)
    
    cfg = DetectorConfig()
    model = SequenceFailureDetector(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        all_preds, all_labels = [], []
        
        for img_agent, states, actions, labels in loader:
            img_agent = img_agent.to(device)
            states = states.to(device)
            actions = actions.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(img_agent, states, actions)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            all_preds.append(torch.sigmoid(logits).detach())
            all_labels.append(labels)
            
        preds_tensor = torch.cat(all_preds)
        labels_tensor = torch.cat(all_labels)
        metrics = evaluate_pu_metrics(preds_tensor, labels_tensor)
        
        print(f"Epoch {epoch} | Loss: {total_loss/len(loader):.4f} | "
              f"Expert Recall: {metrics['expert_recall']:.3f} | "
              f"Unlabeled Positive Rate: {metrics['unlabeled_positive_rate']:.3f} | "
              f"PU AUC: {metrics['pu_auc']:.3f}")


if __name__ == "__main__":
    train()