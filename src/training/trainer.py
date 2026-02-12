import torch
import torch.optim as optim
import os
import time
from src.utils.metrics import masked_mae, masked_rmse, masked_mape

class Trainer:
    def __init__(self, model, config, device, logger):
        self.model = model
        self.config = config
        self.device = device
        self.logger = logger
        
        self.model.to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config['training']['learning_rate'], 
            weight_decay=config['training']['weight_decay']
        )
        
        self.criterion = masked_mae
        self.batch_size = config['data']['batch_size']
        self.checkpoint_dir = config['training']['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def train_epoch(self, dataloader, edge_index):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(self.device)
            y = y.to(self.device)
            edge_index = edge_index.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward
            # x: (B, T, N, F)
            preds = self.model(x, edge_index)
            # preds: (B, T_out, N, F_out)
            # y: (B, T_out, N, F_out)
            
            loss = self.criterion(preds, y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip'])
            
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(dataloader)

    def validate(self, dataloader, edge_index):
        self.model.eval()
        total_loss = 0
        total_rmse = 0
        total_mape = 0
        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(dataloader):
                x = x.to(self.device)
                y = y.to(self.device)
                edge_index = edge_index.to(self.device)
                
                preds = self.model(x, edge_index)
                
                loss = self.criterion(preds, y)
                rmse = masked_rmse(preds, y)
                mape = masked_mape(preds, y)
                
                total_loss += loss.item()
                total_rmse += rmse.item()
                total_mape += mape.item()
                
        n = len(dataloader)
        return total_loss / n, total_rmse / n, total_mape / n

    def train(self, train_loader, val_loader, edge_index, epochs):
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            start_time = time.time()
            
            train_loss = self.train_epoch(train_loader, edge_index)
            val_loss, val_rmse, val_mape = self.validate(val_loader, edge_index)
            
            end_time = time.time()
            
            self.logger.info(f"Epoch {epoch+1}/{epochs} | "
                             f"Train Loss: {train_loss:.4f} | "
                             f"Val Loss: {val_loss:.4f} | "
                             f"Val RMSE: {val_rmse:.4f} | "
                             f"Val MAPE: {val_mape:.4f} | "
                             f"Time: {end_time - start_time:.2f}s")
            
            # Checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, 'best_model.pth'))
                self.logger.info(f"Saved best model at epoch {epoch+1}")
            else:
                patience_counter += 1
                
            if patience_counter >= self.config['training']['patience']:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
