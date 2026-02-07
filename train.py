import torch
import os
import numpy as np

class TrainingHelper(object):
    """
    Helper class for training.
    1. save model if val loss is smaller than min val loss
    2. early stop if val loss does not decrease for early_stop_patience epochs
    """
    def __init__(self, args):
        self.min_val_loss = 1e+30
        self.early_stop_count = 0
        
        if not os.path.exists(args.model_checkpoint_path):
            os.makedirs(args.model_checkpoint_path)
        self.model_checkpoint_path = os.path.join(args.model_checkpoint_path,"best_model.pth")
        self.early_stop_patience = args.early_stop_patience
        
        self.early_stop: bool = False
        
    
    def check(self, val_loss: float, model) -> bool:
        # save model if val loss is smaller than min val loss
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            torch.save(model.state_dict(), self.model_checkpoint_path)
            self.early_stop_count = 0
        else:
            self.early_stop_count += 1
        
        if self.early_stop_count >= self.early_stop_patience:
            self.early_stop = True


def train(args, model, train_loader, val_loader, device):
    optimizer = torch.optim.Adam(model.parameters(),
                    lr=args.lr,
                    betas = (0.9, 0.99),
                    weight_decay=args.weight_decay)
    training_helper = TrainingHelper(args)
    
    print(args)
    print("Training...")
    for epoch in range(1, args.epochs):
        epoch_loss = []
        model.train()
        
        for idx, batch in enumerate(train_loader):
            x, y, _ = [item.to(device) for item in batch]
            optimizer.zero_grad()
            out, plugin_loss, _ = model(x)
            loss = model.total_loss(out,y,plugin_loss,args.lambda_attn)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        train_loss = np.mean(epoch_loss)
        
        
        val_loss = test(model, val_loader, device, mode='val', args=args)
        print("Epoch {:>3}/{} train loss: {:.8f}, val loss: {:.8f}".format(epoch, args.epochs, train_loss, val_loss))
        training_helper.check(val_loss, model)
        if training_helper.early_stop:
            print("Early stop at epoch {} with val loss {:.4f}.".format(epoch, training_helper.min_val_loss))
            print("Model saved at {}.".format(training_helper.model_checkpoint_path))
            break
    print('Final validation loss : ', training_helper.min_val_loss)



def test(model,
         loader: torch.utils.data.DataLoader,
         device: torch.device,
         mode: str = 'test',
         args=None):
    assert mode in ['test', 'val'], "mode must be 'test' or 'val'!"
    model.eval()
    loss_list, prediction_list, ground_truth_list, label_list, att_scores = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x, y, label = [item.to(device) for item in batch]
            out, plugin_loss, att_score= model(x)
            if mode == 'val':
                loss = model.total_loss(out,y, plugin_loss, args.lambda_attn)
                loss_list.append(loss.item())
            else:      
                prediction_list.append(out.cpu().numpy())
                ground_truth_list.append(y.cpu().numpy())
                label_list.append(label.cpu().numpy())
                att_scores.append(att_score.cpu().numpy())

    if mode == 'val':
        loss = np.mean(loss_list)
        return loss
    else:
        prediction = np.concatenate(prediction_list, axis=0) # size [total_time_len, num_nodes]
        ground_truth = np.concatenate(ground_truth_list, axis=0) # size [total_time_len, num_nodes]
        anomaly_label = np.concatenate(label_list, axis=0) # size [total_time_len]
        att_scores_list = np.concatenate(att_scores, axis=0) # size [total_time_len]
        return (prediction, ground_truth, anomaly_label, att_scores_list)
