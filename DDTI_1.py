import os
import time
import argparse
import torch
import torch.optim as optim
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from pytorch_lightning import LightningModule, Trainer
from torch.utils import data
from torch.utils.data import DataLoader


BATCH_SIZE = 16 
ACCUMULATE_GRAD_BATCHES = 8
MAX_EPOCHS = 120

DATA_ROOT = '/data/zqp/XR/data/DDTI_ture'
IMAGE_DIR = os.path.join(DATA_ROOT, 'image')
MASK_DIR = os.path.join(DATA_ROOT, 'mask')

SAVE_DIR = '/data/zqp/XR/DDTI_Classic_Final'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR, exist_ok=True)

ORIGINAL_PAPER_CONFIGS = {
    "Unet": "vgg16", 
    "UnetPlusPlus": "vgg19", 
    "DeepLabV3": "resnet50", 
    "DeepLabV3Plus": "resnet101", 
    "FPN": "resnet50", 
    "Linknet": "resnet18", 
    "PSPNet": "resnet50", 
    "PAN": "resnet50",
    "MANet": "resnet50"
}

def create_dataframe_from_directory(image_dir, mask_dir):
    img_list = sorted(os.listdir(image_dir))
    msk_list = sorted(os.listdir(mask_dir))
    if len(img_list) != len(msk_list):
        print(f"警告: 图片数量({len(img_list)})与掩码数量({len(msk_list)})不一致!")
    image_paths = [os.path.join(image_dir, i) for i in img_list]
    mask_paths = [os.path.join(mask_dir, m) for m in msk_list]
    return pd.DataFrame({'image': image_paths, 'mask': mask_paths})

def read_image_fu(lo_df):
    lo_dict = {'image': [], 'mask': []}
    for i in lo_df.index:
        df_row = lo_df.loc[i, :]
        img = cv2.imread(df_row['image'])
        if img is None: continue
        img = cv2.resize(img, (256, 256)).transpose([2, 0, 1]) 
        lo_dict['image'].append(img)
        mask_img = cv2.imread(df_row['mask'])
        if mask_img is None: continue
        mask_img = cv2.resize(mask_img, (256, 256)).transpose([2, 0, 1])
        mask_img = mask_img[:1, :, :] 
        mask_img[mask_img >= 1] = 1   
        lo_dict['mask'].append(mask_img)
    return lo_dict

class Datas(data.Dataset):
    def __init__(self, lo_dict):
        self.image = lo_dict['image']
        self.mask = lo_dict['mask']

    def __len__(self): return len(self.image)

    def __getitem__(self, index): return {'data': self.image[index], 'mask': self.mask[index]}

def make_dataloader():
    print("正在加载数据并划分数据集 (8:1:1)...")
    df = create_dataframe_from_directory(IMAGE_DIR, MASK_DIR)
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    total_len = df_shuffled.shape[0]
    train_end = int(0.8 * total_len)
    val_end = int(0.9 * total_len)
    train_df = df_shuffled[:train_end].reset_index(drop=True)
    valid_df = df_shuffled[train_end:val_end].reset_index(drop=True)
    test_df = df_shuffled[val_end:].reset_index(drop=True)
    print(f"数据划分完成: Train={len(train_df)}, Val={len(valid_df)}, Test={len(test_df)}")
    train_dict = read_image_fu(train_df)
    val_dict = read_image_fu(valid_df)
    test_dict = read_image_fu(test_df)

    train_loader = DataLoader(Datas(train_dict), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(Datas(val_dict), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(Datas(test_dict), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, test_loader


class ClassicModel(LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, lr, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.arch = arch
        self.encoder_name = encoder_name
        self.lr = lr
        self.model = smp.create_model(
            arch=arch,
            encoder_name=encoder_name,
            in_channels=in_channels, 
            classes=out_classes, 
            encoder_weights=None, 
            **kwargs
        )
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        
        self.step_outputs = {
            'train': {"tp": [], "fp": [], "fn": [], "tn": []},
            'valid': {"tp": [], "fp": [], "fn": [], "tn": []},
            'test':  {"tp": [], "fp": [], "fn": [], "tn": []}
        }
        self.history = {
            'epoch': [],
            'train_loss': [], 
            'val_iou': [], 'val_pre': [], 
            'val_f1': [], 'val_acc': [], 'val_spec': []
        }
        self.best_metrics = {'epoch': 0, 'IoU': 0.0, 'F1': 0.0, 'Acc': 0.0, 'Spec': 0.0, 'Pre': 0.0}
        self.epoch_loss_train = [] 

    def forward(self, image):
        if image.max() > 1: image = image.float() / 255.0
        image = (image - self.mean) / self.std
        return self.model(image)

    def shared_step(self, batch, stage):
        image, mask = batch['data'], batch['mask']
        if mask.max() > 1: mask = mask / 255.0
        mask = mask.float()
        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")
        self.step_outputs[stage]['tp'].append(tp)
        self.step_outputs[stage]['fp'].append(fp)
        self.step_outputs[stage]['fn'].append(fn)
        self.step_outputs[stage]['tn'].append(tn)
        if stage == 'train':
            self.epoch_loss_train.append(loss.item())
        else:
            self.log(f'{stage}_loss', loss, prog_bar=True)
        return loss

    def shared_epoch_end(self, stage):
        if len(self.step_outputs[stage]['tp']) == 0: return
        tp = torch.cat(self.step_outputs[stage]['tp'])
        fp = torch.cat(self.step_outputs[stage]['fp'])
        fn = torch.cat(self.step_outputs[stage]['fn'])
        tn = torch.cat(self.step_outputs[stage]['tn'])
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()
        f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro").item()
        acc = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro").item()
        spec = smp.metrics.specificity(tp, fp, fn, tn, reduction="micro").item()
        pre = smp.metrics.precision(tp, fp, fn, tn, reduction="micro").item()
        if stage == 'valid':
            avg_train_loss = np.mean(self.epoch_loss_train) if len(self.epoch_loss_train) > 0 else 0
            self.history['epoch'].append(self.current_epoch)
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_iou'].append(iou)
            self.history['val_pre'].append(pre)
            self.history['val_f1'].append(f1)
            self.history['val_acc'].append(acc)
            self.history['val_spec'].append(spec)
            if iou > self.best_metrics['IoU']:
                self.best_metrics = {
                    'epoch': self.current_epoch,
                    'IoU': iou, 'F1': f1, 'Acc': acc, 'Spec': spec, 'Pre': pre
                }
            self.epoch_loss_train = []
        for k in self.step_outputs[stage]: self.step_outputs[stage][k].clear()

    def training_step(self, batch, batch_idx): return self.shared_step(batch, 'train')

    def on_train_epoch_end(self): self.shared_epoch_end('train')

    def validation_step(self, batch, batch_idx): return self.shared_step(batch, 'valid')

    def on_validation_epoch_end(self): self.shared_epoch_end('valid')

    def test_step(self, batch, batch_idx): return self.shared_step(batch, 'test')

    def on_test_epoch_end(self): self.shared_epoch_end('test')
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'valid_loss'}

def plot_curves(history, model_name, save_dir):
    epochs = history['epoch']
    train_loss = history['train_loss']
    ious = history['val_iou']
    pres = history['val_pre']
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, ious, label='IoU')
    plt.plot(epochs, pres, label='Precision')
    plt.title(f'{model_name} Metrics')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_dir}/{model_name}_metrics.png')
    plt.close()
    if len(epochs) > 1:
        plt.figure(figsize=(10, 5))
        plt.plot(epochs[1:], train_loss[1:], label='Train Loss', color='red')
        plt.title(f'{model_name} Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{save_dir}/{model_name}_loss.png')
        plt.close()


if __name__ == '__main__':
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model Architecture Name')
    args = parser.parse_args()
    model_name = args.model
    if model_name not in ORIGINAL_PAPER_CONFIGS:
        raise ValueError(f"Model {model_name} not found in config!") 
    encoder_name = ORIGINAL_PAPER_CONFIGS[model_name]
    print(f"\n{'='*60}")
    print(f"[DDTI] 启动单任务: {model_name} (Backbone: {encoder_name})")
    print(f"Batch Size: {BATCH_SIZE} | Accumulate: {ACCUMULATE_GRAD_BATCHES}")
    print(f"{'='*60}\n")
    train_dl, val_dl, test_dl = make_dataloader()
    model = ClassicModel(
        arch=model_name, 
        encoder_name=encoder_name, 
        in_channels=3, 
        out_classes=1, 
        lr=0.001
    )
    trainer = Trainer(
        accelerator="gpu", 
        devices=[0], 
        max_epochs=MAX_EPOCHS,
        log_every_n_steps=10,
        enable_checkpointing=False,
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES
    )
    start_time = time.time()
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    end_time = time.time()
    best = model.best_metrics
    print(f"\n{model_name} 最佳表现 (Epoch {best['epoch']}):")
    print(f"IoU:{best['IoU']:.4f}")
    print(f"F1:{best['F1']:.4f}")
    plot_curves(model.history, model_name, SAVE_DIR)
    torch.save(model.state_dict(), f'{SAVE_DIR}/{model_name}_classic.pth')
    res = {
        "Model": model_name,
        "Encoder": encoder_name,
        "Best Epoch": best['epoch'],
        "IoU": round(best['IoU'], 4),
        "F1": round(best['F1'], 4),
        "Acc": round(best['Acc'], 4),
        "Spec": round(best['Spec'], 4),
        "Pre": round(best['Pre'], 4),
        "Time(h)": round((end_time - start_time) / 3600, 2)
    }
    df_res = pd.DataFrame([res])
    df_res.to_csv(f'{SAVE_DIR}/{model_name}_result.csv', index=False)
    df_history = pd.DataFrame(model.history)
    df_history.to_csv(f'{SAVE_DIR}/{model_name}_history.csv', index=False)
    print(f"{model_name} 任务完成！")