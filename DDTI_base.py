import os
import argparse
import pickle
import traceback

import torch
import torch.optim as optim
import cv2
import pandas as pd
import segmentation_models_pytorch as smp
from pytorch_lightning import LightningModule, Trainer
from torch.utils import data
from torch.utils.data import DataLoader
DEFAULT_IMAGE_DIR = r"/data/zqp/XR/data/DDTI_ture/image"
DEFAULT_MASK_DIR  = r"/data/zqp/XR/data/DDTI_ture/mask"
DEFAULT_SAVE_DIR  = r"/data/zqp/XR/paper/DDTI"
def create_dataframe_from_directory(image_dir, mask_dir):
    image_paths, mask_paths = [], []
    for image_fold, mask_fold in zip(os.listdir(image_dir), os.listdir(mask_dir)):
        image_paths.append(os.path.join(image_dir, image_fold))
        mask_paths.append(os.path.join(mask_dir, mask_fold))
    return pd.DataFrame({'image': image_paths, 'mask': mask_paths})


def read_image_fu(lo_df, length=1000000):
    lo_dict = {'image': [], 'mask': []}
    lo_df = lo_df.sample(frac=1)
    for i in lo_df.index:
        df_row = lo_df.loc[i, :]
        img = cv2.imread(df_row['image'])
        mask_img = cv2.imread(df_row['mask'])
        if img is None or mask_img is None:
            continue
        lo_dict['image'].append(cv2.resize(img, (256, 256)).transpose([2, 0, 1])) 
        mask_img = cv2.resize(mask_img, (256, 256)).transpose([2, 0, 1])
        mask_img = mask_img[:1, :, :]
        mask_img[mask_img >= 1] = 1
        lo_dict['mask'].append(mask_img)
        if i == length:
            break
    return lo_dict


class Datas(data.Dataset):
    def __init__(self, data_type, lo_dict):
        assert data_type in ['train', 'val', 'test']
        self.data_type = data_type
        self.image = lo_dict['image']
        self.mask = lo_dict['mask']

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        return {'data': self.image[index], 'mask': self.mask[index]}


def make_dataloader(image_dir, mask_dir):
    df = create_dataframe_from_directory(image_dir, mask_dir)
    df_shuffled = df.sample(frac=1, random_state=42)
    train_df = df_shuffled[:int(0.8 * df_shuffled.shape[0])]
    valid_df = df_shuffled[int(0.8 * df_shuffled.shape[0]): int(0.9 * df_shuffled.shape[0])]
    test_df = df_shuffled[int(0.9 * df_shuffled.shape[0]):]
    train_dict = read_image_fu(train_df)
    val_dict = read_image_fu(valid_df)
    test_dict = read_image_fu(test_df)
    train_set = Datas('train', train_dict)
    val_set = Datas('val', val_dict)
    test_set = Datas('test', test_dict)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=False, pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_set,   batch_size=64, shuffle=False, pin_memory=True, drop_last=False)
    test_loader  = DataLoader(test_set,  batch_size=64, shuffle=False, pin_memory=True, drop_last=False)
    return train_loader, val_loader, test_loader


class PetModel(LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, lr, weight_decay=0, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            encoder_weights=None, 
            **kwargs
        )
        self.archs = arch
        self.val_loss = 10000
        self.lr = lr
        self.weight_decay = weight_decay
        self.encoder = encoder_name
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.step_outputs = {"tp": [], "fp": [], "fn": [], "tn": []}
        self.loss_list = {'train': [], 'test': [], 'valid': []}
        self.loss_outputs = {'train': [], 'test': [], 'valid': []}
        self.total_metrics = {'train': {"F1": [], "acc": [], "spec": [], "ioU": [], "pre": []},
                              'test': {"F1": [], "acc": [], "spec": [], "ioU": [], "pre": []},
                              'valid': {"F1": [], "acc": [], "spec": [], "ioU": [], "pre": []}}

    def forward(self, image):
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image = batch['data']
        assert image.ndim == 4
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0
        mask = batch['mask']
        assert mask.ndim == 4
        assert mask.max() <= 1.0 and mask.min() >= 0
        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")
        self.step_outputs['tp'].append(tp)
        self.step_outputs['fp'].append(fp)
        self.step_outputs['fn'].append(fn)
        self.step_outputs['tn'].append(tn)
        self.loss_outputs[stage].append(loss)
        self.log('loss', loss)
        return {"loss": loss, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    def shared_epoch_end(self, stage):
        tp = torch.cat([x for x in self.step_outputs['tp']])
        fp = torch.cat([x for x in self.step_outputs['fp']])
        fn = torch.cat([x for x in self.step_outputs['fn']])
        tn = torch.cat([x for x in self.step_outputs['tn']])
        self.total_metrics[stage]['ioU'].append(
            smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").cpu().detach().numpy()
        )
        self.total_metrics[stage]["F1"].append(
            smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro").cpu().detach().numpy()
        )
        self.total_metrics[stage]["acc"].append(
            smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro").cpu().detach().numpy()
        )
        self.total_metrics[stage]["spec"].append(
            smp.metrics.specificity(tp, fp, fn, tn, reduction="micro").cpu().detach().numpy()
        )
        self.total_metrics[stage]["pre"].append(
            smp.metrics.precision(tp, fp, fn, tn, reduction="micro").cpu().detach().numpy()
        )
        lo_n = torch.stack(self.loss_outputs[stage])
        lo_t = lo_n.mean()
        self.loss_list[stage].append(lo_t.cpu().detach().numpy())
        self.loss_outputs[stage].clear()

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def on_train_epoch_end(self):
        return self.shared_epoch_end("train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def on_validation_epoch_end(self):
        return self.shared_epoch_end("valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def on_test_epoch_end(self):
        return self.shared_epoch_end("test")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-6, verbose=True
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'loss'}


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    train_dataloader, val_dataloader, test_dataloader = make_dataloader(args.image_dir, args.mask_dir)
    model = PetModel(args.arch, "resnet34", in_channels=3, out_classes=1, lr=args.lr)
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=args.epochs,
        enable_checkpointing=False,
        logger=False
    )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model, dataloaders=test_dataloader, verbose=False)
    with open(os.path.join(args.save_dir, f'base_{model.archs}_loss.pkl'), 'wb') as f:
        pickle.dump(model.loss_list, f)
    with open(os.path.join(args.save_dir, f'base_{model.archs}_metrics.pkl'), 'wb') as f:
        pickle.dump(model.total_metrics, f)
    torch.save(model.state_dict(), os.path.join(args.save_dir, f'base_weight_{model.archs}.pth'))
    print(f"\n[OK] Finished arch={args.arch}. Saved to: {args.save_dir}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("arch", type=str, help="Model arch, e.g. Unet / Unetplusplus / DeepLabV3Plus / PAN / Linknet / PSPNet / DeepLabV3 / FPN / MANet")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save_dir", type=str, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--image_dir", type=str, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--mask_dir", type=str, default=DEFAULT_MASK_DIR)
    args = parser.parse_args()
    try:
        main(args)
    except Exception:
        err_path = os.path.join(args.save_dir, f"ERROR_base_{args.arch}.log")
        os.makedirs(args.save_dir, exist_ok=True)
        with open(err_path, "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        print(f"\n[ERROR] arch={args.arch} failed. See log: {err_path}\n")
        raise