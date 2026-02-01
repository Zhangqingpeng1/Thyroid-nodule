import os
import pickle
import torch.optim as optim
import cv2
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from pytorch_lightning import LightningModule, Trainer
from torch.utils import data
from torch.utils.data import *


def create_dataframe_from_directory(image_dir, mask_dir):
    image_paths, mask_paths = [], []
    for image_fold, mask_fold in zip(os.listdir(image_dir), os.listdir(mask_dir)):
        image_paths.append(os.path.join(image_dir, image_fold))
        mask_paths.append(os.path.join(mask_dir, mask_fold))

    return pd.DataFrame({
        'image': image_paths,
        'mask': mask_paths, })


def read_image_fu(lo_df, length=1000000):
    lo_dict = {'image': [], 'mask': []}
    lo_df = lo_df.sample(frac=1)
    for i in lo_df.index:
        df_row = lo_df.loc[i, :]
        lo_dict['image'].append(cv2.resize(cv2.imread(df_row['image']), (256, 256)).transpose([2, 0, 1])) 

        mask_img = cv2.resize(cv2.imread(df_row['mask']), (256, 256)).transpose([2, 0, 1])
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
        return {'data': self.image[index],
                'mask': self.mask[index]}  


def make_dataloader():
    trainval_df = create_dataframe_from_directory(r'/data/zqp/XR/data/picture/trainval-image', r'/data/zqp/XR/data/picture/trainval-mask')
    test_df = create_dataframe_from_directory(r'/data/zqp/XR/data/picture/test-image', r'/data/zqp/XR/data/picture/test-mask')

    df_shuffled = trainval_df.sample(frac=1, random_state=42)
    train_df = df_shuffled[:int(0.9 * df_shuffled.shape[0])]
    valid_df = df_shuffled[int(0.9 * df_shuffled.shape[0]):]

    train_dict = read_image_fu(train_df)  
    val_dict = read_image_fu(valid_df)  
    test_dict = read_image_fu(test_df) 
    train_set = Datas('train', train_dict)
    lo_train_dataloader = DataLoader(train_set, batch_size=64, shuffle=False,
                                     pin_memory=True, drop_last=False) 
    val_set = Datas('val', val_dict)
    lo_val_dataloader = DataLoader(val_set, batch_size=64, shuffle=False,
                                   pin_memory=True, drop_last=False)  
    test_set = Datas('val', test_dict)
    lo_test_dataloader = DataLoader(test_set, batch_size=64, shuffle=False,
                                    pin_memory=True, drop_last=False) 

    return lo_train_dataloader, lo_val_dataloader, lo_test_dataloader


class PetModel(LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, lr, weight_decay=0, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
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

        return {"loss": loss, "tp": tp, "fp": fp, "fn": fn, "tn": tn, }

    def shared_epoch_end(self, stage):
        tp = torch.cat([x for x in self.step_outputs['tp']]) 
        fp = torch.cat([x for x in self.step_outputs['fp']])
        fn = torch.cat([x for x in self.step_outputs['fn']])
        tn = torch.cat([x for x in self.step_outputs['tn']])

        self.total_metrics[stage]['ioU'].append(smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").cpu().detach().numpy())
        self.total_metrics[stage]["F1"].append(smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro").cpu().detach().numpy())
        self.total_metrics[stage]["acc"].append(smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro").cpu().detach().numpy())
        self.total_metrics[stage]["spec"].append(smp.metrics.specificity(tp, fp, fn, tn, reduction="micro").cpu().detach().numpy())
        self.total_metrics[stage]["pre"].append(smp.metrics.precision(tp, fp, fn, tn, reduction="micro").cpu().detach().numpy())
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
        val_loss_cur = torch.stack(self.loss_outputs["valid"])
        val_loss_cur = val_loss_cur.mean()

        return self.shared_epoch_end("valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def on_test_epoch_end(self):
        return self.shared_epoch_end("test")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-6,
                                                         verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'loss'}


if __name__ == '__main__':
    train_dataloader, val_dataloader, test_dataloader = make_dataloader()
    model_0 = PetModel("MANet", "resnet34", in_channels=3, out_classes=1, lr=0.001)
    trainer = Trainer(accelerator="gpu", max_epochs=120, devices=[1])
    trainer.fit(model_0,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader, )
    trainer.test(model_0, dataloaders=test_dataloader, verbose=False)
    with open(f'/data/zqp/XR/TN3K/plus_{model_0.archs}_loss.pkl', 'wb') as file:   
        pickle.dump(model_0.loss_list, file)    
    with open(f'/data/zqp/XR/TN3K/plus_{model_0.archs}_metrics.pkl', 'wb') as file:  
        pickle.dump(model_0.total_metrics, file)
    torch.save(model_0.state_dict(), f'/data/zqp/XR/TN3K/plus_weight_{model_0.archs}.pth') 