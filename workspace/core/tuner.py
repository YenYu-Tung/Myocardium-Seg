import os
import copy

from monai.utils import set_determinism
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from ray import tune
import math
try:
    from ray.air import session as air_session
except Exception:  # pragma: no cover
    air_session = None
# from ray import train

from tqdm import tqdm

# set deterministic training for reproducibility
set_determinism(seed=0)


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.ema_model.state_dict().items():
                v.copy_(v * self.decay + msd[k] * (1.0 - self.decay))

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        self.ema_model.load_state_dict(state_dict)

    def copy_from(self, model):
        self.ema_model.load_state_dict(model.state_dict())

    def get_model(self):
        return self.ema_model

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def val_epoch(loader, model, model_inferer, acc_func, post_label, post_pred, global_step, args):
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(loader):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            val_outputs = model_inferer(val_inputs)

            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            acc_func(y_pred=val_output_convert, y=val_labels_convert)
            loader.set_description(
                "Validate (%d / %d Steps)" % (global_step, 10.0)
            )
        mean_dice_val = acc_func.aggregate().item()
        acc_func.reset()
    return mean_dice_val


def train_epoch(loader, model, optimizer, loss_func, writer, global_step, epoch, args, scaler=None, ema_handler=None, use_amp=False, grad_clip_norm=0.0):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(
        loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].to(args.device), batch["label"].to(args.device))
        
        with autocast(enabled=use_amp):
            if args.deep_sup:
                logit_maps = model(x)
                y1 = F.interpolate(y, scale_factor=(1 / args.patch_size))
                y2 = F.interpolate(y, scale_factor=(1 / (args.patch_size * 2)))
                loss0 = loss_func(logit_maps[0], y)
                loss1 = loss_func(logit_maps[1], y1)
                loss2 = loss_func(logit_maps[2], y2)
                loss = 0.6 * loss0 + 0.25 * loss1 + 0.15 * loss2
            else:
                logit_map = model(x)
                loss = loss_func(logit_map, y)
        
        if scaler and use_amp:
            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm > 0:
                clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        epoch_loss += loss.item()
        optimizer.zero_grad()
        if ema_handler is not None:
            ema_handler.update(model)
        epoch_iterator.set_description(
            "[Epoch %d] Training (%d Steps) (loss=%2.5f)"
            % (epoch, global_step, loss)
        )

        writer.add_scalar("lr", get_lr(optimizer), global_step=global_step)
        writer.add_scalar("tr_loss", loss, global_step=global_step)
        global_step += 1
    return global_step


def save_checkpoint(filename, model, epoch, best_acc, early_stop_count, args, optimizer=None, scheduler=None, ema_state_dict=None):
    state_dict = model.state_dict()
    save_dict = {
        "epoch": epoch,
        "best_acc": best_acc,
        "early_stop_count": early_stop_count,
        "state_dict": state_dict,
    }
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    if ema_state_dict is not None:
        save_dict["ema_state_dict"] = ema_state_dict
    filename = os.path.join(args.model_dir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
        start_epoch,
        best_acc,
        early_stop_count,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_func,
        acc_func,
        model_inferer,
        post_label,
        post_pred,
        writer,
        args,
        scaler=None,
        ema_handler=None,
        use_amp=False,
        grad_clip_norm=0.0,
):
    global_step = start_epoch * len(train_loader)
    val_acc_best = best_acc

    for epoch in range(start_epoch, args.max_epoch + 1):
        if early_stop_count == args.max_early_stop_count:
            break

        global_step = train_epoch(
            train_loader,
            model,
            optimizer,
            loss_func,
            writer,
            global_step,
            epoch,
            args,
            scaler=scaler,
            ema_handler=ema_handler,
            use_amp=use_amp,
            grad_clip_norm=grad_clip_norm,
        )

        if (
                epoch % args.val_every == 0 and epoch != 0
        ) or epoch == args.max_epoch:
            epoch_iterator_val = tqdm(
                val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
            )
            eval_model = ema_handler.get_model() if ema_handler is not None else model
            val_avg_acc = val_epoch(
                epoch_iterator_val,
                eval_model,
                model_inferer,
                acc_func,
                post_label,
                post_pred,
                global_step,
                args
            )

            writer.add_scalar("val_dice", val_avg_acc, global_step=global_step)

            if val_avg_acc > val_acc_best:
                val_acc_best = val_avg_acc
                early_stop_count = 0
                save_checkpoint(
                    'best_model.pth',
                    eval_model,
                    epoch,
                    val_acc_best,
                    early_stop_count,
                    args,
                    optimizer,
                    scheduler,
                    ema_state_dict=ema_handler.state_dict() if ema_handler is not None else None
                )
                print(
                    "Best Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        val_acc_best, val_avg_acc
                    )
                )
            else:
                early_stop_count += 1
                print("Early stop count: ", early_stop_count)
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        val_acc_best, val_avg_acc
                    )
                )

            # Send the current training result back to Tune
            metrics = {
                "val_bst_acc": val_acc_best,
                "esc": early_stop_count,
                "inf_time": float("nan"),
            }
            if air_session is not None:
                air_session.report(metrics)
            else:
                tune.report(**metrics)
            

        if scheduler is not None:
            scheduler.step()
