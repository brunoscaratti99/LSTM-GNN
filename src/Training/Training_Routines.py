import torch
from Data_Library import safe_r2, safe_mape, assert_finite, save_error_plots
import torch.nn as nn
import torch
import copy
import time 
import json
from pathlib import Path
import sys

sys.path.append("../src")



from Evaluation.metrics import safe_r2, safe_mape
from Evaluation.comparison_plots import save_error_plots
from Data.preprocessing import assert_finite

loss_fn = nn.MSELoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def eval_with_loader(model, loader, use_amp, amp_device, amp_dtype):
    model.eval()
    total_loss, total_mse, total_mae, total_mape, total_r2 = 0.0, 0.0, 0.0, 0.0, 0.0
    steps = 0
    mse_fn = nn.MSELoss()
    mae_fn = nn.L1Loss()

    with torch.no_grad():
        for batch_idx, (xb, yb) in enumerate(loader):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            assert_finite(f"eval_xb_batch{batch_idx}", xb)
            assert_finite(f"eval_yb_batch{batch_idx}", yb)

            with torch.autocast(device_type=amp_device, dtype=amp_dtype, enabled=use_amp):
                pred = model(xb)
                assert_finite(f"eval_pred_batch{batch_idx}", pred)
                loss = loss_fn(pred, yb)

            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite eval loss on batch={batch_idx}")

            total_loss += loss.item()
            total_mse += mse_fn(pred, yb).item()
            total_mae += mae_fn(pred, yb).item()
            total_mape += safe_mape(yb, pred, eps=1e-3).item()
            total_r2 += safe_r2(yb.reshape(-1), pred.reshape(-1)).item()
            steps += 1

    return {
        "loss": total_loss / max(steps, 1),
        "mse": total_mse / max(steps, 1),
        "mae": total_mae / max(steps, 1),
        "mape": total_mape / max(steps, 1),
        "r2": total_r2 / max(steps, 1),
    }


def train_batched_only(model, train_loader, val_loader, train_period, hidden_dim, epochs, lr, weight_decay, patience, run_dir=None):
    model = copy.deepcopy(model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse_fn = nn.MSELoss()
    mae_fn = nn.L1Loss()

    # Estável para debug: AMP desligado
    use_amp = False
    amp_device = "cuda"
    amp_dtype = torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"train_batched_only: use_amp={use_amp}")

    best_model = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    counter = 0

    history = {
        "train_loss": [], "train_mse": [], "train_mae": [], "train_mape": [], "train_r2": [],
        "val_loss": [], "val_mse": [], "val_mae": [], "val_mape": [], "val_r2": [], "epoch_time": []
    }

    for ep in range(epochs):
        t0 = time.perf_counter()
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        train_mae = 0.0
        train_mape = 0.0
        train_r2 = 0.0
        steps = 0

        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            assert_finite(f"train_xb_epoch{ep}_batch{batch_idx}", xb)
            assert_finite(f"train_yb_epoch{ep}_batch{batch_idx}", yb)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=amp_device, dtype=amp_dtype, enabled=use_amp):
                pred = model(xb)
                assert_finite(f"train_pred_epoch{ep}_batch{batch_idx}", pred)
                loss = loss_fn(pred, yb)

            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss epoch={ep} batch={batch_idx}")

            scaler.scale(loss).backward()
            
            

            for name, param in model.named_parameters():
                #if param.grad is not None:
                    #print(name, param.grad.abs().mean())
                if param.grad is not None and (not torch.isfinite(param.grad).all()):
                    raise RuntimeError(f"non-finite gradient in {name} epoch={ep} batch={batch_idx}")

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.8)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_mse += mse_fn(pred, yb).item()
            train_mae += mae_fn(pred, yb).item()
            train_mape += safe_mape(yb, pred, eps=1e-3).item()
            train_r2 += safe_r2(yb.reshape(-1), pred.reshape(-1)).item()
            steps += 1

        history["train_loss"].append(train_loss / max(steps, 1))
        history["train_mse"].append(train_mse / max(steps, 1))
        history["train_mae"].append(train_mae / max(steps, 1))
        history["train_mape"].append(train_mape / max(steps, 1))
        history["train_r2"].append(train_r2 / max(steps, 1))

        val_metrics = eval_with_loader(model, val_loader, use_amp, amp_device, amp_dtype)
        history["val_loss"].append(val_metrics["loss"])
        history["val_mse"].append(val_metrics["mse"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_mape"].append(val_metrics["mape"])
        history["val_r2"].append(val_metrics["r2"])

        history["epoch_time"].append(time.perf_counter() - t0)

        print(
            f"epoch={ep+1}/{epochs}" 
            f"train_loss={history['train_loss'][-1]:e}/"
            f"val_loss={history['val_loss'][-1]:e}/"
            f"train_mae={history['train_mae'][-1]:e}/"
            f"val_mae={history['val_mae'][-1]:e}/" 
            f"val_mape={history['val_mape'][-1]:e}/"
            f"val_r2={history['val_r2'][-1]:e}"
        )

        if history["val_loss"][-1] < best_val_loss:
            best_val_loss = history["val_loss"][-1]
            counter = 0
            best_model = copy.deepcopy(model.state_dict())
        else:
            counter += 1

        if counter > patience:
            model.load_state_dict(best_model)
            print("Early Stopping")
            break

    model.load_state_dict(best_model)

    summary = {
        "hiddem_dim"        : hidden_dim,
        'train_period'      : train_period,
        "device"            : device,
        "epochs_requested"  : epochs,
        "epochs_ran"        : len(history["epoch_time"]),
        "lr"                : lr,
        "weight_decay"      : weight_decay,
        "patience"          : patience,
        "batch_size"        : train_loader.batch_size,
        "best_val_loss"     : best_val_loss,
        "last_val_mse"      : history["val_mse"][-1] if history["val_mse"] else None,
        "last_val_mae"      : history["val_mae"][-1] if history["val_mae"] else None,
        "last_val_mape"     : history["val_mape"][-1] if history["val_mape"] else None,
        "last_val_r2"       : history["val_r2"][-1] if history["val_r2"] else None,
        "total_time_s"      : float(sum(history["epoch_time"])),
    }

    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        save_error_plots(
            str(run_dir),
            history["train_mse"], history["val_mse"],
            history["train_mae"], history["val_mae"],
            history["train_r2"],  history["val_r2"],
        )

        torch.save(history, run_dir / "hist.pt")
        torch.save(model.state_dict(), run_dir / "model_state_dict.pt")
        with open(run_dir / "run_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return model, history, summary
