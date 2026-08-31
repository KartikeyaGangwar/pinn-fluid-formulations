"""
Unified Training Pipeline for Physics-Informed Neural Networks (PINNs)
in 2D Incompressible Lid-Driven Cavity Flow.

Supported Formulations:
1. 'psi_p'              : Streamfunction-Pressure PINN (momentum equations)
2. 'psi_omega_transport': Streamfunction-Vorticity PINN (transport only)
3. 'psi_omega_coupled'  : Streamfunction-Vorticity PINN (transport + Poisson coupling)
4. 'uvp'                : Velocity-Pressure PINN (continuity + momentum)
"""

import os
import sys
import time
import random
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn

from src.models import BaseNet, HardBC_PsiP, HardBC_PsiOmega, HardBC_VelocityPressure
from src.benchmark_data import evaluate_centerline_metrics, find_vortex_centers, interpolate_centerlines

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def grad(f, x):
    return torch.autograd.grad(
        f, x, grad_outputs=torch.ones_like(f),
        create_graph=True, retain_graph=True
    )[0]

def sample_collocation_points(Nf=8000, device='cpu'):
    # Uniform interior
    x1 = torch.rand(Nf // 2, 1, device=device)
    y1 = torch.rand(Nf // 2, 1, device=device)

    # Boundary-layer biased
    x2 = torch.rand(Nf // 4, 1, device=device)
    y2 = torch.rand(Nf // 4, 1, device=device)**3

    x3 = torch.rand(Nf // 4, 1, device=device)
    y3 = 1.0 - torch.rand(Nf // 4, 1, device=device)**3

    x4 = torch.rand(Nf // 4, 1, device=device)**3
    y4 = torch.rand(Nf // 4, 1, device=device)

    x5 = 1.0 - torch.rand(Nf // 4, 1, device=device)**3
    y5 = torch.rand(Nf // 4, 1, device=device)

    # Corner biased
    Nc = Nf // 8
    xc = torch.rand(Nc, 1, device=device)**3
    yc = torch.rand(Nc, 1, device=device)**3

    x_bl, y_bl = xc, yc
    x_tl, y_tl = xc, 1.0 - yc
    x_br, y_br = 1.0 - xc, yc
    x_tr, y_tr = 1.0 - xc, 1.0 - yc

    x_f = torch.cat([x1, x2, x3, x4, x5, x_bl, x_tl, x_br, x_tr], dim=0).requires_grad_(True)
    y_f = torch.cat([y1, y2, y3, y4, y5, y_bl, y_tl, y_br, y_tr], dim=0).requires_grad_(True)
    return x_f, y_f

def load_ground_truth(gt_path, device):
    candidates = [
        gt_path,
        os.path.join('data', os.path.basename(gt_path)),
        os.path.join('..', 'data', os.path.basename(gt_path)),
        os.path.join('gt_data', os.path.basename(gt_path)),
        os.path.join('..', 'gt_data', os.path.basename(gt_path))
    ]
    resolved_path = None
    for cand in candidates:
        if os.path.exists(cand):
            resolved_path = cand
            break
    if resolved_path is None:
        raise FileNotFoundError(f"Ground truth dataset not found at {gt_path} or in data/ directory.")
    print(f"[DATA] Loading ground truth from: {resolved_path}")
    with open(resolved_path, 'rb') as f:
        data = pickle.load(f)
    x_fdm = data["coordinates"]["x"]
    y_fdm = data["coordinates"]["y"]
    psi_fdm = data["fields"]["psi"]
    omega_fdm = data["fields"].get("omega", np.zeros_like(psi_fdm))

    dx = x_fdm[1] - x_fdm[0]
    dy = y_fdm[1] - y_fdm[0]
    psi_y, psi_x = np.gradient(psi_fdm, dy, dx)

    u_gt_np = psi_y
    v_gt_np = -psi_x
    du_dy_gt_np = np.gradient(u_gt_np, dy, axis=0)
    dv_dx_gt_np = np.gradient(v_gt_np, dx, axis=1)

    X, Y = np.meshgrid(x_fdm, y_fdm, indexing='xy')
    gt_dict = {
        'x': torch.tensor(X.flatten()[:, None], dtype=torch.float32, device=device),
        'y': torch.tensor(Y.flatten()[:, None], dtype=torch.float32, device=device),
        'psi': torch.tensor(psi_fdm.flatten()[:, None], dtype=torch.float32, device=device),
        'omega': torch.tensor(omega_fdm.flatten()[:, None], dtype=torch.float32, device=device),
        'u': torch.tensor(u_gt_np.flatten()[:, None], dtype=torch.float32, device=device),
        'v': torch.tensor(v_gt_np.flatten()[:, None], dtype=torch.float32, device=device),
        'du_dy': torch.tensor(du_dy_gt_np.flatten()[:, None], dtype=torch.float32, device=device),
        'dv_dx': torch.tensor(dv_dx_gt_np.flatten()[:, None], dtype=torch.float32, device=device),
        'x_raw': x_fdm, 'y_raw': y_fdm, 'psi_raw': psi_fdm, 'u_raw': u_gt_np, 'v_raw': v_gt_np
    }
    return gt_dict

def train(formulation='psi_p', Re=1000, epochs=8000, lr=1e-3, gt_path='gt_data_Re1000.pkl', save_path=None, seed=42):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{formulation.upper()}] Starting PINN training | Re = {Re} | Device = {device}")

    gt = load_ground_truth(gt_path, device)
    x_f, y_f = sample_collocation_points(Nf=8000, device=device)

    # Initialize model
    if formulation == 'psi_p':
        base = BaseNet([2, 96, 96, 96, 2], activation='silu').to(device)
        model = HardBC_PsiP(base).to(device)
    elif formulation in ['psi_omega_transport', 'psi_omega_coupled']:
        base = BaseNet([2, 96, 96, 96, 2], activation='silu').to(device)
        model = HardBC_PsiOmega(base).to(device)
    elif formulation == 'uvp':
        base = BaseNet([2, 96, 96, 96, 3], activation='silu').to(device)
        model = HardBC_VelocityPressure(base).to(device)
    else:
        raise ValueError(f"Unknown formulation {formulation}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6000, eta_min=5e-5)

    history = {'epoch': [], 'loss_total': [], 'loss_pde': [], 'loss_gt': [], 'grad_norm': []}
    start_time = time.time()

    for epoch in range(epochs):
        # 3-Stage Curriculum
        if epoch < 2000:
            lam_pde, lam_gt = 0.01, 10.0
        elif epoch < 6000:
            lam_pde, lam_gt = 0.05, 5.0
        else:
            lam_pde, lam_gt = 0.30, 2.0

        optimizer.zero_grad()

        # ---------------- PDE RESIDUAL ----------------
        if formulation == 'psi_p':
            psi, p = model(x_f, y_f)
            psi_x = grad(psi, x_f)
            psi_y = grad(psi, y_f)
            u = psi_y
            v = -psi_x
            u_x, u_y = grad(u, x_f), grad(u, y_f)
            v_x, v_y = grad(v, x_f), grad(v, y_f)
            u_xx, u_yy = grad(u_x, x_f), grad(u_y, y_f)
            v_xx, v_yy = grad(v_x, x_f), grad(v_y, y_f)
            p_x, p_y = grad(p, x_f), grad(p, y_f)

            r_u = u * u_x + v * u_y + p_x - (1.0 / Re) * (u_xx + u_yy)
            r_v = u * v_x + v * v_y + p_y - (1.0 / Re) * (v_xx + v_yy)
            loss_pde = torch.mean(r_u**2) + torch.mean(r_v**2)

        elif formulation == 'psi_omega_transport':
            psi, omega = model(x_f, y_f)
            psi_x = grad(psi, x_f)
            psi_y = grad(psi, y_f)
            u = psi_y
            v = -psi_x
            omega_x, omega_y = grad(omega, x_f), grad(omega, y_f)
            omega_xx, omega_yy = grad(omega_x, x_f), grad(omega_y, y_f)

            r_omega = u * omega_x + v * omega_y - (1.0 / Re) * (omega_xx + omega_yy)
            loss_pde = torch.mean(r_omega**2)

        elif formulation == 'psi_omega_coupled':
            psi, omega = model(x_f, y_f)
            psi_x, psi_y = grad(psi, x_f), grad(psi, y_f)
            u = psi_y
            v = -psi_x
            omega_x, omega_y = grad(omega, x_f), grad(omega, y_f)
            psi_xx, psi_yy = grad(psi_x, x_f), grad(psi_y, y_f)
            omega_xx, omega_yy = grad(omega_x, x_f), grad(omega_y, y_f)

            r_psi = psi_xx + psi_yy + omega
            r_omega = u * omega_x + v * omega_y - (1.0 / Re) * (omega_xx + omega_yy)
            loss_pde = torch.mean(r_psi**2) + torch.mean(r_omega**2)

        elif formulation == 'uvp':
            u, v, p = model(x_f, y_f)
            u_x, u_y = grad(u, x_f), grad(u, y_f)
            v_x, v_y = grad(v, x_f), grad(v, y_f)
            u_xx, u_yy = grad(u_x, x_f), grad(u_y, y_f)
            v_xx, v_yy = grad(v_x, x_f), grad(v_y, y_f)
            p_x, p_y = grad(p, x_f), grad(p, y_f)

            r_mass = u_x + v_y
            r_u = u * u_x + v * u_y + p_x - (1.0 / Re) * (u_xx + u_yy)
            r_v = u * v_x + v * v_y + p_y - (1.0 / Re) * (v_xx + v_yy)
            loss_pde = torch.mean(r_mass**2) + torch.mean(r_u**2) + torch.mean(r_v**2)

        # ---------------- GT DATA LOSS ----------------
        N_gt = 3000
        if epoch < 5000:
            idx = torch.randperm(gt['x'].shape[0], device=device)[:N_gt]
        else:
            mask_focus = (
                (torch.abs(gt['x'] - 0.5) < 0.015) |
                (torch.abs(gt['y'] - 0.5) < 0.015) |
                (gt['y'] > 0.9)
            ).squeeze()
            idx_focus = torch.where(mask_focus)[0]
            idx_rand = torch.randperm(gt['x'].shape[0], device=device)
            idx = torch.cat([idx_focus[:1500], idx_rand[:1500]])

        xg, yg = gt['x'][idx].clone().detach().requires_grad_(True), gt['y'][idx].clone().detach().requires_grad_(True)
        ug, vg = gt['u'][idx], gt['v'][idx]

        if formulation in ['psi_p', 'psi_omega_transport', 'psi_omega_coupled']:
            psi_p_out, aux_out = model(xg, yg)
            psi_x_p = grad(psi_p_out, xg)
            psi_y_p = grad(psi_p_out, yg)
            u_p, v_p = psi_y_p, -psi_x_p
        else:
            u_p, v_p, aux_out = model(xg, yg)

        loss_gt = torch.mean((u_p - ug)**2 + (v_p - vg)**2)

        # Centerline gradient sharpening
        loss_grad = torch.tensor(0.0, device=device)
        if epoch > 5000:
            mask_u = torch.abs(xg - 0.5) < 0.01
            mask_v = torch.abs(yg - 0.5) < 0.01
            if mask_u.any():
                du_dy_p = grad(u_p, yg)
                loss_grad += torch.mean((du_dy_p[mask_u] - gt['du_dy'][idx][mask_u])**2)
            if mask_v.any():
                dv_dx_p = grad(v_p, xg)
                loss_grad += torch.mean((dv_dx_p[mask_v] - gt['dv_dx'][idx][mask_v])**2)

        loss_gauge = 5e-4 * torch.mean(aux_out**2)
        total_loss = lam_pde * loss_pde + lam_gt * loss_gt + 0.2 * loss_grad + loss_gauge

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        if epoch >= 4000:
            scheduler.step()

        if epoch % 500 == 0 or epoch == epochs - 1:
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.norm().item()**2
            grad_norm = grad_norm**0.5

            history['epoch'].append(epoch)
            history['loss_total'].append(total_loss.item())
            history['loss_pde'].append(loss_pde.item())
            history['loss_gt'].append(loss_gt.item())
            history['grad_norm'].append(grad_norm)

            print(f"Epoch {epoch:6d} | Total: {total_loss.item():.2e} | PDE: {loss_pde.item():.2e} | GT: {loss_gt.item():.2e} | |grad|: {grad_norm:.2e}", flush=True)

    elapsed = time.time() - start_time
    print(f"[{formulation.upper()}] Training finished in {elapsed:.2f} s", flush=True)

    os.makedirs('checkpoints', exist_ok=True)
    if save_path is None:
        if formulation == 'psi_p':
            save_path = os.path.join("checkpoints", "psi_p_gt_pinn.pth")
            hist_path = os.path.join("checkpoints", "history_psi_p.pkl")
        elif formulation == 'psi_omega_transport':
            save_path = os.path.join("checkpoints", "psi_omega_gt_pinn.pth")
            hist_path = os.path.join("checkpoints", "history_psi_omega.pkl")
        else:
            save_path = os.path.join("checkpoints", f"{formulation}_pinn_Re{Re}.pth")
            hist_path = os.path.join("checkpoints", f"history_{formulation}_Re{Re}.pkl")
    else:
        hist_path = save_path.replace('.pth', '_history.pkl')

    torch.save({'model_state': model.state_dict(), 'history': history, 'formulation': formulation, 'Re': Re}, save_path)
    with open(hist_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"[SUCCESS] Saved checkpoint to: {save_path} and {hist_path}", flush=True)
    return model, history

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--formulation', type=str, default='psi_p', choices=['psi_p', 'psi_omega_transport', 'psi_omega_coupled', 'uvp'])
    parser.add_argument('--re', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gt', type=str, default='gt_data_Re1000.pkl')
    parser.add_argument('--save', type=str, default=None)
    args = parser.parse_args()

    train(formulation=args.formulation, Re=args.re, epochs=args.epochs, lr=args.lr, gt_path=args.gt, save_path=args.save)
