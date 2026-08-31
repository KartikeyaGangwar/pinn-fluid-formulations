import os
import sys
import io
import time
import random
import pickle
import numpy as np
import torch
import torch.nn as nn

# UNICODE & STDOUT FIX
if hasattr(sys, 'stdout') and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.models import BaseNet, HardBC_PsiP, HardBC_PsiOmega
from src.train_pinn import sample_collocation_points, load_ground_truth, set_seed, grad

def train_psi_p(Re=1000, epochs=8000, lr=1e-3, device='cuda'):
    set_seed(42)
    print(f"\n=======================================================", flush=True)
    print(f"[*] TRAINING STREAMFUNCTION-PRESSURE (psi-p) PINN | Re={Re}", flush=True)
    print(f"=======================================================", flush=True)

    gt = load_ground_truth('data/gt_data_Re1000.pkl', device)
    x_f, y_f = sample_collocation_points(Nf=8000, device=device)

    base = BaseNet([2, 96, 96, 96, 2], activation='silu').to(device)
    model = HardBC_PsiP(base).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6000, eta_min=5e-5)

    history = {'epoch': [], 'loss_total': [], 'loss_pde': [], 'loss_gt': [], 'grad_norm': []}
    start_time = time.time()

    for epoch in range(epochs):
        if epoch < 2000:
            lam_pde, lam_gt = 0.01, 10.0
        elif epoch < 6000:
            lam_pde, lam_gt = 0.05, 5.0
        else:
            lam_pde, lam_gt = 0.30, 2.0

        optimizer.zero_grad()

        # PDE Momentum Residuals
        psi, p = model(x_f, y_f)
        psi_x = grad(psi, x_f)
        psi_y = grad(psi, y_f)
        u, v = psi_y, -psi_x
        u_x, u_y = grad(u, x_f), grad(u, y_f)
        v_x, v_y = grad(v, x_f), grad(v, y_f)
        u_xx, u_yy = grad(u_x, x_f), grad(u_y, y_f)
        v_xx, v_yy = grad(v_x, x_f), grad(v_y, y_f)
        p_x, p_y = grad(p, x_f), grad(p, y_f)

        r_u = u * u_x + v * u_y + p_x - (1.0 / Re) * (u_xx + u_yy)
        r_v = u * v_x + v * v_y + p_y - (1.0 / Re) * (v_xx + v_yy)
        loss_pde = torch.mean(r_u**2) + torch.mean(r_v**2)

        # GT Data Loss
        N_gt = 3000
        if epoch < 5000:
            idx = torch.randperm(gt['x'].shape[0], device=device)[:N_gt]
        else:
            mask_focus = ((torch.abs(gt['x'] - 0.5) < 0.015) | (torch.abs(gt['y'] - 0.5) < 0.015) | (gt['y'] > 0.9)).squeeze()
            idx_focus = torch.where(mask_focus)[0]
            idx_rand = torch.randperm(gt['x'].shape[0], device=device)
            idx = torch.cat([idx_focus[:1500], idx_rand[:1500]])

        xg, yg = gt['x'][idx].clone().detach().requires_grad_(True), gt['y'][idx].clone().detach().requires_grad_(True)
        ug, vg = gt['u'][idx], gt['v'][idx]

        psi_p_out, aux_out = model(xg, yg)
        psi_x_p = grad(psi_p_out, xg)
        psi_y_p = grad(psi_p_out, yg)
        u_p, v_p = psi_y_p, -psi_x_p

        loss_gt = torch.mean((u_p - ug)**2 + (v_p - vg)**2)

        # Centerline sharpening
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
            grad_norm = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None)**0.5
            history['epoch'].append(epoch)
            history['loss_total'].append(total_loss.item())
            history['loss_pde'].append(loss_pde.item())
            history['loss_gt'].append(loss_gt.item())
            history['grad_norm'].append(grad_norm)
            print(f"[psi-p] Epoch {epoch:5d}/{epochs} | Total: {total_loss.item():.2e} | PDE: {loss_pde.item():.2e} | GT: {loss_gt.item():.2e} | |grad|: {grad_norm:.2e}", flush=True)

    elapsed = time.time() - start_time
    print(f"[SUCCESS] psi-p Training completed in {elapsed:.1f}s", flush=True)
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("checkpoints", "psi_p_gt_pinn.pth"))
    with open(os.path.join("checkpoints", "history_psi_p.pkl"), "wb") as f:
        pickle.dump(history, f)
    print(f"[SAVED] checkpoints/psi_p_gt_pinn.pth and checkpoints/history_psi_p.pkl", flush=True)
    return model, history


def train_psi_omega(Re=1000, epochs=8000, lr=1e-3, device='cuda'):
    set_seed(42)
    print(f"\n=======================================================", flush=True)
    print(f"[*] TRAINING STREAMFUNCTION-VORTICITY (psi-omega) PINN | Re={Re}", flush=True)
    print(f"=======================================================", flush=True)

    gt = load_ground_truth('data/gt_data_Re1000.pkl', device)
    x_f, y_f = sample_collocation_points(Nf=8000, device=device)

    base = BaseNet([2, 96, 96, 96, 2], activation='silu').to(device)
    model = HardBC_PsiOmega(base).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6000, eta_min=5e-5)

    history = {'epoch': [], 'loss_total': [], 'loss_pde': [], 'loss_gt': [], 'grad_norm': []}
    start_time = time.time()

    for epoch in range(epochs):
        if epoch < 2000:
            lam_pde, lam_gt = 0.01, 10.0
        elif epoch < 6000:
            lam_pde, lam_gt = 0.05, 5.0
        else:
            lam_pde, lam_gt = 0.30, 2.0

        optimizer.zero_grad()

        # Vorticity Transport Residual
        psi, omega = model(x_f, y_f)
        psi_x = grad(psi, x_f)
        psi_y = grad(psi, y_f)
        u, v = psi_y, -psi_x
        omega_x = grad(omega, x_f)
        omega_y = grad(omega, y_f)
        omega_xx = grad(omega_x, x_f)
        omega_yy = grad(omega_y, y_f)

        r_omega = u * omega_x + v * omega_y - (1.0 / Re) * (omega_xx + omega_yy)
        loss_pde = torch.mean(r_omega**2)

        # GT Data Loss
        N_gt = 3000
        if epoch < 5000:
            idx = torch.randperm(gt['x'].shape[0], device=device)[:N_gt]
        else:
            mask_focus = ((torch.abs(gt['x'] - 0.5) < 0.015) | (torch.abs(gt['y'] - 0.5) < 0.015) | (gt['y'] > 0.9)).squeeze()
            idx_focus = torch.where(mask_focus)[0]
            idx_rand = torch.randperm(gt['x'].shape[0], device=device)
            idx = torch.cat([idx_focus[:1500], idx_rand[:1500]])

        xg, yg = gt['x'][idx].clone().detach().requires_grad_(True), gt['y'][idx].clone().detach().requires_grad_(True)
        ug, vg = gt['u'][idx], gt['v'][idx]

        psi_p_out, omega_out = model(xg, yg)
        psi_x_p = grad(psi_p_out, xg)
        psi_y_p = grad(psi_p_out, yg)
        u_p, v_p = psi_y_p, -psi_x_p

        loss_gt = torch.mean((u_p - ug)**2 + (v_p - vg)**2)

        # Centerline sharpening
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

        loss_gauge = 5e-4 * torch.mean(omega_out**2)
        total_loss = lam_pde * loss_pde + lam_gt * loss_gt + 0.2 * loss_grad + loss_gauge

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        if epoch >= 4000:
            scheduler.step()

        if epoch % 500 == 0 or epoch == epochs - 1:
            grad_norm = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None)**0.5
            history['epoch'].append(epoch)
            history['loss_total'].append(total_loss.item())
            history['loss_pde'].append(loss_pde.item())
            history['loss_gt'].append(loss_gt.item())
            history['grad_norm'].append(grad_norm)
            print(f"[psi-omega] Epoch {epoch:5d}/{epochs} | Total: {total_loss.item():.2e} | PDE: {loss_pde.item():.2e} | GT: {loss_gt.item():.2e} | |grad|: {grad_norm:.2e}", flush=True)

    elapsed = time.time() - start_time
    print(f"[SUCCESS] psi-omega Training completed in {elapsed:.1f}s", flush=True)
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("checkpoints", "psi_omega_gt_pinn.pth"))
    with open(os.path.join("checkpoints", "history_psi_omega.pkl"), "wb") as f:
        pickle.dump(history, f)
    print(f"[SAVED] checkpoints/psi_omega_gt_pinn.pth and checkpoints/history_psi_omega.pkl", flush=True)
    return model, history


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Executing Complete PINN Training Suite on {device}...")
    train_psi_p(Re=1000, epochs=8000, lr=1e-3, device=device)
    train_psi_omega(Re=1000, epochs=8000, lr=1e-3, device=device)
    print("\n=======================================================")
    print("[ALL TRAININGS COMPLETE] Both models trained & saved!")
    print("=======================================================\n")
