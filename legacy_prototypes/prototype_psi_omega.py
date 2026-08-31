#Streamfunction–Vorticity (ψ–ω)
# ============================================================
# STREAMFUNCTION–VORTICITY PINN (GT-AWARE)
# Lid-Driven Cavity Flow (2D, Steady)
# Uses FDM Ground Truth from .pkl
# NOTE:
# At Re = 1000, the ψ–ω formulation is intentionally shown without
# explicit wall-vorticity closure (e.g. Thom's formula).
# This highlights the sensitivity of ψ–ω PINNs at high Reynolds numbers.
# For physically accurate Re=1000 results, see the ψ–p formulation.
# ============================================================

import torch
import torch.nn as nn
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random
import sys
import io

# ---- NumPy pickle compatibility patch (Colab → Kaggle) ----
sys.modules['numpy._core.numeric'] = np._core.numeric

# UNICODE FIX FOR TERMINALS
if hasattr(sys, 'stdout') and hasattr(sys.stdout, 'buffer'):
    try:
        current_encoding = getattr(sys.stdout, 'encoding', None)
        if current_encoding and current_encoding.lower() != 'utf-8':
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass
        
# ------------------------------------------------------------
# REPRODUCIBILITY
# ------------------------------------------------------------
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# For deterministic behavior (slower but reproducible)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ------------------------------------------------------------
# DEVICE
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------------------------------------------
# LOAD GT DATA (FROM YOUR FDM SOLVER)
# ------------------------------------------------------------
with open("gt_data_Re1000.pkl", "rb") as f:
    data_fdm = pickle.load(f)

x_fdm = data_fdm["coordinates"]["x"]
y_fdm = data_fdm["coordinates"]["y"]
psi_fdm = data_fdm["fields"]["psi"]
omega_fdm = data_fdm["fields"]["omega"]

X, Y = np.meshgrid(x_fdm, y_fdm)

# Flatten
x_gt = torch.tensor(X.flatten()[:,None], dtype=torch.float32, device=device, requires_grad=True)
y_gt = torch.tensor(Y.flatten()[:,None], dtype=torch.float32, device=device, requires_grad=True)
psi_gt = torch.tensor(psi_fdm.flatten()[:,None], dtype=torch.float32, device=device)
omega_gt = torch.tensor(omega_fdm.flatten()[:,None], dtype=torch.float32, device=device)

# ------------------------------------------------------------
# NETWORK (ψ, ω)
# ------------------------------------------------------------
class BaseNet(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.net = nn.ModuleList()
        for i in range(len(layers)-1):
            self.net.append(nn.Linear(layers[i], layers[i+1]))
            if i != len(layers)-2:
                # self.net.append(nn.Tanh())
                self.net.append(nn.SiLU())


    def forward(self, x, y):
        x_n = 2.0*x - 1.0
        y_n = 2.0*y - 1.0
        z = torch.cat([x_n, y_n], dim=1)

        for layer in self.net:
            z = layer(z)
        psi_net = z[:,0:1]
        omega_net = z[:,1:2]
        return psi_net, omega_net

class HardBC_PsiOmega(nn.Module):
    def __init__(self, base_net):
        super().__init__()
        self.base_net = base_net

    def forward(self, x, y):
        psi_net, omega_net = self.base_net(x, y)

        # Homogeneous envelope
        D = (x**2) * ((1-x)**2) * (y**2) * ((1-y)**2)

        # Lid-driven particular solution
        u_lid = 16.0 * (x**2) * ((1-x)**2)
        psi_part = u_lid * (y**2) * (y - 1.0)
        psi = D * psi_net + psi_part

        # ω behaves EXACTLY like pressure (gauge only)
        omega = omega_net - torch.mean(omega_net)

        return psi, omega

# ------------------------------------------------------------
# AUTOGRAD HELPERS
# ------------------------------------------------------------
def grad(f, x):
    return torch.autograd.grad(
        f, x, torch.ones_like(f),
        create_graph=True, retain_graph=True
    )[0]

# ------------------------------------------------------------
# PHYSICS RESIDUALS
# ------------------------------------------------------------
def residuals(model, x, y, Re):
    psi, omega = model(x, y)

    psi_x = grad(psi, x)
    psi_y = grad(psi, y)

    u = psi_y
    v = -psi_x

    omega_x = grad(omega, x)
    omega_y = grad(omega, y)

    psi_xx = grad(psi_x, x)
    psi_yy = grad(psi_y, y)

    omega_xx = grad(omega_x, x)
    omega_yy = grad(omega_y, y)

    # Poisson equation
    r_psi = psi_xx + psi_yy + omega

    # Vorticity transport (steady)
    r_omega = u*omega_x + v*omega_y - (1/Re)*(omega_xx + omega_yy)

    return r_psi, r_omega, u, v

# ------------------------------------------------------------
# COLLOCATION POINTS
# ------------------------------------------------------------
Nf = 20000
# uniform interior
x1 = torch.rand(Nf//2, 1, device=device)
y1 = torch.rand(Nf//2, 1, device=device)

# boundary-layer biased
x2 = torch.rand(Nf//4, 1, device=device)
y2 = torch.rand(Nf//4, 1, device=device)**3

x3 = torch.rand(Nf//4, 1, device=device)
y3 = 1.0 - torch.rand(Nf//4, 1, device=device)**3

x4 = torch.rand(Nf//4, 1, device=device)**3
y4 = torch.rand(Nf//4, 1, device=device)

x5 = 1.0 - torch.rand(Nf//4, 1, device=device)**3
y5 = torch.rand(Nf//4, 1, device=device)

x_f = torch.cat([x1,x2,x3,x4,x5],0).requires_grad_(True)
y_f = torch.cat([y1,y2,y3,y4,y5],0).requires_grad_(True)

# ------------------------------------------------------------
# BOUNDARY CONDITIONS (CORRECT LID-DRIVEN CAVITY)
# Domain: [0,1] × [0,1]
# ------------------------------------------------------------
Nb = 2000

# All walls: ψ = 0
x_bc = torch.cat([
    torch.zeros(Nb,1), torch.ones(Nb,1),
    torch.rand(Nb,1), torch.rand(Nb,1)
], dim=0).to(device).requires_grad_()

y_bc = torch.cat([
    torch.rand(Nb,1), torch.rand(Nb,1),
    torch.zeros(Nb,1), torch.ones(Nb,1)
], dim=0).to(device).requires_grad_()

# Lid velocity u = 1 at y = 1
x_lid = torch.rand(Nb,1,device=device,requires_grad=True)
y_lid = torch.ones(Nb,1,device=device,requires_grad=True)
u_lid = torch.ones(Nb,1,device=device)

dx = x_fdm[1] - x_fdm[0]
dy = y_fdm[1] - y_fdm[0]

psi_y_gt, psi_x_gt = np.gradient(psi_fdm, dy, dx)

u_gt = torch.tensor(
    psi_y_gt.flatten()[:,None],
    dtype=torch.float32,
    device=device
)

v_gt = torch.tensor(
    -psi_x_gt.flatten()[:,None],
    dtype=torch.float32,
    device=device
)

# ------------------------------------------------------------
# GT VELOCITY GRADIENTS (FOR CENTERLINE SHARPENING)
# ------------------------------------------------------------
u_gt_np = psi_y_gt
v_gt_np = -psi_x_gt

du_dy_gt_np = np.gradient(u_gt_np, dy, axis=0)
dv_dx_gt_np = np.gradient(v_gt_np, dx, axis=1)

du_dy_gt = torch.tensor(
    du_dy_gt_np.flatten()[:, None],
    dtype=torch.float32,
    device=device
)

dv_dx_gt = torch.tensor(
    dv_dx_gt_np.flatten()[:, None],
    dtype=torch.float32,
    device=device
)

# ------------------------------------------------------------
# TOTAL LOSS (GT-AWARE, HARD-BC)
# ------------------------------------------------------------
def total_loss(model, Re, lam, epoch):
    # -------------------------
    # PDE
    # -------------------------
    r_psi, r_omega, _, _ = residuals(model, x_f, y_f, Re)

    # The Poisson equation is an elliptic smoother — if you include it, you are no longer testing ψ–ω fairly.
    # loss_pde = torch.mean(r_psi**2) + torch.mean(r_omega**2)

    loss_pde = torch.mean(r_omega**2) # only vorticity transport (((Strict fairness)))
    # -------------------------
    # GT SAMPLING (MATCH ψ–p)
    # -------------------------
    N_gt = 3000

    if epoch < 6000:
        idx = torch.randperm(x_gt.shape[0], device=device)[:N_gt]
    else:
        mask_focus = (
            (torch.abs(x_gt - 0.5) < 0.015) |
            (torch.abs(y_gt - 0.5) < 0.015) |
            (y_gt > 0.9)
        ).squeeze()

        idx_focus = torch.where(mask_focus)[0]
        idx_rand  = torch.randperm(x_gt.shape[0], device=device)

        idx = torch.cat([idx_focus[:1500], idx_rand[:1500]])

    xg = x_gt[idx].clone().detach().requires_grad_(True)
    yg = y_gt[idx].clone().detach().requires_grad_(True)

    ug = u_gt[idx]
    vg = v_gt[idx]

    # -------------------------
    # PREDICTION
    # -------------------------
    psi_p, omega_p = model(xg, yg)

    psi_x_p = grad(psi_p, xg)
    psi_y_p = grad(psi_p, yg)

    u_p = psi_y_p
    v_p = -psi_x_p

    loss_gt = torch.mean((u_p - ug)**2 + (v_p - vg)**2)

    # -------------------------
    # CENTERLINE SHARPENING
    # -------------------------
    loss_grad = torch.tensor(0.0, device=device)

    if epoch > 6000:
        mask_u = torch.abs(xg - 0.5) < 0.01
        mask_v = torch.abs(yg - 0.5) < 0.01

        if mask_u.any():
            du_dy_p = grad(u_p, yg)
            loss_grad += torch.mean((du_dy_p[mask_u] - du_dy_gt[idx][mask_u])**2)

        if mask_v.any():
            dv_dx_p = grad(v_p, xg)
            loss_grad += torch.mean((dv_dx_p[mask_v] - dv_dx_gt[idx][mask_v])**2)

    # -------------------------
    # OMEGA GAUGE
    # -------------------------
    loss_omega_anchor = 5e-4 * torch.mean(omega_p**2)

    total = (
        lam["pde"] * loss_pde +
        lam["gt"]  * loss_gt +
        0.2 * loss_grad +
        loss_omega_anchor
    )

    return total, loss_pde, loss_gt

# ------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------
base_net = BaseNet([2, 96, 96, 96, 2]).to(device)
model = HardBC_PsiOmega(base_net).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=6000,
    eta_min=5e-5
)

Re = 1000
lam = {
    "pde": 1.0,
    "gt": 5.0
}

for epoch in range(10000):

    # ---- PHASE 1: Early eddy nucleation ----
    if epoch < 2000:
        lam["pde"] = 0.01
        lam["gt"]  = 10.0

    # ---- PHASE 2: Balanced physics + data ----
    elif epoch < 8000:
        lam["pde"] = 0.05
        lam["gt"]  = 5.0

    # ---- PHASE 3: Physics-dominant refinement ----
    else:
        lam["pde"] = 0.3
        lam["gt"]  = 2.0

    optimizer.zero_grad()
    loss, lpde, lgt = total_loss(model, Re, lam, epoch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()

    if epoch >= 4000:
        scheduler.step()

    if epoch % 1000 == 0:
        print(
            f"Epoch {epoch:6d} | "
            f"Total {loss.item():.2e} | "
            f"PDE {lpde.item():.2e} | "
            f"GT {lgt.item():.2e} | "
            f"LR {optimizer.param_groups[0]['lr']:.2e}"
        )

# ------------------------------------------------------------
# SAVE MODEL
# ------------------------------------------------------------
torch.save(model.state_dict(), "psi_omega_gt_pinn.pth")
print("Model saved: psi_omega_gt_pinn.pth")

# ============================================================
# VISUALIZATION
# ============================================================

model.eval()

# ----------------------------------
# CREATE GRID
# ----------------------------------
Nx = 300
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Nx)
X, Y = np.meshgrid(x, y)

x_t = torch.tensor(
    X.flatten()[:, None],
    dtype=torch.float32,
    device=device,
    requires_grad=True
)
y_t = torch.tensor(
    Y.flatten()[:, None],
    dtype=torch.float32,
    device=device,
    requires_grad=True
)

# Forward pass
psi_pred, omega_pred = model(x_t, y_t)

# ----------------------------------
# VELOCITIES FROM STREAMFUNCTION
# ----------------------------------
psi_x = torch.autograd.grad(
    psi_pred, x_t,
    grad_outputs=torch.ones_like(psi_pred),
    create_graph=True,
    retain_graph=True
)[0]

psi_y = torch.autograd.grad(
    psi_pred, y_t,
    grad_outputs=torch.ones_like(psi_pred),
    create_graph=True,
    retain_graph=True
)[0]

psi = psi_pred.detach().cpu().numpy().reshape(Nx, Nx)

# ---------- Second derivatives ----------
psi_xx = torch.autograd.grad(
    psi_x, x_t,
    grad_outputs=torch.ones_like(psi_x),
    create_graph=True,
    retain_graph=True
)[0]

psi_yy = torch.autograd.grad(
    psi_y, y_t,
    grad_outputs=torch.ones_like(psi_y),
    create_graph=True,
    retain_graph=False
)[0]

# ---------- FAIR vorticity (derived, not learned) ----------
omega = -(psi_xx + psi_yy)

u = psi_y.detach().cpu().numpy().reshape(Nx, Nx)
v = -psi_x.detach().cpu().numpy().reshape(Nx, Nx)
vel_mag = np.sqrt(u**2 + v**2)

# ----------------------------------
# PLOTS
# ----------------------------------
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, axs = plt.subplots(2, 2, figsize=(10,10))

ax_psi    = axs[0,0]
ax_omega  = axs[0,1]
ax_vel    = axs[1,0]
ax_center = axs[1,1]

# -----------------------------
# STREAMLINES
# -----------------------------
ax_psi.contour(X, Y, psi, levels=60, colors="black", linewidths=0.6)
ax_psi.set_title("Streamlines (ψ)")
ax_psi.set_aspect("equal")
ax_psi.set_xlim(0,1); ax_psi.set_ylim(0,1)

# -----------------------------
# VORTICITY
# -----------------------------
omega_np = omega.detach().cpu().numpy().reshape(Nx, Nx)
omega_abs = np.abs(omega_np)
omega_max = np.percentile(omega_abs, 99.5)

im_omega = ax_omega.contourf(
    X, Y, omega_np,
    levels=100,
    cmap="RdBu_r",
    vmin=-omega_max,
    vmax= omega_max,
    extend="both"
)

# -----------------------------
# VELOCITY MAGNITUDE
# -----------------------------
im_vel = ax_vel.contourf(X, Y, vel_mag, levels=40, cmap="viridis")
ax_vel.set_title("|Velocity|")
ax_vel.set_aspect("equal")
ax_vel.set_xlim(0,1); ax_vel.set_ylim(0,1)

div2 = make_axes_locatable(ax_vel)
cax2 = div2.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im_vel, cax=cax2)

# -----------------------------
# CENTERLINE
# -----------------------------
mid = Nx // 2
ax_center.plot(u[:,mid], y, label="u(x=0.5)")
ax_center.plot(x, v[mid,:], label="v(y=0.5)")
ax_center.legend()
ax_center.grid(True, ls="--", alpha=0.4)
ax_center.set_title("Centerline Velocities")
ax_center.set_box_aspect(1)

plt.suptitle(
    f"GT-Aware Streamfunction PINN | Lid-Driven Cavity (Re={Re})",
    fontsize=14
)

plt.tight_layout()
plt.savefig("pinn_lid_cavity_clean.png", dpi=300)
plt.show()
