# Streamfunction–Pressure (ψ–p)
# Recommended formulation for high-Re lid-driven cavity flow
# ============================================================
# STREAMFUNCTION–PRESSURE PINN (GT-AWARE)
# Recommended formulation for high-Re lid-driven cavity flow
# Lid-Driven Cavity Flow (2D, Steady)
# Uses FDM Ground Truth from .pkl
# NOTE:
# Pressure is learned up to an arbitrary constant (gauge freedom).
# Only pressure gradients enter the momentum equations.
# Vorticity is derived from the streamfunction for visualization only.
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
sys.modules['numpy._core.numeric'] = np.core.numeric

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

# omega_fdm is loaded for reference only (not used in ψ–p training)
omega_fdm = data_fdm["fields"]["omega"]

# ------------------------------------------------------------
# PRECOMPUTE GT VELOCITIES
# ------------------------------------------------------------
dx = x_fdm[1] - x_fdm[0]
dy = y_fdm[1] - y_fdm[0]

psi_y_gt_np, psi_x_gt_np = np.gradient(psi_fdm, dy, dx)

u_gt = torch.tensor(
    psi_y_gt_np.flatten()[:, None],
    dtype=torch.float32,
    device=device
)

v_gt = torch.tensor(
    -psi_x_gt_np.flatten()[:, None],
    dtype=torch.float32,
    device=device
)

# ------------------------------------------------------------
# GT VELOCITY GRADIENTS (FOR CENTERLINE SHARPENING)
# ------------------------------------------------------------
u_gt_np = psi_y_gt_np
v_gt_np = -psi_x_gt_np

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

X, Y = np.meshgrid(x_fdm, y_fdm)

# Flatten
x_gt = torch.tensor(X.flatten()[:,None], dtype=torch.float32, device=device)
y_gt = torch.tensor(Y.flatten()[:,None], dtype=torch.float32, device=device)
psi_gt = torch.tensor(psi_fdm.flatten()[:,None], dtype=torch.float32, device=device)
omega_gt = torch.tensor(omega_fdm.flatten()[:,None], dtype=torch.float32, device=device)

# ------------------------------------------------------------
# NETWORK (ψ, p)
# ------------------------------------------------------------
class BaseNet(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.net = nn.ModuleList()
        for i in range(len(layers)-1):
            self.net.append(nn.Linear(layers[i], layers[i+1]))
            if i != len(layers)-2:
                # self.net.append(nn.Tanh())
                self.net.append(nn.SiLU())  # Swish

    def forward(self, x, y):
        x_n = 2.0*x - 1.0
        y_n = 2.0*y - 1.0
        z = torch.cat([x_n, y_n], dim=1)

        for layer in self.net:
            z = layer(z)

        psi_net = z[:, 0:1]
        p_net   = z[:, 1:2]
        return psi_net, p_net

class HardBC_PsiP(nn.Module):
    def __init__(self, base_net):
        super().__init__()
        self.base_net = base_net

    def forward(self, x, y):
        psi_net, p_net = self.base_net(x, y)

        D = (x**2) * ((1-x)**2) * (y**2) * ((1-y)**2)
        u_lid = 16.0 * (x**2) * ((1-x)**2)
        psi_part = u_lid * (y**2) * (y - 1.0)

        psi = D * psi_net + psi_part
        p   = p_net - torch.mean(p_net)

        return psi, p

# ------------------------------------------------------------
# AUTOGRAD HELPERS
# ------------------------------------------------------------
def grad(f, x):
    return torch.autograd.grad(
        f, x,
        grad_outputs=torch.ones_like(f),
        create_graph=True
    )[0]

# ------------------------------------------------------------
# PHYSICS RESIDUALS
# ------------------------------------------------------------
def residuals(model, x, y, Re):
    psi, p = model(x, y)

    # Velocities
    psi_x = grad(psi, x)
    psi_y = grad(psi, y)

    u = psi_y
    v = -psi_x

    # Velocity gradients
    u_x = grad(u, x)
    u_y = grad(u, y)
    v_x = grad(v, x)
    v_y = grad(v, y)

    # Second derivatives
    u_xx = grad(u_x, x)
    u_yy = grad(u_y, y)
    v_xx = grad(v_x, x)
    v_yy = grad(v_y, y)

    # Pressure gradients
    p_x = grad(p, x)
    p_y = grad(p, y)

    # Momentum residuals (steady incompressible NS)
    r_u = u*u_x + v*u_y + p_x - (1/Re)*(u_xx + u_yy)
    r_v = u*v_x + v*v_y + p_y - (1/Re)*(v_xx + v_yy)

    return r_u, r_v, u, v

# ------------------------------------------------------------
# COLLOCATION POINTS (BOUNDARY-LAYER AWARE)
# ------------------------------------------------------------
Nf = 20000

# uniform interior
x1 = torch.rand(Nf//2, 1, device=device)
y1 = torch.rand(Nf//2, 1, device=device)

# boundary-layer biased sampling
x2 = torch.rand(Nf//4, 1, device=device)
y2 = torch.rand(Nf//4, 1, device=device)**3          # near y = 0

x3 = torch.rand(Nf//4, 1, device=device)
y3 = 1.0 - torch.rand(Nf//4, 1, device=device)**3    # near y = 1

x4 = torch.rand(Nf//4, 1, device=device)**3          # near x = 0
y4 = torch.rand(Nf//4, 1, device=device)

x5 = 1.0 - torch.rand(Nf//4, 1, device=device)**3    # near x = 1
y5 = torch.rand(Nf//4, 1, device=device)

x_f = torch.cat([x1, x2, x3, x4, x5], dim=0).requires_grad_(True)
y_f = torch.cat([y1, y2, y3, y4, y5], dim=0).requires_grad_(True)

# ------------------------------------------------------------
# CORNER-BIASED COLLOCATION
# ------------------------------------------------------------
Nc = Nf // 8

xc = torch.rand(Nc, 1, device=device)**3
yc = torch.rand(Nc, 1, device=device)**3

# bottom-left corner
x_bl, y_bl = xc, yc

# top-left corner
x_tl, y_tl = xc, 1.0 - yc

# bottom-right corner
x_br, y_br = 1.0 - xc, yc

# top-right corner
x_tr, y_tr = 1.0 - xc, 1.0 - yc

# append to collocation set
x_f = torch.cat([x_f, x_bl, x_tl, x_br, x_tr], dim=0).requires_grad_(True)
y_f = torch.cat([y_f, y_bl, y_tl, y_br, y_tr], dim=0).requires_grad_(True)

print("Total collocation points:", x_f.shape[0])

# ------------------------------------------------------------
# BOUNDARY CONDITIONS (CORRECT LID-DRIVEN CAVITY)
# Domain: [0,1] × [0,1]
# ------------------------------------------------------------
Nb = 2000
# ------------------------------------------------------------
# TOTAL LOSS (GT-AWARE, HARD-BC, RESIDUAL-AWARE)
# ------------------------------------------------------------
def total_loss(model, Re, lam, epoch):
    # --------------------------------------------------
    # PDE RESIDUALS (COLLOCATION POINTS)
    # --------------------------------------------------
    if lam["pde"] > 0.0:
        r_u, r_v, _, _ = residuals(model, x_f, y_f, Re)
        loss_pde = torch.mean(r_u**2) + torch.mean(r_v**2)
    else:
        loss_pde = torch.tensor(0.0, device=device)

    # --------------------------------------------------
    # GT SAMPLING
    # --------------------------------------------------
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

        idx = torch.cat([
            idx_focus[:1500],
            idx_rand[:1500]
        ])

    # --------------------------------------------------
    # GT DATA
    # --------------------------------------------------
    xg = x_gt[idx].clone().detach().requires_grad_(True)
    yg = y_gt[idx].clone().detach().requires_grad_(True)

    ug = u_gt[idx]
    vg = v_gt[idx]

    du_dy_g = du_dy_gt[idx]
    dv_dx_g = dv_dx_gt[idx]

    # --------------------------------------------------
    # PREDICTED FIELDS
    # --------------------------------------------------
    psi_p, p_p = model(xg, yg)

    psi_x_p = torch.autograd.grad(
        psi_p, xg,
        grad_outputs=torch.ones_like(psi_p),
        create_graph=True
    )[0]

    psi_y_p = torch.autograd.grad(
        psi_p, yg,
        grad_outputs=torch.ones_like(psi_p),
        create_graph=True
    )[0]

    u_p = psi_y_p
    v_p = -psi_x_p

    # --------------------------------------------------
    # VELOCITY GT LOSS
    # --------------------------------------------------
    loss_gt = (
        torch.mean((u_p - ug)**2) +
        torch.mean((v_p - vg)**2)
    )

    
    # --------------------------------------------------
    # CENTERLINE GRADIENT SHARPENING (LATE ONLY)
    # --------------------------------------------------
    loss_grad = torch.tensor(0.0, device=device)
    
    if epoch > 6000:
        mask_u = torch.abs(xg - 0.5) < 0.01
        mask_v = torch.abs(yg - 0.5) < 0.01
    
        if mask_u.any():
            du_dy_full = torch.autograd.grad(
                u_p,
                yg,
                grad_outputs=torch.ones_like(u_p),
                create_graph=True
            )[0]
    
            loss_grad += torch.mean(
                (du_dy_full[mask_u] - du_dy_g[mask_u])**2
            )
    
        if mask_v.any():
            dv_dx_full = torch.autograd.grad(
                v_p,
                xg,
                grad_outputs=torch.ones_like(v_p),
                create_graph=True
            )[0]
    
            loss_grad += torch.mean(
                (dv_dx_full[mask_v] - dv_dx_g[mask_v])**2
            )

    # --------------------------------------------------
    # PRESSURE GAUGE STABILIZATION
    # --------------------------------------------------
    loss_p_anchor = 5e-4 * torch.mean(p_p**2)

    # --------------------------------------------------
    # TOTAL LOSS
    # --------------------------------------------------
    total = (
        lam["pde"] * loss_pde +
        lam["gt"]  * loss_gt +
        0.2 * loss_grad +          # << SHARPNESS ENFORCER
        loss_p_anchor
    )

    return total, loss_pde, loss_gt

# ------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------
base_net = BaseNet([2, 96, 96, 96, 2]).to(device)
model = HardBC_PsiP(base_net).to(device)

# ----------------------------
# WEIGHT INITIALIZATION
# ----------------------------
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

model.apply(init_weights)

# ----------------------------
# SANITY CHECK (ONE-TIME)
# ----------------------------
for name, p in model.named_parameters():
    if "weight" in name:
        print("[init check]", name,
              "mean =", p.mean().item(),
              "std =", p.std().item())
        break

# ----------------------------
# OPTIMIZER
# ----------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=6000,
    eta_min=5e-5
)

Re = 1000
lam = {
    "pde": 0.0,
    "gt": 10.0
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
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    optimizer.step()
    
    if epoch >= 4000:
        scheduler.step()

    if epoch % 200 == 0:
       grad_norm = 0.0
       for p in model.parameters():
           if p.grad is not None:
               grad_norm += p.grad.norm()**2
       grad_norm = grad_norm**0.5

       print(
          f"Epoch {epoch:6d} | "
          f"Total {loss.item():.2e} | "
          f"PDE {lpde.item():.2e} | "
          f"GT {lgt.item():.2e} | "
          f"|∇θ| {grad_norm:.2e}"
      )

    if epoch % 1000 == 0:
        print("LR:", optimizer.param_groups[0]["lr"])

# ------------------------------------------------------------
# SAVE MODEL
# ------------------------------------------------------------
torch.save(model.state_dict(), "psi_p_gt_pinn.pth")
print("Model saved: psi_p_gt_pinn.pth")

# ----------------------------
# VISUALIZATION (HIGH-RES)
# ----------------------------
model.eval()

# ---------- Higher-res grid for visualization (no retrain) ----------
Nx_vis = 300
x = np.linspace(0, 1, Nx_vis)
y = np.linspace(0, 1, Nx_vis)
X, Y = np.meshgrid(x, y)

# prepare tensors for autograd
x_t = torch.tensor(X.flatten()[:, None], dtype=torch.float32, device=device).requires_grad_(True)
y_t = torch.tensor(Y.flatten()[:, None], dtype=torch.float32, device=device).requires_grad_(True)

# Forward pass
psi_pred, p_pred = model(x_t, y_t)

# ---------- First derivatives ----------
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

# ---------- Derived fields ----------
omega = -(psi_xx + psi_yy)

# Move to CPU and reshape
omega_np = omega.detach().cpu().numpy().reshape(Nx_vis, Nx_vis)
psi_np   = psi_pred.detach().cpu().numpy().reshape(Nx_vis, Nx_vis)
u_np     = psi_y.detach().cpu().numpy().reshape(Nx_vis, Nx_vis)
v_np     = -psi_x.detach().cpu().numpy().reshape(Nx_vis, Nx_vis)
speed_np = np.sqrt(u_np**2 + v_np**2)

# ----------------------------
# PLOTTING (improved)
# ----------------------------
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, axs = plt.subplots(2, 2, figsize=(12,10))

ax_psi    = axs[0,0]
ax_omega  = axs[0,1]
ax_vel    = axs[1,0]
ax_center = axs[1,1]

# -----------------------------
# STREAMLINES
# -----------------------------
ax_psi.contour(
    X, Y, psi_np,
    levels=60,
    colors="black",
    linewidths=0.6
)

ax_psi.set_title("Streamfunction (ψ contours)")
ax_psi.set_aspect("equal")
ax_psi.set_xlim(0,1)
ax_psi.set_ylim(0,1)

# -----------------------------
# VORTICITY (SHARP & SYMMETRIC)
# -----------------------------
omega_abs = np.abs(omega_np)
omega_max = np.percentile(omega_abs, 99.5)   # clip outliers for visibility; try 95/99 to taste

im_omega = ax_omega.contourf(
    X, Y, omega_np,
    levels=100,
    cmap="RdBu_r",
    vmin=-omega_max,
    vmax= omega_max,
    extend="both"
)

ax_omega.set_title("Vorticity (ω)")
ax_omega.set_aspect("equal")
ax_omega.set_xlim(0,1); ax_omega.set_ylim(0,1)

div1 = make_axes_locatable(ax_omega)
cax1 = div1.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im_omega, cax=cax1)

# -----------------------------
# VELOCITY MAGNITUDE
# -----------------------------
im_vel = ax_vel.contourf(X, Y, speed_np, levels=50, cmap="viridis")
ax_vel.set_title("|Velocity|")
ax_vel.set_aspect("equal")
ax_vel.set_xlim(0,1); ax_vel.set_ylim(0,1)

div2 = make_axes_locatable(ax_vel)
cax2 = div2.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im_vel, cax=cax2)

# -----------------------------
# CENTERLINE (use high-res grid)
# -----------------------------
mid = Nx_vis // 2
ax_center.plot(u_np[:,mid], y, label="u(x=0.5)")
ax_center.plot(x, v_np[mid,:], label="v(y=0.5)")
ax_center.legend()
ax_center.grid(True, ls="--", alpha=0.4)
ax_center.set_title("Centerline Velocities")
ax_center.set_box_aspect(1)

plt.suptitle(
    "GT-Aware Streamfunction–Pressure PINN | Lid-Driven Cavity",
    fontsize=14
)

plt.tight_layout()
plt.savefig("pinn_lid_cavity_psi_p_highres.png", dpi=300)
plt.show()
