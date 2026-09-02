"""
Physics-Informed Neural Network (PINN) Architectures and Exact Boundary Lifting Operators.
"""

import torch
import torch.nn as nn

class BaseNet(nn.Module):
    def __init__(self, layers, activation='silu'):
        super().__init__()
        self.net = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.net.append(nn.Linear(layers[i], layers[i+1]))
            if i != len(layers) - 2:
                if activation == 'silu':
                    self.net.append(nn.SiLU())
                elif activation == 'tanh':
                    self.net.append(nn.Tanh())
                elif activation == 'gelu':
                    self.net.append(nn.GELU())
                else:
                    self.net.append(nn.SiLU())
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, y):
        # Normalize coordinates to [-1, 1]
        x_n = 2.0 * x - 1.0
        y_n = 2.0 * y - 1.0
        z = torch.cat([x_n, y_n], dim=1)
        for layer in self.net:
            z = layer(z)
        return z

class HardBC_PsiP(nn.Module):
    """
    Exact Hard Boundary Condition for Streamfunction-Pressure (psi-p) PINN.
    Boundary conditions enforced identically:
      - All 4 walls: psi = 0 (no penetration)
      - Top lid (y = 1): u = dpsi/dy = 16 x^2 (1-x)^2 (regularized top lid)
      - Bottom/Left/Right: u = 0, v = 0 (no-slip)
    """
    def __init__(self, base_net):
        super().__init__()
        self.base_net = base_net

    def forward(self, x, y):
        z = self.base_net(x, y)
        psi_net = z[:, 0:1]
        p_net   = z[:, 1:2]

        D = (x**2) * ((1.0 - x)**2) * (y**2) * ((1.0 - y)**2)
        u_lid = 16.0 * (x**2) * ((1.0 - x)**2)
        psi_part = u_lid * (y**2) * (y - 1.0)

        psi = D * psi_net + psi_part
        p   = p_net

        return psi, p

class HardBC_PsiOmega(nn.Module):
    """
    Exact Hard Boundary Condition for Streamfunction-Vorticity (psi-omega) PINN.
    """
    def __init__(self, base_net):
        super().__init__()
        self.base_net = base_net

    def forward(self, x, y):
        z = self.base_net(x, y)
        psi_net   = z[:, 0:1]
        omega_net = z[:, 1:2]

        D = (x**2) * ((1.0 - x)**2) * (y**2) * ((1.0 - y)**2)
        u_lid = 16.0 * (x**2) * ((1.0 - x)**2)
        psi_part = u_lid * (y**2) * (y - 1.0)

        psi = D * psi_net + psi_part
        omega = omega_net

        return psi, omega

class HardBC_VelocityPressure(nn.Module):
    """
    Direct Velocity-Pressure (u-v-p) PINN with exact Dirichlet boundary conditions.
    """
    def __init__(self, base_net):
        super().__init__()
        self.base_net = base_net

    def forward(self, x, y):
        z = self.base_net(x, y)
        u_net = z[:, 0:1]
        v_net = z[:, 1:2]
        p_net = z[:, 2:3]

        D_xy = x * (1.0 - x) * y * (1.0 - y)
        u_lid = 16.0 * (x**2) * ((1.0 - x)**2)

        u = y * u_lid + D_xy * u_net
        v = D_xy * v_net
        p = p_net - torch.mean(p_net)

        return u, v, p
