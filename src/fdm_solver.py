"""
High-Resolution 2D Incompressible Lid-Driven Cavity Finite-Difference Solver.
Numerical Scheme:
- Streamfunction-Vorticity formulation
- Alternating Direction Implicit (ADI) for vorticity transport
- Red-Black Successive Over-Relaxation (SOR) for streamfunction Poisson equation
- Thom's second-order boundary condition for wall vorticity closure
- Jacobi relaxation for Pressure Poisson equation
"""

import numpy as np
import time
import pickle

class LidDrivenCavityFDM:
    def __init__(self, N=251, Re=1000, U=1.0, L=1.0, lid_profile='regularized'):
        self.N = N
        self.Re = Re
        self.U = U
        self.L = L
        self.lid_profile = lid_profile
        
        self.h = L / (N - 1)
        self.nu = U * L / Re
        
        self.x = np.linspace(0, L, N)
        self.y = np.linspace(0, L, N)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='xy')

        if self.lid_profile == 'regularized':
            self.u_lid = 16.0 * (self.x**2) * ((1.0 - self.x)**2)
        else:
            self.u_lid = np.full(N, U)
        
        self.psi = np.zeros((N, N))
        self.omega = np.zeros((N, N))
        self.u = np.zeros((N, N))
        self.v = np.zeros((N, N))
        self.p = np.zeros((N, N))
        
        CFL_FACTOR = 0.1
        self.dt = CFL_FACTOR * self.h / (self.U if self.U > 0 else 1.0)
        self.alpha_adi = (self.nu * self.dt) / (2.0 * self.h**2)
        
        self.u_c = np.zeros((N - 2, N - 2))
        self.v_c = np.zeros((N - 2, N - 2))
        self.history = {'iterations': [], 'max_omega_change': [], 'psi_min': []}

    def apply_boundary_conditions(self):
        h = self.h
        
        # Streamfunction BCs: psi = 0 on all walls
        self.psi[0, :] = 0.0
        self.psi[-1, :] = 0.0
        self.psi[:, 0] = 0.0
        self.psi[:, -1] = 0.0
        
        # Thom's formula for wall vorticity
        self.omega[-1, 1:-1] = -2.0 * self.psi[-2, 1:-1] / h**2 - 2.0 * self.u_lid[1:-1] / h # Top lid
        self.omega[0, 1:-1]  = -2.0 * self.psi[1, 1:-1] / h**2                                # Bottom
        self.omega[1:-1, 0]  = -2.0 * self.psi[1:-1, 1] / h**2                                # Left
        self.omega[1:-1, -1] = -2.0 * self.psi[1:-1, -2] / h**2                               # Right
        
        # Corner averaging
        self.omega[0, 0]   = 0.5 * (self.omega[1, 0] + self.omega[0, 1])
        self.omega[0, -1]  = 0.5 * (self.omega[1, -1] + self.omega[0, -2])
        self.omega[-1, 0]  = 0.5 * (self.omega[-2, 0] + self.omega[-1, 1])
        self.omega[-1, -1] = 0.5 * (self.omega[-2, -1] + self.omega[-1, -2])

    def solve_streamfunction(self, max_iterations=1000, tolerance=1e-5, omega_rb=1.8):
        N = self.N
        h = self.h
        i_v, j_v = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
        red = ((i_v + j_v) % 2 == 0) & (i_v > 0) & (i_v < N-1) & (j_v > 0) & (j_v < N-1)
        black = ~red & (i_v > 0) & (i_v < N-1) & (j_v > 0) & (j_v < N-1)
        
        source = h**2 * self.omega
        for _ in range(max_iterations):
            psi_old = self.psi.copy()
            # Red sweep
            calc_red = 0.25 * (
                np.roll(self.psi, 1, axis=0)[red] + np.roll(self.psi, -1, axis=0)[red] +
                np.roll(self.psi, 1, axis=1)[red] + np.roll(self.psi, -1, axis=1)[red] + source[red]
            )
            self.psi[red] = (1.0 - omega_rb) * self.psi[red] + omega_rb * calc_red
            
            # Black sweep
            calc_black = 0.25 * (
                np.roll(self.psi, 1, axis=0)[black] + np.roll(self.psi, -1, axis=0)[black] +
                np.roll(self.psi, 1, axis=1)[black] + np.roll(self.psi, -1, axis=1)[black] + source[black]
            )
            self.psi[black] = (1.0 - omega_rb) * self.psi[black] + omega_rb * calc_black
            
            max_change = np.max(np.abs(self.psi - psi_old))
            if max_change < tolerance:
                break
        return max_change

    def calculate_velocities(self):
        h = self.h
        self.u[1:-1, 1:-1] = (self.psi[2:, 1:-1] - self.psi[0:-2, 1:-1]) / (2.0 * h)
        self.v[1:-1, 1:-1] = -(self.psi[1:-1, 2:] - self.psi[1:-1, 0:-2]) / (2.0 * h)
        
        self.u_c = self.u[1:-1, 1:-1]
        self.v_c = self.v[1:-1, 1:-1]
        
        self.u[-1, :] = self.u_lid
        self.v[-1, :] = 0.0
        self.u[0, :] = 0.0
        self.v[0, :] = 0.0
        self.u[:, 0] = 0.0
        self.v[:, 0] = 0.0
        self.u[:, -1] = 0.0
        self.v[:, -1] = 0.0

    def solve_vorticity_transport_ADI(self):
        N, h, nu, dt = self.N, self.h, self.nu, self.dt
        alpha = self.alpha_adi
        omega_old = self.omega.copy()
        
        # Upwind convection
        domega_dx = np.where(
            self.u_c > 0,
            (omega_old[1:-1, 1:-1] - omega_old[1:-1, :-2]) / h,
            (omega_old[1:-1, 2:] - omega_old[1:-1, 1:-1]) / h
        )
        domega_dy = np.where(
            self.v_c > 0,
            (omega_old[1:-1, 1:-1] - omega_old[:-2, 1:-1]) / h,
            (omega_old[2:, 1:-1] - omega_old[1:-1, 1:-1]) / h
        )
        conv = self.u_c * domega_dx + self.v_c * domega_dy
        
        # Step 1: Implicit X, Explicit Y
        diff_y = (nu / h**2) * (omega_old[2:, 1:-1] + omega_old[:-2, 1:-1] - 2.0 * omega_old[1:-1, 1:-1])
        omega_star = np.zeros_like(self.omega)
        omega_star[0, :] = self.omega[0, :]
        omega_star[-1, :] = self.omega[-1, :]
        omega_star[:, 0] = self.omega[:, 0]
        omega_star[:, -1] = self.omega[:, -1]
        
        A_x, B_x, C_x = -alpha, 1.0 + 2.0 * alpha, -alpha
        for i in range(1, N - 1):
            RHS_x = omega_old[i, 1:-1] + dt * (diff_y[i-1, :] - conv[i-1, :])
            P = np.zeros(N - 2)
            Q = np.zeros(N - 2)
            P[0] = C_x / B_x
            Q[0] = RHS_x[0] / B_x
            for j in range(1, N - 2):
                denom = B_x - A_x * P[j-1]
                P[j] = C_x / denom
                Q[j] = (RHS_x[j] - A_x * Q[j-1]) / denom
            omega_star[i, -2] = Q[N - 3]
            for j in range(N - 4, -1, -1):
                omega_star[i, j+1] = Q[j] - P[j] * omega_star[i, j+2]
                
        # Step 2: Implicit Y, Explicit X
        diff_x = (nu / h**2) * (omega_star[1:-1, 2:] + omega_star[1:-1, :-2] - 2.0 * omega_star[1:-1, 1:-1])
        omega_new = np.zeros_like(self.omega)
        omega_new[0, :] = self.omega[0, :]
        omega_new[-1, :] = self.omega[-1, :]
        omega_new[:, 0] = self.omega[:, 0]
        omega_new[:, -1] = self.omega[:, -1]
        
        A_y, B_y, C_y = -alpha, 1.0 + 2.0 * alpha, -alpha
        for j in range(1, N - 1):
            RHS_y = omega_star[1:-1, j] + dt * (diff_x[:, j-1] - conv[:, j-1])
            P = np.zeros(N - 2)
            Q = np.zeros(N - 2)
            P[0] = C_y / B_y
            Q[0] = RHS_y[0] / B_y
            for i in range(1, N - 2):
                denom = B_y - A_y * P[i-1]
                P[i] = C_y / denom
                Q[i] = (RHS_y[i] - A_y * Q[i-1]) / denom
            omega_new[-2, j] = Q[N - 3]
            for i in range(N - 4, -1, -1):
                omega_new[i+1, j] = Q[i] - P[i] * omega_new[i+2, j]
                
        return omega_new

    def calculate_pressure(self, max_iter=5000, tol=1e-5):
        N, h = self.N, self.h
        dudx = (self.u[1:-1, 2:] - self.u[1:-1, :-2]) / (2.0 * h)
        dudy = (self.u[2:, 1:-1] - self.u[:-2, 1:-1]) / (2.0 * h)
        dvdx = (self.v[1:-1, 2:] - self.v[1:-1, :-2]) / (2.0 * h)
        dvdy = (self.v[2:, 1:-1] - self.v[:-2, 1:-1]) / (2.0 * h)

        rhs = -(dudx**2 + 2.0 * dudy * dvdx + dvdy**2)
        
        p_new = self.p.copy()
        for _ in range(max_iter):
            p_old = p_new.copy()
            p_new[1:-1, 1:-1] = 0.25 * (
                p_old[2:, 1:-1] + p_old[0:-2, 1:-1] +
                p_old[1:-1, 2:] + p_old[1:-1, 0:-2] - h**2 * rhs
            )
            if np.max(np.abs(p_new[1:-1, 1:-1] - p_old[1:-1, 1:-1])) < tol:
                break
        self.p = p_new - np.mean(p_new[1:-1, 1:-1])

    def solve(self, max_iterations=50000, tolerance=1e-6):
        start_time = time.time()
        for iteration in range(max_iterations):
            omega_old = self.omega.copy()
            self.calculate_velocities()
            self.apply_boundary_conditions()
            self.omega = self.solve_vorticity_transport_ADI()
            self.apply_boundary_conditions()
            self.solve_streamfunction(max_iterations=1000, tolerance=1e-4)
            
            max_change = np.max(np.abs(self.omega[1:-1, 1:-1] - omega_old[1:-1, 1:-1]))
            if max_change < tolerance and iteration > 500:
                self.calculate_velocities()
                self.calculate_pressure()
                return True, iteration
        self.calculate_velocities()
        self.calculate_pressure()
        return False, max_iterations

    def save(self, filename='gt_data_Re1000.pkl'):
        idx_min = np.unravel_index(np.argmin(self.psi), self.psi.shape)
        data = {
            'parameters': {'N': self.N, 'Re': self.Re, 'U': self.U, 'L': self.L},
            'coordinates': {'x': self.x, 'y': self.y, 'X': self.X, 'Y': self.Y},
            'fields': {'psi': self.psi, 'omega': self.omega, 'u': self.u, 'v': self.v, 'p': self.p},
            'vortex_center': {'x': float(self.x[idx_min[1]]), 'y': float(self.y[idx_min[0]])}
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        np.savez(f'flow_fields_Re{self.Re}_N{self.N}.npz',
                 x=self.x, y=self.y, X=self.X, Y=self.Y,
                 psi=self.psi, omega=self.omega, u=self.u, v=self.v, p=self.p)
        print(f"[SUCCESS] Saved FDM dataset: {filename}")
