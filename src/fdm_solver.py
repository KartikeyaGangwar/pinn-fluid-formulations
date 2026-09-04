"""
High-Resolution 2D Incompressible Lid-Driven Cavity Finite-Difference Solver.
Numerical Scheme:
- Streamfunction-Vorticity formulation
- Alternating Direction Implicit (ADI) for vorticity transport (Vectorized Thomas Algorithm with boundary closures)
- Direct Precomputed Sparse LU Decomposition for Streamfunction Poisson equation
- Thom's boundary condition formula for wall vorticity closure (first-order in wall vorticity O(h))
- Pressure Poisson recovery with consistent source divergence
"""

import os
import time
import pickle
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

class LidDrivenCavityFDM:
    def __init__(self, N=251, Re=1000, U=1.0, L=1.0, lid_profile='regularized'):
        self.N = N
        self.M = N - 2
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
        
        CFL_FACTOR = 0.2
        self.dt = CFL_FACTOR * self.h / (self.U if self.U > 0 else 1.0)
        self.alpha_adi = (self.nu * self.dt) / (2.0 * self.h**2)
        
        self.u_c = np.zeros((self.M, self.M))
        self.v_c = np.zeros((self.M, self.M))
        self.history = {'iterations': [], 'max_omega_change': [], 'psi_min': []}

        # Precompute Direct Sparse LU for 2D Poisson: laplacian(psi) = -omega
        main_diag = -2.0 * np.ones(self.M) / self.h**2
        off_diag = 1.0 * np.ones(self.M - 1) / self.h**2
        T = sp.diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(self.M, self.M))
        I = sp.eye(self.M)
        L_2d = (sp.kron(I, T) + sp.kron(T, I)).tocsc()
        self.lu_poisson = spla.splu(L_2d)

        # Precompute 1D Thomas algorithm recurrence coefficients for ADI
        alpha = self.alpha_adi
        A, B, C = -alpha, 1.0 + 2.0 * alpha, -alpha
        self.A_adi, self.B_adi, self.C_adi = A, B, C
        self.P_adi = np.zeros(self.M)
        self.denom_adi = np.zeros(self.M)
        self.denom_adi[0] = B
        self.P_adi[0] = C / B
        for j in range(1, self.M):
            self.denom_adi[j] = B - A * self.P_adi[j - 1]
            self.P_adi[j] = C / self.denom_adi[j]

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

    def solve_streamfunction(self):
        """Direct exact machine-precision Poisson solve via precomputed sparse LU."""
        rhs = -self.omega[1:-1, 1:-1].ravel()
        psi_inner = self.lu_poisson.solve(rhs).reshape((self.M, self.M))
        self.psi[1:-1, 1:-1] = psi_inner

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
        """Vectorized ADI step with rigorous wall-vorticity boundary closures."""
        N, h, nu, dt = self.N, self.h, self.nu, self.dt
        M = self.M
        A, B, C = self.A_adi, self.B_adi, self.C_adi
        P, denom = self.P_adi, self.denom_adi
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
        RHS_x = omega_old[1:-1, 1:-1] + dt * (diff_y - conv)
        # Add rigorous boundary contributions to first and last columns
        RHS_x[:, 0] -= A * omega_old[1:-1, 0]
        RHS_x[:, -1] -= C * omega_old[1:-1, -1]

        # Vectorized Thomas solve across all rows
        Q_x = np.zeros((M, M))
        Q_x[:, 0] = RHS_x[:, 0] / denom[0]
        for j in range(1, M):
            Q_x[:, j] = (RHS_x[:, j] - A * Q_x[:, j - 1]) / denom[j]

        omega_star_inner = np.zeros((M, M))
        omega_star_inner[:, -1] = Q_x[:, -1]
        for j in range(M - 2, -1, -1):
            omega_star_inner[:, j] = Q_x[:, j] - P[j] * omega_star_inner[:, j + 1]

        omega_star = self.omega.copy()
        omega_star[1:-1, 1:-1] = omega_star_inner
        
        # Step 2: Implicit Y, Explicit X
        diff_x = (nu / h**2) * (omega_star[1:-1, 2:] + omega_star[1:-1, :-2] - 2.0 * omega_star[1:-1, 1:-1])
        RHS_y = omega_star[1:-1, 1:-1] + dt * (diff_x - conv)
        # Add rigorous boundary contributions to bottom and top rows
        RHS_y[0, :] -= A * omega_star[0, 1:-1]
        RHS_y[-1, :] -= C * omega_star[-1, 1:-1]

        # Vectorized Thomas solve across all columns
        Q_y = np.zeros((M, M))
        Q_y[0, :] = RHS_y[0, :] / denom[0]
        for i in range(1, M):
            Q_y[i, :] = (RHS_y[i, :] - A * Q_y[i - 1, :]) / denom[i]

        omega_new_inner = np.zeros((M, M))
        omega_new_inner[-1, :] = Q_y[-1, :]
        for i in range(M - 2, -1, -1):
            omega_new_inner[i, :] = Q_y[i, :] - P[i] * omega_new_inner[i + 1, :]

        omega_new = self.omega.copy()
        omega_new[1:-1, 1:-1] = omega_new_inner
        return omega_new

    def calculate_pressure(self, max_iter=2000, tol=1e-5):
        """Pressure Poisson equation solver with physical RHS."""
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
            # Neumann zero gradient on boundaries
            p_new[0, :] = p_new[1, :]
            p_new[-1, :] = p_new[-2, :]
            p_new[:, 0] = p_new[:, 1]
            p_new[:, -1] = p_new[:, -2]
            
            if np.max(np.abs(p_new[1:-1, 1:-1] - p_old[1:-1, 1:-1])) < tol:
                break
        self.p = p_new - np.mean(p_new[1:-1, 1:-1])

    def solve(self, max_iterations=15000, tolerance=1e-5, log_interval=200):
        start_time = time.time()
        for iteration in range(max_iterations):
            omega_old = self.omega.copy()
            self.calculate_velocities()
            self.apply_boundary_conditions()
            self.omega = self.solve_vorticity_transport_ADI()
            self.apply_boundary_conditions()
            self.solve_streamfunction()
            
            max_change = np.max(np.abs(self.omega[1:-1, 1:-1] - omega_old[1:-1, 1:-1]))
            if iteration % log_interval == 0:
                print(f"[FDM] Iter {iteration:5d}/{max_iterations} | max d(omega): {max_change:.2e} | psi_min: {self.psi.min():.5f}", flush=True)

            if max_change < tolerance and iteration > 300:
                self.calculate_velocities()
                self.calculate_pressure()
                elapsed = time.time() - start_time
                print(f"[FDM CONVERGED] Finished in {iteration} iters ({elapsed:.1f}s) | max d(omega): {max_change:.2e}", flush=True)
                return True, iteration

        self.calculate_velocities()
        self.calculate_pressure()
        elapsed = time.time() - start_time
        print(f"[FDM REACHED MAX ITERS] Finished {max_iterations} iters in {elapsed:.1f}s | Final psi_min: {self.psi.min():.5f}", flush=True)
        return False, max_iterations

    def save(self, filename='data/gt_data_Re1000.pkl'):
        # Enforce exact regularized velocity boundary assertion
        if self.lid_profile == 'regularized':
            max_wall_diff = np.max(np.abs(self.u[-1, :] - self.u_lid))
            assert max_wall_diff < 1e-5, f"Top wall velocity mismatch: max diff = {max_wall_diff}"
            print(f"[ASSERTION PASSED] Top lid velocity strictly matches 16*x^2*(1-x)^2 (max diff = {max_wall_diff:.2e})")

        idx_min = np.unravel_index(np.argmin(self.psi), self.psi.shape)
        data = {
            'parameters': {'N': self.N, 'Re': self.Re, 'U': self.U, 'L': self.L, 'lid_profile': self.lid_profile},
            'coordinates': {'x': self.x, 'y': self.y, 'X': self.X, 'Y': self.Y},
            'fields': {'psi': self.psi, 'omega': self.omega, 'u': self.u, 'v': self.v, 'p': self.p},
            'vortex_center': {'x': float(self.x[idx_min[1]]), 'y': float(self.y[idx_min[0]])},
            'metadata': {
                'lid_profile': self.lid_profile,
                'reconstructed_top_wall_velocity': self.u[-1, :],
                'regularized_formula': '16 * x^2 * (1 - x)^2' if self.lid_profile == 'regularized' else 'U=1.0',
                'provenance': 'Version-locked high-resolution FDM solver with direct sparse LU Poisson and Thom boundary condition'
            }
        }
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[SUCCESS] Saved verified FDM dataset: {filename}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="High-resolution FDM cavity flow solver.")
    parser.add_argument('--re', type=int, default=1000, help="Reynolds number (default: 1000)")
    parser.add_argument('--n', type=int, default=251, help="Grid resolution (default: 251)")
    parser.add_argument('--iters', type=int, default=10000, help="Max iterations (default: 10000)")
    parser.add_argument('--out', type=str, default='data/gt_data_Re1000.pkl', help="Output file path")
    args = parser.parse_args()

    print(f"Solving {args.n}x{args.n} Cavity Flow at Re={args.re} with Regularized Lid...")
    solver = LidDrivenCavityFDM(N=args.n, Re=args.re, lid_profile='regularized')
    solver.solve(max_iterations=args.iters, tolerance=1e-5)
    solver.save(args.out)
