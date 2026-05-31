# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 19:58:52 2025

@author: OU_HAO-ISCT
"""

import numpy as np
import time
from scipy.optimize import minimize
import gc
class TTGPaperSolver:
    """
    Strict implementation of the continuum model relaxation for Twisted Trilayer Graphene.
    Reference: Nakatsuji et al. (2023) Physical Review X
    Method: Explicit G-vector summation (Galerkin method), no FFT grid approximation.
    Includes: Layer 3 Translation Scanning & Energy Optimization.
    """
    def __init__(self):
        # --- Physical Constants ---
        # Converted to consistent units: Energy in eV, Length in nm
        #print(">>> Initializing solver")
        self.mater = 'WSe2'
        if self.mater == 'WSe2':
            self.a_nm = 0.332  # WSe2 lattice constant [nm]
            self.V0 = 0.07889*1  # Interlayer potential [eV/nm^2]  WSe2
            self.lam = 185.2  # Lambda = 185.2 eV/nm^2
            self.mu = 302.2  # Mu = 302.2 eV/nm^2
        elif self.mater == 'TTG':
            self.a_nm = 0.246  # Graphene lattice constant [nm]
            self.V0 = 0.16  # Interlayer potential [eV/nm^2]   gr
            # Elastic moduli [eV/nm^2] (1 eV/A^2 = 100 eV/nm^2)
            self.lam = 325  # Lambda = 325 eV/nm^2  gr
            self.mu = 957  # Mu = 302.2 eV/nm^2    gr
        else:
            raise Exception("Unknown mater")

        self.v13_ratio = 0  # Stronger than graphene due to d-orbitals
        self.V0_13 = self.V0 * self.v13_ratio
        
        # --- Solver Settings (Default) ---
        self.max_iter = 50000       # Safe upper limit
        self.tolerance = 1e-5      # Convergence tolerance
        self.grid_density_base = 64  # Will be adjusted based on needs
        
        # Supercell Factor: 1 = Single MoM cell, 2 = 2x2 MoM cells directly calculated
        self.supercell_factor = 1
        
        # Dynamic Alpha
        self.alpha_init = 0.01
        self.alpha_max = 1.0
        self.alpha_inc = 1.05
        self.alpha_dec = 0.7

        self.alpha_start = 0.5  # High initial learning rate
        self.alpha_end = 0.05  # Low final learning rate
        self.momentum = 0.9  # Momentum coefficient (0.8-0.95)
        
        self.use_modified_signs = True
        self.use_smart_init = False

    def _rotate_vec(self, vec, theta_rad):
        c, s = np.cos(theta_rad), np.sin(theta_rad)
        return np.array([c*vec[0] - s*vec[1], s*vec[0] + c*vec[1]])

    def setup_system(self, theta12, theta23, n, m,n_p,m_p):
        """
        Setup system. Adapts G-vectors for a larger supercell if factor > 1.
        """
        #print(f">>> Setup System: ({theta12}°, {theta23}°) | Integers: ({n}, {m})")
        #print(f"    Supercell Factor: {self.supercell_factor} (Calculating {self.supercell_factor}x{self.supercell_factor} area directly)")
        
        self.theta12 = np.deg2rad(theta12)
        self.theta23 = np.deg2rad(theta23)
        self.n, self.m = n, m
        self.n_p,self.m_p = n_p,m_p
        
        # Adjust Grid Density: Scale with supercell factor to keep resolution constant
        self.grid_density = int(self.grid_density_base * self.supercell_factor * (1 + 0.05 * max(n, m)))
        #print(f"    Grid Density: {self.grid_density}x{self.grid_density}")
        
        self._build_geometry()
        self._generate_G_vectors()
        self._precompute_grid()

    def _build_geometry(self):
        # 1. Intrinsic Lattice
        k_val = 2 * np.pi / self.a_nm
        self.b1_0 = np.array([k_val, -k_val/np.sqrt(3)])
        self.b2_0 = np.array([0, 2*k_val/np.sqrt(3)])
        self.b3_0 = -self.b1_0 - self.b2_0
        self.b_vecs_intrinsic = [self.b1_0, self.b2_0, self.b3_0]
        
        # Rotated layers
        b1_l1 = self._rotate_vec(self.b1_0, -self.theta12)
        b2_l1 = self._rotate_vec(self.b2_0, -self.theta12)
        b1_l2 = self.b1_0; b2_l2 = self.b2_0
        b1_l3 = self._rotate_vec(self.b1_0, self.theta23)
        b2_l3 = self._rotate_vec(self.b2_0, self.theta23)

        self.G1_12 = b1_l1 - b1_l2; self.G2_12 = b2_l1 - b2_l2
        self.G1_23 = b1_l2 - b1_l3; self.G2_23 = b2_l2 - b2_l3
        self.G1_13 = b1_l1 - b1_l3;
        self.G2_13 = b2_l1 - b2_l3  # Direct 1-3
        
        self.G_vecs_12 = [self.G1_12, self.G2_12, -self.G1_12 - self.G2_12]
        self.G_vecs_23 = [self.G1_23, self.G2_23, -self.G1_23 - self.G2_23]
        self.G_vecs_13 = [self.G1_13, self.G2_13, -self.G1_13 - self.G2_13]

        # 3. Super Moiré Lattice (Single Cell Definition)
        def get_dual(g1, g2):
            cross_z = g1[0]*g2[1] - g1[1]*g2[0]
            l1 = (2*np.pi/cross_z) * np.array([g2[1], -g2[0]])
            l2 = (2*np.pi/cross_z) * np.array([-g1[1], g1[0]])
            return l1, l2
            
        L1_12, L2_12 = get_dual(self.G1_12, self.G2_12)
        self.L1_SM_single = self.n * L1_12 + self.m * L2_12
        self.L2_SM_single = self._rotate_vec(self.L1_SM_single, np.pi/3)
        
        # --- SCALE UP THE DOMAIN ---
        # The calculation domain L_SM is now factor * L_single
        self.L1_SM = self.L1_SM_single * self.supercell_factor
        self.L2_SM = self.L2_SM_single * self.supercell_factor
        
        # Reciprocal vectors for this LARGER domain will be SMALLER (finer spacing)
        #self.G1_SM, self.G2_SM = self._get_reciprocal(self.L1_SM, self.L2_SM)
        self.G1_SM = ((self.n + self.m) * self.G1_12 + self.m * self.G2_12) / (self.n ** 2 + self.m ** 2 + self.n * self.m)
        self.G2_SM = (-self.m * self.G1_12 + self.n * self.G2_12) / (self.n ** 2 + self.m ** 2 + self.n * self.m)
        #print(self.G1_12)
        #self.G_vecs_SM = self.G_vecs_23-self.G_vecs_12
        #self.G1_SM,self.G2_SM = np.array(self.G1_23)-np.array(self.G1_12), np.array(self.G2_23)-np.array(self.G2_12)
        
        self.period_len = np.linalg.norm(self.L1_SM) # This is the full box size
        #print(f"    Calculation Box Size L = {self.period_len:.2f} nm (Factor={self.supercell_factor})")

    
    def _get_reciprocal(self, a1, a2):
        cross_z = a1[0]*a2[1] - a1[1]*a2[0]
        b1 = (2 * np.pi / cross_z) * np.array([a2[1], -a2[0]])
        b2 = (2 * np.pi / cross_z) * np.array([-a1[1], a1[0]])
        return b1, b2

    def _generate_G_vectors(self):
        """
        Generate G vectors.
        IMPORTANT: Because G_SM is finer (smaller), we need a larger index limit
        to reach the same physical cutoff k_max.
        """
        # Physical cutoff based on Single Cell parameters
        # We want to cover the same physics, so k_cutoff depends on MoM structure, not box size.
        # k_cutoff ~ 3 * G_moire
        
        # Estimate G_moire magnitude from the single cell N, M relation
        # G_moire approx G_SM_single * sqrt(n^2+...)
        # We can just use the same logic: cover harmonics of the fundamental moire frequency.
        
        # Magnitude of the SM vector for the SINGLE cell
        G_SM_single_mag = np.linalg.norm(self.G1_SM) * self.supercell_factor
        
        # Physical cutoff in 1/nm
        max_idx_single =3 * max(abs(self.n), abs(self.m))
        k_cutoff_phys = max_idx_single * G_SM_single_mag
        
        # In terms of our finer grid indices:
        # |i * G1 + j * G2| < k_cutoff
        # Since |G1| is 1/factor of before, i and j will need to be factor times larger.
        
        limit_idx = int(max_idx_single * self.supercell_factor * 1.5)
        #print(f"    G-vector generation limit index: {limit_idx} (Physical k_cut={k_cutoff_phys:.2f})")
        
        g_list = []
        find_mom_base=0
        find_mom_second=0
        find_mom_third=0
        find_mom_forth=0
        find_mom_fifth=0
        find_m12_base = 0
        find_m12_second = 0
        find_m23_base = 0
        find_m23_second = 0
        self.m12_second = 0
        self.m23_second = 0
        for i in range(-limit_idx, limit_idx+1):
            for j in range(-limit_idx, limit_idx+1):
                if i==0 and j==0: continue
                g_vec = i * self.G1_SM + j * self.G2_SM
                if np.linalg.norm(g_vec) <= k_cutoff_phys:
                    g_list.append(g_vec)
                    if i ==1 and j ==0:
                        self.mom_base=find_mom_base
                        #print(f"MoM base harmony index= {self.mom_base}")
                    if i ==1 and j ==-1:
                        self.mom_second = find_mom_second
                        #print(f"MoM second harmony index= {self.mom_second}")
                    if i ==2 and j ==0:
                        self.mom_third = find_mom_third
                        #print(f"MoM third harmony index= {self.mom_third}")
                    if i ==2 and j ==1:
                        self.mom_forth = find_mom_forth
                        #print(f"MoM forth harmony index= {self.mom_forth}")
                    if i ==3 and j ==0:
                        self.mom_fifth = find_mom_fifth
                        #print(f"MoM fifth harmony index= {self.mom_fifth}")
                    if i==self.n and j == -self.m:
                        self.m12_base = find_m12_base
                        #print(f"Moire_12 base harmony index= {self.m_base}")
                    if i==self.n_p and j == -self.m_p:
                        self.m23_base = find_m23_base
                    if i==2*self.n and j == -2*self.m:
                        self.m12_second = find_m12_second
                    if i==2*self.n_p and j == -2*self.m_p:
                        self.m23_second = find_m23_second
                    find_mom_base+=1
                    find_mom_second+=1
                    find_mom_third+=1
                    find_mom_forth+=1
                    find_mom_fifth+=1
                    find_m12_base+=1
                    find_m12_second+=1
                    find_m23_base+=1
                    find_m23_second+=1
        self.harmonics=[self.mom_base,self.mom_second,self.mom_third,self.mom_forth,self.mom_fifth,self.m12_base,self.m23_base]
        
        self.G_vecs = np.array(g_list) 
        #print(self.G_vecs[self.m_base])
        self.N_G = len(self.G_vecs)
        #print(f"    Number of G vectors: {self.N_G}")
        
        # Precompute K matrices
        self.K_inv = np.zeros((self.N_G, 2, 2))
        self.K_mat = np.zeros((self.N_G, 2, 2))
        
        Gx, Gy = self.G_vecs[:, 0], self.G_vecs[:, 1]
        A = self.lam + 2*self.mu
        B = self.mu
        C = self.lam + self.mu
        
        Kxx = A*Gx**2 + B*Gy**2
        Kyy = A*Gy**2 + B*Gx**2
        Kxy = C*Gx*Gy
        
        Det = Kxx*Kyy - Kxy**2
        
        self.K_inv[:, 0, 0] = Kyy / Det
        self.K_inv[:, 1, 1] = Kxx / Det
        self.K_inv[:, 0, 1] = -Kxy / Det
        self.K_inv[:, 1, 0] = -Kxy / Det
        
        self.K_mat[:, 0, 0] = Kxx
        self.K_mat[:, 1, 1] = Kyy
        self.K_mat[:, 0, 1] = Kxy
        self.K_mat[:, 1, 0] = Kxy

    def _precompute_grid(self):
        """Precompute Real Space Grid and Phase Factors."""
        N = self.grid_density
        n1 = np.linspace(-1/2, 1/2, N, endpoint=False)
        n2 = np.linspace(-1/2, 1/2, N, endpoint=False)
        N1, N2 = np.meshgrid(n1, n2)
        
        self.R_real = np.zeros((N*N, 2))
        self.R_real[:, 0] = (N1.flatten() * self.L1_SM[0] + N2.flatten() * self.L2_SM[0])
        self.R_real[:, 1] = (N1.flatten() * self.L1_SM[1] + N2.flatten() * self.L2_SM[1])
        
        # Phase Matrix (N_pts, N_G)
        phase = np.dot(self.R_real, self.G_vecs.T)
        self.exp_iGr = np.exp(1j * phase).astype(np.complex64)
        self.exp_minus_iGr = np.conjugate(self.exp_iGr)

    def solve(self, s_shift=np.zeros(2), quick_run=False):
        """
        Robust Solver using Momentum + Adaptive Restart.
        This method is much more stable than simple Gradient Descent.
        """
        # 1. Initialization with STRONGER Noise (to find curved domains)
        # 使用时间作为种子，确保每次运行不同
        np.random.seed(int(time.time() * 1000) % 2 ** 32)

        # 增大初始噪声幅度，有助于打破对称性，形成弯曲畴壁
        # 建议 0.02 ~ 0.05 * a_nm
        random_amp = 0.00 * self.a_nm

        u_G = (np.random.randn(self.N_G, 2) + 1j * np.random.randn(self.N_G, 2)) * random_amp
        v_G = (np.random.randn(self.N_G, 2) + 1j * np.random.randn(self.N_G, 2)) * random_amp

        # Velocity buffers for Momentum
        vel_u = np.zeros_like(u_G)
        vel_v = np.zeros_like(v_G)

        # Parameters
        iters = 2000 if quick_run else 8000  # 增加最大步数，给动量法足够时间
        tol = 1e-5  # Residual Tolerance (1e-5 is physically sufficient)

        # 动态参数
        alpha = 0.05  # Initial Learning rate
        momentum = 0.9  # Inertia factor (0.8 - 0.95)
        alpha_min = 1e-4
        alpha_max = 0.2

        prev_resid = 1e9
        best_resid = 1e9
        best_state = (u_G.copy(), v_G.copy())

        print(f"    Running Robust Solver (Momentum={momentum}, Init Alpha={alpha}, Noise={random_amp:.3f})...")

        for it in range(iters):
            if it == 1:
                print("successful iteration")
            # 1. G -> Real
            u_real = np.dot(self.exp_iGr, u_G)
            v_real = np.dot(self.exp_iGr, v_G)

            # 2. Compute Forces (Standard)
            Force_u_acc = np.zeros((self.N_G, 2), dtype=complex)
            Force_v_acc = np.zeros((self.N_G, 2), dtype=complex)

            for j in range(3):
                G12, G23 = self.G_vecs_12[j], self.G_vecs_23[j]
                b_vec = self.b_vecs_intrinsic[j]
                shift_phase = np.dot(b_vec, s_shift)

                arg1 = np.dot(self.R_real, G12) - np.dot(u_real + v_real, b_vec) * 0.5 + shift_phase
                arg2 = np.dot(self.R_real, G23) + np.dot(u_real - v_real, b_vec) * 0.5

                sin_1 = np.sin(arg1)
                sin_2 = np.sin(arg2)

                f12_G = np.dot(sin_1, self.exp_minus_iGr) / (self.grid_density ** 2)
                f23_G = np.dot(sin_2, self.exp_minus_iGr) / (self.grid_density ** 2)

                K_inv_b = np.einsum('nij,j->ni', self.K_inv, b_vec)

                if self.use_modified_signs:
                    Force_u_acc += (f12_G - f23_G)[:, None] * K_inv_b
                    Force_v_acc += (f12_G + f23_G)[:, None] * K_inv_b
                else:
                    Force_u_acc += (f12_G + f23_G)[:, None] * K_inv_b
                    Force_v_acc += (f12_G - f23_G)[:, None] * K_inv_b

            # 3. Calculate Residual (Driving Force)
            # u_target is the equilibrium position if forces were constant
            u_target = -6 * self.V0 * Force_u_acc
            v_target = -2 * self.V0 * Force_v_acc

            resid_u = u_target - u_G
            resid_v = v_target - v_G

            # Current "Force" magnitude
            current_resid = np.linalg.norm(resid_u) + np.linalg.norm(resid_v)

            # --- Robust Update Strategy ---

            # 如果残差变大 (Overshooting)，说明跑过头了或者震荡了
            if current_resid > prev_resid * 1.05:  # 允许微小波动，但惩罚大幅上升
                # 1. 紧急刹车：清空动量
                vel_u *= 0
                vel_v *= 0
                # 2. 减小步长
                alpha = max(alpha * 0.5, alpha_min)

                # (可选) 你甚至可以回滚到上一步 best_state，但这比较耗内存，通常清空动量就够了
                if it % 5 == 0:
                    pass
                    #print(f"Iter {it:4d}: Resid={current_resid:.2e} (RESET MOMENTUM) | Alpha={alpha:.2e}")
            else:
                # 残差在下降或持平，说明路走对了
                # 1. 稍微加速，奖励一下
                alpha = min(alpha * 1.01, alpha_max)
                if it % 5 == 0:
                    pass
                    #print(f"Iter {it:4d}: Resid={current_resid:.2e} | Alpha={alpha:.2e}")

            # Keep track of best state
            if current_resid < best_resid:
                best_resid = current_resid
                best_state = (u_G.copy(), v_G.copy())

            prev_resid = current_resid

            # --- Check Convergence ---
            if current_resid < tol:
                #print(f"Converged at iter {it}: Residual={current_resid:.2e}")
                break

            # --- Momentum Step Update ---
            # v_new = momentum * v_old + alpha * force
            vel_u = momentum * vel_u + alpha * resid_u
            vel_v = momentum * vel_v + alpha * resid_v

            u_G += vel_u
            v_G += vel_v

        # End of Loop
        # 如果最后没有收敛，建议使用过程中遇到的最小残差状态
        if current_resid > best_resid:
            #print("Warning: Using best historic state instead of final state.")
            u_G, v_G = best_state

        # Store final state
        self.u_final_G = u_G
        self.v_final_G = v_G
        self.u_final_real = np.dot(self.exp_iGr, u_G).real
        self.v_final_real = np.dot(self.exp_iGr, v_G).real

        return self.calculate_energy(u_G, v_G, s_shift)

    from scipy.optimize import minimize

    def solve_lbfgs(self, s_shift=np.zeros(2)):
        """
        L-BFGS-B with Step Counter.
        """
        print(">>> Starting L-BFGS-B Optimization...")

        # 1. Init
        np.random.seed(int(time.time()))
        random_amp = 0.0 * self.a_nm
        u_G = (np.random.randn(self.N_G, 2) + 1j * np.random.randn(self.N_G, 2)) * random_amp
        v_G = (np.random.randn(self.N_G, 2) + 1j * np.random.randn(self.N_G, 2)) * random_amp
        u_G.astype(np.complex64)
        v_G.astype(np.complex64)

        x0 = np.concatenate([u_G.real.flatten(), u_G.imag.flatten(),
                             v_G.real.flatten(), v_G.imag.flatten()])

        # --- 初始化计数器和计时器 ---
        self._iter_count = 0  # <--- 新增：计数器归零
        self._last_E = 0.0
        self._last_grad_norm = 0.0
        self._start_time = time.time()

        # 2. Objective Function
        def objective(x):
            N = self.N_G
            u_r = x[0:2 * N].reshape(N, 2);
            u_i = x[2 * N:4 * N].reshape(N, 2)
            v_r = x[4 * N:6 * N].reshape(N, 2);
            v_i = x[6 * N:8 * N].reshape(N, 2)
            u_curr = u_r + 1j * u_i
            v_curr = v_r + 1j * v_i

            # --- CRITICAL FIX: Force Real Space Fields to be Real ---
            # 这一步防止虚部进入 cos 函数导致指数爆炸
            u_real = np.dot(self.exp_iGr, u_curr).real
            v_real = np.dot(self.exp_iGr, v_curr).real
            N_pts = self.grid_density ** 2

            # C. Forces
            Force_u_acc = np.zeros((self.N_G, 2), dtype=complex)
            Force_v_acc = np.zeros((self.N_G, 2), dtype=complex)

            U_pot_sum = 0

            for j in range(3):
                G12, G23 = self.G_vecs_12[j], self.G_vecs_23[j]
                b_vec = self.b_vecs_intrinsic[j]
                shift_phase = np.dot(b_vec, s_shift)

                # Arguments are guaranteed REAL now
                arg1 = np.dot(self.R_real, G12) - np.dot(u_real + v_real, b_vec) * 0.5 + shift_phase
                arg2 = np.dot(self.R_real, G23) + np.dot(u_real - v_real, b_vec) * 0.5

                # Energy (Real)
                U_pot_sum += 2 * self.V0 * np.sum(np.cos(arg1) + np.cos(arg2))

                # Force
                sin_1 = np.sin(arg1)
                sin_2 = np.sin(arg2)

                f12_G = np.dot(sin_1, self.exp_minus_iGr) / N_pts
                f23_G = np.dot(sin_2, self.exp_minus_iGr) / N_pts
                K_inv_b = np.einsum('nij,j->ni', self.K_inv, b_vec)

                if self.use_modified_signs:
                    term_u = (f12_G - f23_G)[:, None] * K_inv_b
                    term_v = (f12_G + f23_G)[:, None] * K_inv_b
                else:
                    term_u = (f12_G + f23_G)[:, None] * K_inv_b
                    term_v = (f12_G - f23_G)[:, None] * K_inv_b

                Force_u_acc += term_u
                Force_v_acc += term_v

            # Elastic Energy
            def get_k_term(vec_G):
                # Ensure this is calculated as real
                term = np.einsum('ni,nij,nj->n', np.conj(vec_G), self.K_mat, vec_G)
                return 0.5 * np.sum(np.real(term))

            s1_G = u_curr / 6 + v_curr / 2
            s2_G = -u_curr / 3
            s3_G = u_curr / 6 - v_curr / 2
            E_el = get_k_term(s1_G) + get_k_term(s2_G) + get_k_term(s3_G)

            E_pot = U_pot_sum / N_pts
            E_total = E_el + E_pot

            # Gradient
            u_target = -6 * self.V0 * Force_u_acc
            v_target = -2 * self.V0 * Force_v_acc

            resid_u = u_curr - u_target
            resid_v = v_curr - v_target

            grad_u = np.einsum('nij,nj->ni', self.K_mat, resid_u)
            grad_v = np.einsum('nij,nj->ni', self.K_mat, resid_v)

            grad_flat = np.concatenate([grad_u.real.flatten(), grad_u.imag.flatten(),
                                        grad_v.real.flatten(), grad_v.imag.flatten()])

            # Casting explicitly to float to avoid ComplexWarning
            E_total_real = float(np.real(E_total))

            self._last_E = E_total_real
            self._last_grad_norm = np.max(np.abs(grad_flat))

            return E_total_real, grad_flat

        # 3. Callback
        def print_progress(xk):
            self._iter_count += 1  # <--- 新增：每次回调时计数器 +1

            elapsed = time.time() - self._start_time
            # 打印 Step 数
            if self._iter_count % 30 == 0:
                pass
                #print(f"   >> Step {self._iter_count:3d}: E = {self._last_E:.6f} | Max Force = {self._last_grad_norm:.1e} | Time: {elapsed:.1f}s")
            else:
                pass

        # 4. Run
        # 记得把 disp 关掉，因为我们现在自己打印了
        print("doping iteration")
        res = minimize(objective, x0, method='L-BFGS-B', jac=True,
                       callback=print_progress,
                       options={'ftol': 1e-11, 'gtol': 1e-5, 'maxiter': 5000})

        # ... (后续解包代码不变) ...
        x_final = res.x
        N = self.N_G
        u_final = x_final[0:2 * N].reshape(N, 2) + 1j * x_final[2 * N:4 * N].reshape(N, 2)
        v_final = x_final[4 * N:6 * N].reshape(N, 2) + 1j * x_final[6 * N:8 * N].reshape(N, 2)

        self.u_final_real = np.dot(self.exp_iGr, u_final).real
        self.v_final_real = np.dot(self.exp_iGr, v_final).real
        self.u_final_G = u_final
        self.v_final_G = v_final
        u_g = []
        for i in self.harmonics:
            u_g.append(np.linalg.norm(self.u_final_G[i]))
        #print("Now printing the u_g vectors")
        #print(u_g)

        return u_g

    def calculate_energy(self, u_G, v_G, s_shift):
        """计算总能量密度 (eV/nm^2)"""
        # 1. 弹性势能 (Fourier Space)
        s1_G = u_G / 6.0 + v_G / 2.0
        s2_G = -u_G / 3.0
        s3_G = u_G / 6.0 - v_G / 2.0

        def get_k_term(vec_G):
            term = np.einsum('ni,nij,nj->n', np.conj(vec_G), self.K_mat, vec_G)
            return 0.5 * np.sum(np.real(term))

        E_el = get_k_term(s1_G) + get_k_term(s2_G) + get_k_term(s3_G)

        # 2. 层间势能 (Real Space Integral)
        u_r = np.dot(self.exp_iGr, u_G).real
        v_r = np.dot(self.exp_iGr, v_G).real

        U_pot_sum = 0
        for j in range(3):
            b = self.b_vecs_intrinsic[j]
            G12, G23 = self.G_vecs_12[j], self.G_vecs_23[j]
            # 关键：能量计算必须包含 s_shift 导致的相位
            shift_phase = np.dot(b, s_shift)

            arg1 = np.dot(self.R_real, G12) - np.dot(u_r + v_r, b) * 0.5 + shift_phase
            arg2 = np.dot(self.R_real, G23) + np.dot(u_r - v_r, b) * 0.5

            U_pot_sum += 2 * self.V0 * (np.cos(arg1) + np.cos(arg2))

        E_pot = np.mean(U_pot_sum)
        return E_el + E_pot

    def save_results(self, filename=None):
        """
        Save the G-space coefficients and system parameters to a compact .npz file.
        Automatically generates filename based on parameters if not provided.
        """
        # 1. 自动构建文件名 (如果未指定)
        if filename is None:
            # 将弧度转换为角度用于文件名
            t12_deg = self.theta12 * 180 / np.pi
            t23_deg = self.theta23 * 180 / np.pi

            # 基础文件名: 角度_n_m
            fname_base = f"{t12_deg:.2f}_{t23_deg:.2f}_n{self.n}_m{self.m}"

            # 如果有应变，加入文件名
            if hasattr(self, 'strain_eps') and abs(self.strain_eps) > 1e-9:
                strain_pct = self.strain_eps * 100
                fname_base += f"_s{strain_pct:.1f}"

            filename = f"{fname_base}_LBFGS.npz"

        #print(f">>> Saving solver state to {filename}...")

        # 2. 保存数据 (np.savez_compressed 生成压缩文件，体积很小)
        np.savez_compressed(
            filename,
            # 核心解
            u_G=self.u_final_G,
            v_G=self.v_final_G,

            # 必须保存 G 向量，因为解是基于这些向量的
            G_vecs=self.G_vecs,

            # 几何参数
            L1_SM=self.L1_SM,
            L2_SM=self.L2_SM,

            # 物理/元数据 (保存下来备查)
            # params: [theta12, theta23, n, m, factor, a_nm, V0]
            params=np.array([self.theta12, self.theta23, self.n, self.m,
                             self.supercell_factor, self.a_nm, self.V0]),

            # 保存应变信息 (如果存在)
            strain_info=np.array([getattr(self, 'strain_eps', 0.0),
                                  getattr(self, 'strain_theta', 0.0)])
        )
        #print("    Save complete.")

    def scan_and_optimize(self,best_shift ):

        # --- Final Relaxation ---
        #print("\n>>> Running Final High-Precision Relaxation...")
        #final_E = self.solve(s_shift=best_shift, quick_run=False)
        u_g = self.solve_lbfgs(s_shift=best_shift)
        if hasattr(self, 'exp_iGr'): del self.exp_iGr
        if hasattr(self, 'exp_minus_iGr'): del self.exp_minus_iGr
        if hasattr(self, 'K_mat'): del self.K_mat
        if hasattr(self, 'K_inv'): del self.K_inv
        gc.collect()
        return u_g
        #print(f"Final Energy: {final_E:.6f} eV/nm^2")
        
        # --- Outputs ---
        #self.visualize_layers()
        #self.analyze_binding_energy(best_shift)
if __name__ == "__main__":
    solver = TTGPaperSolver()
    solver.setup_system(3.252870296, 1.470129726, 7, 4,3,2)


    # Run the full optimization loop
    a1 = np.array([solver.a_nm, 0.0])
    a2 = np.array([solver.a_nm * 0.5, solver.a_nm * np.sqrt(3) / 2])

    # B 原子位置 (相对于 A 的位移)
    #vec_AB = (a1 + a2) / 3.0
    best_shift = (a1 + a2) / 3.0
    solver.scan_and_optimize(best_shift)
    solver.save_results()