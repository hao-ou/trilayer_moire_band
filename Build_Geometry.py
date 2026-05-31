import numpy as np
import time
from scipy.optimize import minimize
import gc
import sys

class TTGPaperSolver:

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


    def _rotate_vec(self, vec, theta_rad):
        c, s = np.cos(theta_rad), np.sin(theta_rad)
        return np.array([c*vec[0] - s*vec[1], s*vec[0] + c*vec[1]])
    def theta_func(self,n, m, np_i, mp_i):
        # Numerator: sqrt(3) * { m(2n' + m') - (2n + m)m' }
        term1 = m * (2 * np_i + mp_i)
        term2 = (2 * n + m) * mp_i
        num = np.sqrt(3) * (term1 - term2)

        # Denominator: (2n+m)(2n'+m') + 3mm' + (2n'+m')^2 + 3m'^2
        den1 = (2 * n + m) * (2 * np_i + mp_i)
        den2 = 3 * m * mp_i
        den3 = (2 * np_i + mp_i) ** 2 + 3 * mp_i ** 2
        den = den1 + den2 + den3

        if den == 0: return 0.0
        return 2 * np.arctan(num / den)
    def check_angle(self):
        self.theta12_calc =  self.theta_func(self.n, self.m, self.n_p, self.m_p)
        self.theta23_calc = -self.theta_func(self.n_p, self.m_p, self.n, self.m)
        check_12 = (np.abs(self.theta12-self.theta12_calc))<1e-3
        check_23 = ((self.theta23 - self.theta23_calc))<1e-3
        return check_12, check_23



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
        check_12, check_23 = self.check_angle()
        if (check_12 ==0) or (check_23 == 0):
            print("Error! the calculated twist angles are not matching the input angles!")
            sys.exit(1)
        self._build_geometry()

    def _build_geometry(self):
        self.supercell_factor = 1
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

        self.G1_12_rebuilt = self.n * self.G1_SM - self.m * self.G2_SM
        self.G2_12_rebuilt = self.m * self.G1_SM + (self.n + self.m) * self.G2_SM
        self.G1_23_rebuilt = self.n_p * self.G1_SM - self.m_p * self.G2_SM
        self.G2_23_rebuilt = self.m_p * self.G1_SM + (self.n_p + self.m_p) * self.G2_SM


        self.period_len = np.linalg.norm(self.L1_SM) # This is the full box size
        #print(f"    Calculation Box Size L = {self.period_len:.2f} nm (Factor={self.supercell_factor})")

    
    def _get_reciprocal(self, a1, a2):
        cross_z = a1[0]*a2[1] - a1[1]*a2[0]
        b1 = (2 * np.pi / cross_z) * np.array([a2[1], -a2[0]])
        b2 = (2 * np.pi / cross_z) * np.array([-a1[1], a1[0]])
        return b1, b2

    def plot_reciprocal_space(self):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        from scipy.spatial import Voronoi

        def get_wigner_seitz_vertices(g1, g2):
            points = [i * g1 + j * g2 for i in range(-2, 3) for j in range(-2, 3)]
            vor = Voronoi(points)
            origin_idx = np.argmin(np.linalg.norm(points, axis=1))
            vertices = vor.vertices[vor.regions[vor.point_region[origin_idx]]]
            angles = np.arctan2(vertices[:, 1], vertices[:, 0])
            return vertices[np.argsort(angles)]

        fig, ax = plt.subplots(figsize=(10, 10), dpi=150)

        # 1. 获取单层倒格矢
        b1_l1 = self._rotate_vec(self.b1_0, -self.theta12)
        b2_l1 = self._rotate_vec(self.b2_0, -self.theta12)
        b1_l2 = self.b1_0
        b2_l2 = self.b2_0
        b1_l3 = self._rotate_vec(self.b1_0, self.theta23)
        b2_l3 = self._rotate_vec(self.b2_0, self.theta23)

        # 2. 计算全局 K 点坐标 (以全局 Gamma (0,0) 为原点)
        K1 = (2 * b1_l1 + b2_l1) / 3.0
        K2 = (2 * b1_l2 + b2_l2) / 3.0
        K3 = (2 * b1_l3 + b2_l3) / 3.0

        # 3. 寻找 Moiré BZ 的正确中心：确立 K_a 和 K_b 为相邻顶点 (构成一条边)
        def find_moire_center(K_a, K_b, G1, G2):
            verts_0 = get_wigner_seitz_vertices(G1, G2)
            E = K_a - K_b
            M = (K_a + K_b) / 2.0
            N = np.array([-E[1], E[0]]) # 法向量
            
            # 正六边形中心到边的垂直距离为 边长 * sqrt(3)/2
            center1 = M + N * (np.sqrt(3) / 2.0)
            center2 = M - N * (np.sqrt(3) / 2.0)
            
            def check_center(c):
                shifted = verts_0 + c
                dist_a = np.min(np.linalg.norm(shifted - K_a, axis=1))
                dist_b = np.min(np.linalg.norm(shifted - K_b, axis=1))
                return dist_a < 1e-4 and dist_b < 1e-4
                
            if check_center(center1): return center1
            if check_center(center2): return center2
            return M

        center_m12 = find_moire_center(K1, K2, self.G1_12, self.G2_12)
        center_m23 = find_moire_center(K2, K3, self.G1_23, self.G2_23)

        verts_m12 = get_wigner_seitz_vertices(self.G1_12, self.G2_12) + center_m12
        verts_m23 = get_wigner_seitz_vertices(self.G1_23, self.G2_23) + center_m23

        # 4. 铺设 SM 网格：从全局 Gamma (0,0) 开始计算，只在 K 谷附近进行绘制
        verts_sm_base = get_wigner_seitz_vertices(self.G1_SM, self.G2_SM)
        
        # 寻找距离 K2 最近的 SM 倒格点作为网格绘制参考中心
        inv_G_SM = np.linalg.inv(np.column_stack((self.G1_SM, self.G2_SM)))
        ij_approx = np.dot(inv_G_SM, K2)
        i0, j0 = int(np.round(ij_approx[0])), int(np.round(ij_approx[1]))

        grid_range = 8
        for i in range(i0 - grid_range, i0 + grid_range + 1):
            for j in range(j0 - grid_range, j0 + grid_range + 1):
                shift = i * self.G1_SM + j * self.G2_SM
                shifted_verts = verts_sm_base + shift
                ax.add_patch(Polygon(shifted_verts, closed=True, fill=False, edgecolor='lightgray', linewidth=0.5))
                ax.scatter(shift[0], shift[1], color='gray', s=5, alpha=0.6, zorder=2)

        # 5. 画 Moiré 12 和 Moiré 23 的 BZ
        ax.add_patch(Polygon(verts_m12, closed=True, fill=False, edgecolor='blue', linewidth=1.5, zorder=4))
        ax.add_patch(Polygon(verts_m23, closed=True, fill=False, edgecolor='red', linewidth=1.5, zorder=4))

        # 6. 绘制宏观的单层 BZ 边界 (虚线)
        verts_l1 = get_wigner_seitz_vertices(b1_l1, b2_l1)
        verts_l2 = get_wigner_seitz_vertices(b1_l2, b2_l2)
        verts_l3 = get_wigner_seitz_vertices(b1_l3, b2_l3)
        ax.add_patch(Polygon(verts_l1, closed=True, fill=False, edgecolor='blue', linewidth=0.5, linestyle='--', alpha=0.5))
        ax.add_patch(Polygon(verts_l2, closed=True, fill=False, edgecolor='black', linewidth=0.8, linestyle='--', alpha=0.5))
        ax.add_patch(Polygon(verts_l3, closed=True, fill=False, edgecolor='red', linewidth=0.5, linestyle='--', alpha=0.5))

        # 7. 标出狄拉克点 (K points)
        ax.scatter(K1[0], K1[1], color='blue', marker='o', s=80, zorder=5, label='$K^{(1)}$')
        ax.scatter(K2[0], K2[1], color='black', marker='*', s=150, zorder=6, label='$K^{(2)}$')
        ax.scatter(K3[0], K3[1], color='red', marker='^', s=80, zorder=5, label='$K^{(3)}$')

        # 8. 视角聚焦于 K2 附近
        q_max = max(np.linalg.norm(self.G1_12), np.linalg.norm(self.G1_23))
        ax.set_xlim(K2[0] - q_max * 2.5, K2[0] + q_max * 2.5)
        ax.set_ylim(K2[1] - q_max * 2.5, K2[1] + q_max * 2.5)

        ax.set_aspect('equal')
        ax.set_xlabel('$k_x$ (nm$^{-1}$)', fontsize=12)
        ax.set_ylabel('$k_y$ (nm$^{-1}$)', fontsize=12)
        ax.set_title('Absolute BZ Alignment at K Valley', fontsize=14)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    solver = TTGPaperSolver()
    solver.setup_system(-2.645908381, -1.575126206, 7, 3,4,2)
    solver.plot_reciprocal_space()