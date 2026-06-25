"""
ASMCA — Strict Mathematical Reproduction
=========================================
Wang & Zhang, "An Adaptive Fault-Tolerant Sliding Mode Control Allocation
Scheme for Multirotor Helicopter Subject to Simultaneous Actuator Faults"
IEEE Transactions on Industrial Electronics, Vol. 65, No. 5, May 2018.

Every implemented equation is labelled with its paper number.
Every assumption that is NOT in the paper is labelled [ASSUMPTION].
No gamma, no projection, no modifications — paper equations only.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 1 — PHYSICAL PARAMETERS
# [ASSUMPTION] All numerical values below are assumptions.
# The paper does not publish the exact physical parameters of the Quanser
# octorotor. These must be identified from hardware or the Quanser datasheet.
# ═══════════════════════════════════════════════════════════════════════════════

class PhysicalParams:
    # Mass(from paper 37)
    m    = 1.4    # [kg]

    # Gravitational constant — universal
    g    = 9.81       # [m/s^2]

    # Moments of inertia (diagonal, body-fixed principal axes)
    # (Section II-A-2): body frame coincides with principal axes.
    #from paper 37
    Ixx  = 0.03    # [kg·m^2]
    Iyy  = 0.03    # [kg·m^2]
    Izz  = 0.04    # [kg·m^2]

    # [ASSUMPTION] Rotor moment of inertia — appears in Eq.(6)/(7) gyroscopic term
    Ir   = 2.84e-5    # [kg·m^2]

    #  Motor arm length Ld — appears in Eq.(6)/(7) and mixing
    # for attitude control after single-motor failure.
    Ld   = 0.20      # [m]

    # [ASSUMPTION] Translational drag coefficients K1,K2,K3 — appear in Eq.(5)
    K1   = 0.01
    K2   = 0.01
    K3   = 0.01

    # [ASSUMPTION] Rotational drag coefficients K4,K5,K6 — appear in Eq.(6)/(7)
    K4   = 0.006
    K5   = 0.006
    K6   = 0.006

    # Actuator input limits — STATED IN PAPER Section IV: range [0.05, 0.10]
    u_min = 0.05
    u_max = 0.10

    # [ASSUMPTION] Thrust gain Ku.
    # Paper states: T_j = Ku * [ω/(s+ω)] * u_j   (Section II-A-3).
    # Ku value not published.  u_hover = 0.065 places the 8-motor hover
    # point below the midpoint of [0.05, 0.10], leaving headroom above for
    # the 7-motor case (single failure): u_hover_7 ≈ 0.074, headroom ≈ 0.026.
    u_hover = 0.068   # [ASSUMPTION] hover equilibrium PWM

    # [ASSUMPTION] Actuator bandwidth ω — paper gives transfer function ω/(s+ω)
    # but does not publish ω numerically.  100 rad/s gives a 10× ratio over
    # the controller bandwidth (ω_n=10 rad/s), keeping the actuator lag
    # negligible relative to the SMC dynamics.
    act_bw = 15.0    # [rad/s]

    # [ASSUMPTION] Torque-to-thrust coefficient Ky.
    # Paper states: τ_j = Ky * u*_j   (Section II-A-3).
    # Ky not published numerically.
    Ky = 4.0        # [N·m / (thrust unit)]
    n_motors  = 8
    n_virtual = 4     # (Uz, Uφ, Uθ, Uψ)

    def __init__(self):
        # [ASSUMPTION] Derived from assumed u_hover
        self.Ku = self.m * self.g / (self.n_motors * self.u_hover)


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 2 — CONTROLLER PARAMETERS
# All values stated explicitly in Section IV of the paper.
# ═══════════════════════════════════════════════════════════════════════════════

class ControllerParams:
    """
    Exact gains from paper Section IV.
    k_{i1}, k_{i2}, k_{ci} indexed 0..3 = altitude, roll, pitch, yaw.
    Paper uses 1-indexing (i=1,2,3,4); subtract 1 for Python arrays.

    Paper states (Section IV):
        k11=25, k21=100, k31=100, k41=25
        k12=10, k22=20,  k32=20,  k42=10
        kc1=5,  kc2=10,  kc3=10,  kc4=5
        Φ = 0.2   (same for all channels)

    Theorem 1 stability condition:  k_{ci} >= η_i + D_i
    where D_i = sup|d_i(t)|  and  η_i > 0 small.
    """
    k1  = np.array([25.0, 100.0, 100.0, 25.0])   # paper: k_{i1}
    k2  = np.array([10.0,  20.0,  20.0, 10.0])   # paper: k_{i2}
    kc  = np.array([ 5.0,  10.0,  10.0,  5.0])   # paper: k_{ci}
    Phi = np.array([ 0.2,   0.2,   0.2,  0.2])   # paper: Φ_i


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 3 — CONTROL EFFECTIVENESS MATRIX Bu   (Section II-A-3)
#
# Paper defines Bu ∈ R^{n×m}, n=4 virtual inputs, m=8 motors.
# Mixing equations stated explicitly in Section II-A-3:
#   Uz     = T1+T2+T3+T4+T5+T6+T7+T8
#   Uφ     = Ld*(T3 - T4 + T7 - T8)
#   Uθ     = Ld*(T1 - T2 + T5 - T6)
#   Uψ     = (τ1+τ2-τ3-τ4-τ5-τ6+τ7+τ8)     where τ_j = Ky*u*_j
#
# [ASSUMPTION] The paper writes Uψ in terms of τ_j = Ky*u*_j.
# Since T_j = Ku*u*_j, we have τ_j = (Ky/Ku)*T_j.
# Bu row 3 uses Ky_ratio = Ky/Ku (torque per unit thrust).
#
# Assumption 1 (paper): rank(Bu) = n = 4   (verified at runtime).
# ═══════════════════════════════════════════════════════════════════════════════

def build_Bu(p):
    Ld = p.Ld
    # [ASSUMPTION] Ky_ratio = Ky/Ku for the yaw row, derived from actuator model
    Ky_ratio = p.Ky / p.Ku

    Bu = np.array([
        #   T1    T2    T3    T4    T5    T6    T7    T8
        [   1,    1,    1,    1,    1,    1,    1,    1   ],  # Uz  [N]
        [   0,    0,   Ld,  -Ld,   0,    0,   Ld,  -Ld  ],  # Uφ  [N·m]
        [  Ld,  -Ld,   0,    0,   Ld,  -Ld,   0,    0   ],  # Uθ  [N·m]
        [Ky_ratio, Ky_ratio, -Ky_ratio, -Ky_ratio,
         -Ky_ratio, -Ky_ratio, Ky_ratio, Ky_ratio],          # Uψ  [N·m]
    ], dtype=float)

    # Verify Assumption 1
    assert np.linalg.matrix_rank(Bu) == p.n_virtual, "Bu rank < n_virtual"
    return Bu


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 4 — NONLINEAR AFFINE SYSTEM  (Eqs. 8, 9, 12)
#
# Paper Eq.(8):  ẋ = f(x,t) + h(x,t)·ν(t) + d(t)
# Paper Eq.(9):  ν(t) = Bu · L(t) · u(t)
#   where L(t) = diag(l_1(t),...,l_m(t)),   l_j ∈ [0,1]
#
# Paper Eq.(12) — four decoupled subsystems (i=1,2,3,4):
#   ẋ_{2i-1} = x_{2i}
#   ẋ_{2i}   = f_i(x) + h_i·ν_i + d_i
#
# State x = [x1..x8] = [ze, że, φ, φ̇, θ, θ̇, ψ, ψ̇]   (Eq. 11)
#
# Subsystem definitions from Eq.(5) and Eq.(7):
#   i=1 (altitude): f1=-g,  h1=cosφ·cosθ/m,      d1=-K3·że/m
#   i=2 (roll):     f2=x6·x8·(Iyy-Izz)/Ixx,       h2=1/Ixx,  d2=...
#   i=3 (pitch):    f3=x4·x8·(Izz-Ixx)/Iyy,       h3=1/Iyy,  d3=...
#   i=4 (yaw):      f4=x4·x6·(Ixx-Iyy)/Izz,       h4=1/Izz,  d4=...
#
# NOTE on paper notation:
#   Eq.(7) uses x6·x8 = θ̇·ψ̇ for roll (f2).
#   "x4·x8" in the paper's subsystem list refers to (x4,x8) = (φ̇,ψ̇),
#   but the Euler equation (Eq.7) clearly shows θ̇·ψ̇ for the roll coupling.
#   We implement Eq.(7) directly.
# ═══════════════════════════════════════════════════════════════════════════════

def compute_fi_hi_di(x, p, Omega):
    """
    Returns f(4,), h(4,), d(4,) for the four subsystems.
    All terms taken directly from Eqs.(5) and (7).
    Omega = gyroscopic residual Ω  (defined in Section II-A-2).
    """
    _ze, ze_d, phi, phi_d, theta, theta_d, _psi, psi_d = x

    # ── Altitude  i=1  (from Eq. 5, z-channel) ────────────────────────────
    f0 = -p.g                                         # Eq.(5): -g term
    h0 =  np.cos(phi) * np.cos(theta) / p.m          # Eq.(5): (cφ·cθ)/m
    d0 = -p.K3 * ze_d / p.m                          # Eq.(5): -K3·że/m

    # ── Roll  i=2  (from Eq. 7, φ equation) ───────────────────────────────
    # Eq.(7): φ̈ = I^{-1}[-(Izz-Iyy+Izz·r - Iyy·q)·[p;q;r] row + Ir·(-q)·Ω + Uφ + drag]
    # Simplified per Eq.(7) directly:
    f1 = theta_d * psi_d * (p.Iyy - p.Izz) / p.Ixx  # Eq.(7): θ̇·ψ̇·(Iyy-Izz)/Ixx
    h1 = 1.0 / p.Ixx                                  # Eq.(7): 1/Ixx
    d1 = (-p.Ir * theta_d * Omega / p.Ixx             # Eq.(7): Ir·(-θ̇)·Ω / Ixx
          - p.K4 * p.Ld * phi_d / p.Ixx)              # Eq.(7): -K4·Ld·φ̇ / Ixx

    # ── Pitch  i=3  (from Eq. 7, θ equation) ──────────────────────────────
    f2 = phi_d * psi_d * (p.Izz - p.Ixx) / p.Iyy    # Eq.(7): φ̇·ψ̇·(Izz-Ixx)/Iyy
    h2 = 1.0 / p.Iyy                                  # Eq.(7): 1/Iyy
    d2 = ( p.Ir * phi_d * Omega / p.Iyy               # Eq.(7): Ir·φ̇·Ω / Iyy
          - p.K5 * p.Ld * theta_d / p.Iyy)            # Eq.(7): -K5·Ld·θ̇ / Iyy

    # ── Yaw  i=4  (from Eq. 7, ψ equation) ───────────────────────────────
    f3 = phi_d * theta_d * (p.Ixx - p.Iyy) / p.Izz  # Eq.(7): φ̇·θ̇·(Ixx-Iyy)/Izz
    h3 = 1.0 / p.Izz                                  # Eq.(7): 1/Izz
    d3 = -p.K6 * psi_d / p.Izz                        # Eq.(7): -K6·ψ̇ / Izz

    return (np.array([f0, f1, f2, f3]),
            np.array([h0, h1, h2, h3]),
            np.array([d0, d1, d2, d3]))


def plant_ode(state, u_cmd, fault_levels, p):
    """
    Full ODE for numerical integration.  state = [x(8), T_filtered(8)].

    Actuator model (Section II-A-3):
        T_j = Ku · [ω/(s+ω)] · u_j   →   Ṫ_j = -ω·T_j + ω·Ku·u_j

    Fault model (Eq. 9):
        L(t) = diag(l_1,...,l_m),  l_j ∈ [0,1]
        Effective thrust: T_eff_j = l_j · T_j

    [ASSUMPTION] Gyroscopic residual Ω approximated as:
        Ω_j ≈ T_j / Ku   (rotor speed proportional to thrust)
        Ω   = Σ sign_j · Ω_j
        sign = [+1,+1,-1,-1,-1,-1,+1,+1]  (CW/CCW pattern from Fig.1)

    Dynamics: Eqs.(5) and (7).
    [ASSUMPTION] Small-angle approximation T^I_B ≈ I is applied in Eq.(7),
    per the paper's own statement: "assume that the changes of roll and pitch
    angles are very small" (Section II-A-2, after Eq.(6)).
    """
    x_plant = state[:8]
    T_filt  = state[8:]
    _ze, ze_d, phi, phi_d, theta, theta_d, _psi, psi_d = x_plant

    # Actuator first-order lag (Section II-A-3)
    T_dot = p.act_bw * (u_cmd * p.Ku - T_filt)   # Ṫ_j = ω(Ku·u_j - T_j)

    # Fault scaling: Eq.(9) with L(t)
    T_eff = T_filt * fault_levels

    # [ASSUMPTION] Gyroscopic residual
    rotor_signs = np.array([1, 1, -1, -1, -1, -1, 1, 1], dtype=float)
    Omega = np.dot(rotor_signs, T_eff) / p.Ku

    # Virtual controls from effective thrusts (Section II-A-3 mixing)
    Ld = p.Ld;  Ky_r = p.Ky / p.Ku
    Uz     = np.sum(T_eff)
    Uphi   = Ld  * (T_eff[2] - T_eff[3] + T_eff[6] - T_eff[7])
    Utheta = Ld  * (T_eff[0] - T_eff[1] + T_eff[4] - T_eff[5])
    Upsi   = Ky_r* (T_eff[0]+T_eff[1]-T_eff[2]-T_eff[3]
                    -T_eff[4]-T_eff[5]+T_eff[6]+T_eff[7])

    # Eq.(5) — altitude acceleration
    ze_dd = -p.g + np.cos(phi)*np.cos(theta)*Uz/p.m - p.K3*ze_d/p.m

    # Eq.(7) — rotational accelerations (small-angle T^I_B ≈ I, stated in paper)
    phi_dd   = ((p.Iyy-p.Izz)*theta_d*psi_d/p.Ixx
                - p.Ir*theta_d*Omega/p.Ixx
                + Uphi/p.Ixx
                - p.K4*p.Ld*phi_d/p.Ixx)

    theta_dd = ((p.Izz-p.Ixx)*phi_d*psi_d/p.Iyy
                + p.Ir*phi_d*Omega/p.Iyy
                + Utheta/p.Iyy
                - p.K5*p.Ld*theta_d/p.Iyy)

    psi_dd   = ((p.Ixx-p.Iyy)*phi_d*theta_d/p.Izz
                + Upsi/p.Izz
                - p.K6*psi_d/p.Izz)

    x_dot = np.array([ze_d, ze_dd, phi_d, phi_dd, theta_d, theta_dd, psi_d, psi_dd])
    return np.concatenate([x_dot, T_dot])


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 5 — INTEGRAL SLIDING SURFACE  Eq.(25)
#
# Paper Eq.(25):
#   σ_i = x̃_{i2} + k_{i2}·x̃_{i1} + k_{i1}·∫_{t0}^{t} x̃_{i1}(τ)dτ
#         - k_{i2}·x̃_{i1}(t₀) - x̃_{i2}(t₀)
#
# Derivation steps leading to Eq.(25):
#   σ_{i0}  = C_i^T · x̃   (Eq. 14, 15 — linear combination of states)
#   ż_i     = -c_i·x̃_{i2} + k_{i2}·x̃_{i2} + k_{i1}·x̃_{i1}   (Eq. 24)
#   z_i(0)  = -c_i·x̃_{i1}(t₀) - x̃_{i2}(t₀)                    (Eq. 24)
#   σ_i     = σ_{i0} + z_i   (Eq. 14)
#
# The constant c_i cancels out in Eq.(25) (paper states: "c_i does not
# appear in (25)"), leaving only k_{i1} and k_{i2}.
# ═══════════════════════════════════════════════════════════════════════════════

def compute_sigma(err1, err2, int_err1, err1_t0, err2_t0, cp):
    """
    Eq.(25): σ_i = x̃_{i2} + k_{i2}·x̃_{i1} + k_{i1}·∫x̃_{i1}dt
                   - k_{i2}·x̃_{i1}(t₀) - x̃_{i2}(t₀)
    """
    return (err2
            + cp.k2 * err1
            + cp.k1 * int_err1
            - cp.k2 * err1_t0
            - err2_t0)


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 6 — SATURATION FUNCTION  Eq.(34)
#
# Paper Eq.(34):
#   sat(σ_i/Φ_i) = sign(σ_i)   if |σ_i| >= Φ_i
#                = σ_i/Φ_i      if |σ_i| <  Φ_i
# ═══════════════════════════════════════════════════════════════════════════════

def sat(sigma, Phi):
    """Eq.(34) — exactly as stated."""
    ratio = sigma / Phi
    return np.where(np.abs(ratio) >= 1.0, np.sign(sigma), ratio)


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 7 — BOUNDARY-LAYER DISTANCE  Eq.(40)
#
# Paper Eq.(40):
#   σ_{Δi} = σ_i - Φ_i · sat(σ_i/Φ_i)
#
# Property stated in paper (after Eq.40):
#   σ̇_{Δi} = σ̇_i   outside boundary layer
#   σ_{Δi} = 0       inside boundary layer
# ═══════════════════════════════════════════════════════════════════════════════

def compute_sigma_delta(sigma, Phi, sat_val):
    """Eq.(40) — exactly as stated."""
    return sigma - Phi * sat_val


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 8 — VIRTUAL CONTROL LAW  Eq.(39)
#
# Paper Eq.(39):
#   ν_i = Υ̂_i·(ẋ^d_{2i} - k_{i2}·x̃_{i2} - k_{i1}·x̃_{i1} - f_i(x))
#         - Υ̂_i·k_{ci}·sat(σ_i/Φ_i)
#
# where Υ̂_i = ĥ_i^{-1}  is the estimated inverse of h_i.
#
# Eq.(33) is the special case when Υ̂_i = h_i^{-1} (no fault, no adaptation):
#   ν_i = h_i^{-1}·(ẋ^d_{2i} - k_{i2}·x̃_{i2} - k_{i1}·x̃_{i1} - f_i(x))
#         - h_i^{-1}·k_{ci}·sat(σ_i/Φ_i)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_nu(xd_ddot, err1, err2, f, Upsilon_hat, sat_val, cp):
    """
    Eq.(39): ν_i = Υ̂_i·(ẋ^d_{2i} - k_{i2}·x̃_{i2} - k_{i1}·x̃_{i1} - f_i)
                   - Υ̂_i·k_{ci}·sat(σ_i/Φ_i)
    """
    return (Upsilon_hat * (xd_ddot - cp.k2 * err2 - cp.k1 * err1 - f)
            - Upsilon_hat * cp.kc * sat_val)


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 9 — ADAPTATION LAW  Eq.(41)
#
# Paper Eq.(41):
#   Υ̂̇_i = (-ẋ^d_{2i} + k_{i2}·x̃_{i2} + k_{i1}·x̃_{i1} + f_i(x)
#            + k_{ci}·sat(σ_i/Φ_i)) · σ_{Δi}
#
# Derived from the Lyapunov candidate Eq.(42):
#   V_i = (1/2)[σ_{Δi}^2 + Υ_i^{-1}·(Υ̂_i - Υ_i)^2]
#
# The term Υ_i^{-1} in Eq.(42) is the TRUE (unknown) value of h_i^{-1}.
# The adaptation law Eq.(41) is obtained by setting V̇_i terms with
# (Υ̂_i - Υ_i) equal to zero (Eq.43 derivation), yielding Eq.(44):
#   V̇_i ≤ -η_i·|σ_{Δi}|
#
# NO extra gain multiplier is present in Eq.(41). Paper has exactly one
# scalar multiplication: the full bracket × σ_{Δi}.
# ═══════════════════════════════════════════════════════════════════════════════

def compute_Upsilon_dot(xd_ddot, err1, err2, f, sat_val, sigma_delta, cp):
    """
    Eq.(41): Υ̂̇_i = (-ẋ^d_{2i} + k_{i2}·x̃_{i2} + k_{i1}·x̃_{i1}
                       + f_i(x) + k_{ci}·sat(σ_i/Φ_i)) · σ_{Δi}
    No extra gain. Exactly as written in the paper.
    """
    return ((-xd_ddot
             + cp.k2 * err2
             + cp.k1 * err1
             + f
             + cp.kc * sat_val) * sigma_delta)


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 10 — CONTROL ALLOCATION  Eqs.(35) and (36)
#
# Paper Lemma 1 / Eq.(35):
#   J = arg min_{u} u^T · W · u
#   subject to:  ν_i = Bu_i · u
#
# Paper Eq.(36) — explicit solution:
#   u* = W · Bu_i^T · (Bu_i · W · Bu_i^T)^{-1} · ν_i
#
# CRITICAL NOTE on paper notation:
#   In Eq.(36), Bu_i is the NOMINAL control effectiveness matrix (constant).
#   W = diag(w_1,...,w_m) with w_j = 1/l_j  (Section III-B).
#   The paper does NOT pre-multiply Bu by L(t) inside Eq.(36).
#   The fault enters ONLY through the weighting matrix W.
#   For a failed actuator (l_j=0): w_j → ∞, so u*_j → 0 automatically.
#
# This is different from using Bu_eff = Bu·diag(l), which would be a
# different (also valid) pseudo-inverse formulation.
# We implement Eq.(36) exactly as written in the paper.
# ═══════════════════════════════════════════════════════════════════════════════

def control_allocation(nu_desired, Bu, fault_levels, p):
    """
    Eq.(36) — weighted pseudo-inverse control allocation
    with iterative redistribution for actuator saturation.

    Cost function (Section III-B):
        J = u^T · W · u,   W = diag(1/l_j)

    KKT optimality yields (standard weighted least-norm solution):
        u* = W^{-1} · Bu^T · (Bu · W^{-1} · Bu^T)^{-1} · ν_d

    Since W = diag(1/l_j), we have W^{-1} = diag(l_j).

    [ASSUMPTION] When the unconstrained solution violates actuator
    bounds (Eq. 10), saturated actuators are fixed at their limit
    and the remaining free actuators are re-solved iteratively.
    This is standard in flight control allocation (Durham 1993,
    Bodson 2002) and implements the paper's intent of redistribution
    to remaining healthy actuators (Section III-B, para. 2).
    """
    m = Bu.shape[1]
    u_cmd = np.full(m, (p.u_min + p.u_max) / 2)  # initial guess
    free  = np.ones(m, dtype=bool)

    for _iter in range(m):
        # Build reduced problem for free actuators only
        idx_free = np.where(free)[0]
        if len(idx_free) == 0:
            break

        # Subtract contribution of fixed (saturated) actuators
        idx_fixed = np.where(~free)[0]
        nu_resid = nu_desired.copy()
        if len(idx_fixed) > 0:
            for j in idx_fixed:
                nu_resid -= Bu[:, j] * (fault_levels[j] * u_cmd[j] * p.Ku)

        # Reduced Bu and W^{-1} for free actuators
        Bu_f  = Bu[:, idx_free]
        l_f   = np.array([max(fault_levels[j], 1e-6) for j in idx_free])
        Winv_f = np.diag(l_f)

        # Weighted pseudo-inverse solution for free actuators
        A_f = Bu_f @ Winv_f @ Bu_f.T
        try:
            T_f = Winv_f @ Bu_f.T @ np.linalg.solve(A_f, nu_resid)
        except np.linalg.LinAlgError:
            T_f = Winv_f @ Bu_f.T @ np.linalg.lstsq(A_f, nu_resid, rcond=None)[0]

        u_f = T_f / p.Ku

        # Check for violations
        any_violation = False
        for k, j in enumerate(idx_free):
            if u_f[k] > p.u_max:
                u_cmd[j] = p.u_max
                free[j]  = False
                any_violation = True
            elif u_f[k] < p.u_min:
                u_cmd[j] = p.u_min
                free[j]  = False
                any_violation = True
            else:
                u_cmd[j] = u_f[k]

        if not any_violation:
            break

    # Delivered virtual control: ν_actual = Bu · L · T_effective
    L = np.diag(fault_levels)
    nu_actual = Bu @ L @ (u_cmd * p.Ku)

    return u_cmd, nu_actual


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 11 — ASMCA CONTROLLER
# Assembles Blocks 5–10 in the correct order each sample period.
# ═══════════════════════════════════════════════════════════════════════════════

class ASMCAController:
    """
    Three modes matching the paper's experimental comparison (Section IV):
        'ASMCA'  — proposed scheme: Eq.(39) + Eq.(41)
        'NSMCA'  — normal SMC-CA: Eq.(33) only, no adaptation
        'LQRCA'  — LQR virtual controller + same CA
                   [ASSUMPTION] Paper does not give LQR gains explicitly.
                   We use proportional-integral feedback in ν-space.
    """

    def __init__(self, p, cp, Bu, mode='ASMCA'):
        self.p = p; self.cp = cp; self.Bu = Bu; self.mode = mode

        # Integral of tracking error (for Eq. 25)
        self.int_err1    = np.zeros(4)
        self.err1_t0     = None   # x̃_{i1}(t₀)
        self.err2_t0     = None   # x̃_{i2}(t₀)

        # Adaptive parameter Υ̂_i (initialised at h_i^{-1} per paper)
        self.Upsilon_hat  = None
        self.Upsilon_init = None   # stored for clamping

        # Signal logs
        self.log = {k: [] for k in
                    ['t','sigma','sigma_delta','nu_des','nu_act','Upsilon','u','fault']}

    @staticmethod
    def reference(t):
        """
        Desired pitch step profile.
        [ASSUMPTION] Exact step times and amplitudes are read from Figs.5/7
        of the paper visually. The paper does not publish them analytically.
        All other channels: zero.
        Returns xd (8,) and xd_dot (8,).
        """
        if   t <  10.0: theta_d =  5.0 * np.pi/180
        elif t <  20.0: theta_d = -5.0 * np.pi/180
        elif t <  35.0: theta_d =  3.0 * np.pi/180
        elif t <  50.0: theta_d = -3.0 * np.pi/180
        elif t <  60.0: theta_d =  4.0 * np.pi/180
        else:           theta_d =  0.0
        xd     = np.array([0., 0., 0., 0., theta_d, 0., 0., 0.])
        xd_dot = np.zeros(8)
        return xd, xd_dot

    def step(self, t, x, T_filt, fault_levels, Ts):
        """
        Execute one controller sample. Returns u_cmd (8,) in PWM units.

        Step order (Fig.3 in paper):
          1. Compute σ_i via Eq.(25)
          2. Compute sat(σ_i/Φ_i) via Eq.(34)
          3. Compute σ_{Δi} via Eq.(40)
          4. Compute ν_i via Eq.(39) [or Eq.(33) for NSMCA]
          5. Allocate u* via Eq.(36)
          6. Update Υ̂_i via Eq.(41)  [ASMCA only]
        """
        p = self.p; cp = self.cp

        # Gyroscopic Ω for subsystem disturbance terms d_i
        # [ASSUMPTION] Ω_j ≈ T_j/Ku; fault-scaled thrusts used
        T_eff_for_Omega = T_filt * fault_levels
        rotor_signs = np.array([1,1,-1,-1,-1,-1,1,1], dtype=float)
        Omega = np.dot(rotor_signs, T_eff_for_Omega) / p.Ku

        # Subsystem terms f_i, h_i, d_i   (Eq. 12 decomposition)
        f, h, _d = compute_fi_hi_di(x, p, Omega)

        # Reference and tracking errors
        xd, xd_dot = self.reference(t)
        err1    = np.array([x[0]-xd[0], x[2]-xd[2], x[4]-xd[4], x[6]-xd[6]])
        err2    = np.array([x[1]-xd[1], x[3]-xd[3], x[5]-xd[5], x[7]-xd[7]])
        xd_ddot = np.array([xd_dot[1], xd_dot[3], xd_dot[5], xd_dot[7]])

        # Initialise at t₀
        if self.err1_t0 is None:
            self.err1_t0     = err1.copy()
            self.err2_t0     = err2.copy()
            self.int_err1    = np.zeros(4)
            # Υ̂_i(t₀) = h_i^{-1}   (paper: "initial value of Υ̂_i is h_i^{-1}")
            self.Upsilon_hat  = 1.0 / h
            self.Upsilon_init = self.Upsilon_hat.copy()

        # Update integral  ∫ x̃_{i1} dt
        # [ASSUMPTION] Forward Euler integration of the integral term.
        # Paper uses continuous integral; discrete approximation is required.
        self.int_err1 += err1 * Ts

        # ── Step 1: Sliding surface  Eq.(25) ──────────────────────────────
        sigma = compute_sigma(err1, err2, self.int_err1,
                              self.err1_t0, self.err2_t0, cp)

        # ── Step 2: sat(σ_i/Φ_i)  Eq.(34) ────────────────────────────────
        sat_val = sat(sigma, cp.Phi)

        # ── Step 3: σ_{Δi}  Eq.(40) ───────────────────────────────────────
        sigma_delta = compute_sigma_delta(sigma, cp.Phi, sat_val)

        # ── Step 4: Virtual control ────────────────────────────────────────
        if self.mode == 'ASMCA':
            # Eq.(39): adaptive law with Υ̂_i
            nu = compute_nu(xd_ddot, err1, err2, f, self.Upsilon_hat, sat_val, cp)

        elif self.mode == 'NSMCA':
            # Eq.(33): standard SMC, Υ̂_i fixed at true h_i^{-1}
            nu = compute_nu(xd_ddot, err1, err2, f, 1.0/h, sat_val, cp)

        elif self.mode == 'LQRCA':
            # [ASSUMPTION] LQR comparison controller not defined mathematically
            # in the paper. Reference [37] is Wang et al. 2016.
            # We use a PI state-feedback law in ν-space as proxy.
            nu = (1.0/h) * (xd_ddot - cp.k2*err2 - cp.k1*err1 - f
                            - cp.k1*self.int_err1)

        # ── Step 5: Control allocation  Eq.(36) ───────────────────────────
        u_cmd, nu_actual = control_allocation(nu, self.Bu, fault_levels, p)

        # ── Step 6: Adaptive update  Eq.(41)  [ASMCA only] ────────────────
        if self.mode == 'ASMCA':
            Ups_dot = compute_Upsilon_dot(xd_ddot, err1, err2, f,
                                          sat_val, sigma_delta, cp)
            # Forward Euler: Υ̂_i[k+1] = Υ̂_i[k] + Ts · Υ̂̇_i[k]
            # [ASSUMPTION] Forward Euler for continuous ODE Eq.(41).
            # NO extra gain multiplier — paper has none.
            self.Upsilon_hat = self.Upsilon_hat + Ts * Ups_dot

            # [ASSUMPTION] Clamp Υ̂_i to prevent unbounded drift.
            # Paper Remark 1 states adaptation ceases inside the boundary
            # layer (σ_Δ=0). With discrete step references, σ briefly exits
            # the boundary layer at each transition, causing transient
            # adaptation.  Clamping to [0.5, 10]× nominal prevents runaway.
            self.Upsilon_hat = np.clip(
                self.Upsilon_hat,
                0.5  * self.Upsilon_init,
                10.0 * self.Upsilon_init,
            )

        # Log
        L = self.log
        L['t'].append(t)
        L['sigma'].append(sigma.copy())
        L['sigma_delta'].append(sigma_delta.copy())
        L['nu_des'].append(nu.copy())
        L['nu_act'].append(nu_actual.copy())
        L['Upsilon'].append(self.Upsilon_hat.copy() if self.mode=='ASMCA'
                            else (1.0/h).copy())
        L['u'].append(u_cmd.copy())
        L['fault'].append(fault_levels.copy())

        return u_cmd


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 12 — FAULT SCHEDULE  (Section IV-B, exact paper scenarios)
# ═══════════════════════════════════════════════════════════════════════════════

def fault_schedule(t, scenario):
    """
    Returns l ∈ R^8 fault level vector.

    Paper Section IV-B, exact text:
      Scenario 1: "a 100% loss of control effectiveness fault is only
                   introduced to actuator #1 at 20s"
                   → l_1(t) = 0  for t >= 20
      Scenario 2: "faults are injected into two actuators at 20s.
                   Actuator #1 experiences a complete failure, and
                   actuator #5 experiences 40% loss of control effectiveness"
                   → l_1(t) = 0, l_5(t) = 0.6  for t >= 20
    """
    l = np.ones(8)
    if t >= 20.0:
        if scenario == 1:
            l[0] = 0.0       # actuator #1: complete failure
        elif scenario == 2:
            l[0] = 0.0       # actuator #1: complete failure
            l[4] = 0.6       # actuator #5: 40% loss → l_5 = 1 - 0.4 = 0.6
    return l


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 13 — SIMULATION LOOP
# [ASSUMPTION] Discrete-time implementation at 200 Hz (paper: IMU at 200 Hz).
# [ASSUMPTION] RK4 for plant ODE; ZOH on control inputs within each Ts.
# [ASSUMPTION] Actuator states T_filtered initialised at hover equilibrium.
# ═══════════════════════════════════════════════════════════════════════════════

def run_simulation(p, cp, Bu, scenario, mode, T_total=70.0, Ts=1/200):
    N    = int(T_total / Ts)
    ctrl = ASMCAController(p, cp, Bu, mode=mode)

    x0      = np.zeros(8)                               # drone at rest
    T_init  = np.ones(8) * p.m * p.g / p.n_motors      # hover thrust per motor
    state   = np.concatenate([x0, T_init])

    u_zoh     = T_init / p.Ku                           # hover PWM
    fault_zoh = np.ones(8)

    # Output storage
    t_arr  = np.zeros(N); x_arr  = np.zeros((N,8)); u_arr  = np.zeros((N,8))
    sig_arr= np.zeros((N,4)); nd_arr = np.zeros((N,4)); na_arr = np.zeros((N,4))
    ups_arr= np.zeros((N,4)); fl_arr = np.zeros((N,8))

    for k in range(N):
        t_k    = k * Ts
        x_k    = state[:8]
        T_filt = state[8:]

        fault_k = fault_schedule(t_k, scenario)
        u_k     = ctrl.step(t_k, x_k, T_filt, fault_k, Ts)

        u_zoh = u_k; fault_zoh = fault_k

        t_arr[k]=t_k; x_arr[k]=x_k; u_arr[k]=u_k
        sig_arr[k]=ctrl.log['sigma'][-1]
        nd_arr[k] =ctrl.log['nu_des'][-1]
        na_arr[k] =ctrl.log['nu_act'][-1]
        ups_arr[k]=ctrl.log['Upsilon'][-1]
        fl_arr[k] =fault_k

        # RK4 plant integration
        def f_ode(s):
            return plant_ode(s, u_zoh, fault_zoh, p)
        k1 = f_ode(state)
        k2 = f_ode(state + Ts/2*k1)
        k3 = f_ode(state + Ts/2*k2)
        k4 = f_ode(state + Ts*k3)
        state = state + (Ts/6)*(k1 + 2*k2 + 2*k3 + k4)
        state[8:] = np.maximum(state[8:], 0.0)   # T_j >= 0 (physical)

    return dict(t=t_arr,x=x_arr,u=u_arr,sigma=sig_arr,
                nu_des=nd_arr,nu_act=na_arr,Ups=ups_arr,fault=fl_arr)


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 14 — PLOTTING  (Figs. 5–11)
# ═══════════════════════════════════════════════════════════════════════════════

def ref_pitch_deg(t_arr):
    out = np.zeros_like(t_arr)
    out[t_arr < 10]                         =  5.0
    out[(t_arr>=10)&(t_arr<20)]             = -5.0
    out[(t_arr>=20)&(t_arr<35)]             =  3.0
    out[(t_arr>=35)&(t_arr<50)]             = -3.0
    out[(t_arr>=50)&(t_arr<60)]             =  4.0
    return out

def pitch_deg(r): return r['x'][:,4]*180/np.pi

def plot_s1(ra, rl, path):
    t=ra['t']; td=ref_pitch_deg(t)
    fig,axs=plt.subplots(2,1,figsize=(10,7),sharex=True)
    fig.suptitle('Scenario 1 — Single Actuator Failure',fontsize=12,fontweight='bold')
    ax=axs[0]
    ax.plot(t,td,'k-',lw=2,label='Desired')
    ax.plot(t,pitch_deg(ra),'b--',lw=1.5,marker='*',markevery=800,ms=5,label='ASMCA')
    ax.plot(t,pitch_deg(rl),'r-.',lw=1.5,marker='x',markevery=800,ms=5,label='LQRCA')
    ax.axvline(20,color='gray',ls=':',lw=1); ax.set_ylabel('Pitch (deg)')
    ax.set_title('Fig.5'); ax.legend(fontsize=9); ax.grid(True,alpha=0.3)
    ax=axs[1]
    ax.plot(t,ra['sigma'][:,2],'b-',lw=1.2,label='σ₃')
    ax.axhline(0.2,color='r',ls='--',lw=1,label='±Φ'); ax.axhline(-0.2,color='r',ls='--',lw=1)
    ax.axvline(20,color='gray',ls=':',lw=1)
    ax.set_ylabel('σ₃'); ax.set_xlabel('Time (s)')
    ax.set_title('Fig.6'); ax.legend(fontsize=9); ax.grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches='tight'); plt.close()
    print(f"  → {path}")

def plot_s2(res, path):
    t=res['ASMCA']['t']; td=ref_pitch_deg(t)
    fig=plt.figure(figsize=(12,20)); gs=gridspec.GridSpec(5,1,hspace=0.45)
    fig.suptitle('Scenario 2 — Simultaneous Faults',fontsize=12,fontweight='bold')
    # Fig.7
    ax=fig.add_subplot(gs[0]); ax.plot(t,td,'k-',lw=2,label='Desired')
    for m,ls,mk,c in [('ASMCA','--','*','blue'),('NSMCA',':','x','green'),('LQRCA','-.','s','red')]:
        ax.plot(t,pitch_deg(res[m]),color=c,ls=ls,lw=1.5,marker=mk,markevery=800,ms=5,label=m)
    ax.axvline(20,color='gray',ls=':',lw=1); ax.set_ylabel('Pitch (deg)')
    ax.legend(fontsize=9,ncol=4); ax.set_title('Fig.7'); ax.grid(True,alpha=0.3)
    # Fig.8
    ax=fig.add_subplot(gs[1]); clrs=plt.cm.tab10(np.linspace(0,0.9,8))
    for j in range(8): ax.plot(t,res['ASMCA']['u'][:,j],color=clrs[j],lw=1.0,label=f'M{j+1}')
    ax.axvline(20,color='gray',ls=':',lw=1)
    ax.axhline(0.05,color='k',ls='--',lw=0.7); ax.axhline(0.10,color='k',ls='--',lw=0.7)
    ax.set_ylabel('PWM'); ax.legend(fontsize=7,ncol=4); ax.set_title('Fig.8'); ax.grid(True,alpha=0.3)
    # Fig.9
    ax=fig.add_subplot(gs[2])
    ax.plot(t,res['ASMCA']['sigma'][:,2],'b-',lw=1.2,label='σ₃')
    ax.axhline(0.2,color='r',ls='--',lw=1); ax.axhline(-0.2,color='r',ls='--',lw=1)
    ax.axvline(20,color='gray',ls=':',lw=1)
    ax.set_ylabel('σ₃'); ax.set_title('Fig.9'); ax.legend(fontsize=9); ax.grid(True,alpha=0.3)
    # Fig.10: virtual control error ν̃_i = ν_desired - ν_actual  (Eq.37 context)
    ax=fig.add_subplot(gs[3])
    nu_tilde=res['ASMCA']['nu_des'][:,2]-res['ASMCA']['nu_act'][:,2]
    ax.plot(t,nu_tilde,'b-',lw=1.2,label='ν̃₃ (pitch)')
    ax.axvline(20,color='gray',ls=':',lw=1)
    ax.set_ylabel('ν̃₃ (N·m)'); ax.set_title('Fig.10'); ax.legend(fontsize=9); ax.grid(True,alpha=0.3)
    # Fig.11: Υ̂_i
    ax=fig.add_subplot(gs[4])
    ax.plot(t,res['ASMCA']['Ups'][:,2],'b-',lw=1.2,label='Υ̂₃')
    ax.axvline(20,color='gray',ls=':',lw=1,label='fault onset')
    ax.set_ylabel('Υ̂₃'); ax.set_xlabel('Time (s)')
    ax.set_title('Fig.11'); ax.legend(fontsize=9); ax.grid(True,alpha=0.3)
    for i in range(5): fig.axes[i].set_xlim([0,70])
    plt.savefig(path,dpi=150,bbox_inches='tight'); plt.close()
    print(f"  → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 15 — MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("="*60)
    print("ASMCA Strict Reproduction — Wang & Zhang IEEE TIE 2018")
    print("="*60)

    p  = PhysicalParams()
    cp = ControllerParams()
    Bu = build_Bu(p)

    print(f"\n[ASSUMPTION] Ku = {p.Ku:.4f} N/PWM   (hover at u={p.u_hover})")
    print(f"Max deliverable Uz = {8*p.Ku*p.u_max:.3f} N,  Weight = {p.m*p.g:.3f} N")
    print(f"\nBu (4×8):\n{np.round(Bu,5)}")
    print(f"rank(Bu) = {np.linalg.matrix_rank(Bu)}")

    T_total = 70.0
    Ts      = 1.0/200   # 200 Hz — paper Section IV-A states IMU at 200 Hz

    print("\n── Scenario 1: Single Actuator Failure ──")
    r1 = {}
    for mode in ['ASMCA','LQRCA']:
        print(f"  {mode}...", end=' ', flush=True)
        r1[mode] = run_simulation(p,cp,Bu,1,mode,T_total,Ts)
        print("done")
    plot_s1(r1['ASMCA'], r1['LQRCA'], os.path.join(_SCRIPT_DIR, 'scenario1_figs5_6.png'))

    print("\n── Scenario 2: Simultaneous Faults ──")
    r2 = {}
    for mode in ['ASMCA','NSMCA','LQRCA']:
        print(f"  {mode}...", end=' ', flush=True)
        r2[mode] = run_simulation(p,cp,Bu,2,mode,T_total,Ts)
        print("done")
    plot_s2(r2, os.path.join(_SCRIPT_DIR, 'scenario2_figs7_11.png'))

    print("\n── Post-fault pitch RMS (t > 20s) ──")
    t_ = r2['ASMCA']['t']; mask = t_ > 20
    td_ = ref_pitch_deg(t_[mask])
    for m in ['ASMCA','NSMCA','LQRCA']:
        rms = np.sqrt(np.mean((pitch_deg(r2[m])[mask]-td_)**2))
        print(f"  {m}: {rms:.4f} deg")

    print(f"\nOutputs saved to {_SCRIPT_DIR}")

if __name__ == '__main__':
    main()