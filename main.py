import numpy as np
from casadi import *
from finger import Finger
import numpy as np

d2r = np.pi / 180
r2d = 180 / np.pi
eps = 1e-6

T = 2
dt = 0.01
N = int(T / dt)
t = np.linspace(0, T, N)

circle_center = DM([-0.6920, 0.8380 + 0.15 / 2])
finger_center = DM([0, 0.95])
robot = Finger(circle_center, finger_center)

solver = Opti()

# solvermization variables
xc = solver.variable(3, N)
vc = solver.variable(3, N)

q = solver.variable(robot.dof, N)
dq = solver.variable(robot.dof, N)
us = solver.variable(robot.dof, N)

lam = solver.variable(6, N)
gammas = solver.variable(2, N)

dx_finger = solver.variable(1, N)
dis = solver.variable(1, N)

x2_var = solver.variable(2, N)
x3_var = solver.variable(2, N)

# Initial values
q_init = DM([2.6690, 0.2618, 0.3770])
dq_init = DM.zeros(3, 1)
gammas_init = DM.zeros(2, 1)
xc_init = DM([circle_center[0], circle_center[1], 0])
vc_init = DM.zeros(3, 1)
dx_finger_init = DM([0])
lam_init = DM([0, 0, 0, 0, 0, -robot.circle["mass"] * robot.gravity])

# Initial constraints
solver.subject_to(xc[:, 0] == xc_init)
solver.subject_to(vc[:, 0] == vc_init)
solver.subject_to(us[:, 0] == vc_init)
solver.subject_to(q[:, 0] == q_init)
solver.subject_to(dq[:, 0] == dq_init)
solver.subject_to(gammas[:, 0] == gammas_init)
solver.subject_to(lam[:, 0] == lam_init)
solver.subject_to(dx_finger[:, 0] == dx_finger_init)

# Terminal goal
dgoal = 0.05
x_goal = DM([circle_center[0] + dgoal, 0.913])
joint_limit_min = DM([1.571, 0, -0.262])
joint_limit_max = DM([1.571 * 2, 1.571, 1.571])
c_state_max = 5

# Cost matrices
Q = diag(DM([1, 1]))
Qt = 100 * diag(DM([1, 1]))
R = 0.01 * diag(DM([1, 1, 1]))
Rq = 0.01 * diag(DM([1, 1, 1]))
Qd = 100

cost = MX(0)
# Constraints and cost loop
for i in range(N - 1):
    solver.subject_to(us[:, i] <= inf)
    solver.subject_to(us[:, i] >= -inf)
    solver.subject_to(q[:, i] <= joint_limit_max)
    solver.subject_to(q[:, i] >= joint_limit_min)
    solver.subject_to(dq[:, i] <= 5)
    solver.subject_to(dq[:, i] >= -5)
    solver.subject_to(xc[0:2, i] <= c_state_max)
    solver.subject_to(xc[0:2, i] >= -c_state_max)
    solver.subject_to(dx_finger[:, i] <= 0.11)
    solver.subject_to(dx_finger[:, i] >= 0)

    # Kinematics & Dynamics
    x, dx, a, da, normal_vec_c, d, x_close, x_act = robot.kinematics(
        q[:, i], dq[:, i], xc[:, i], vc[:, i], dx_finger[:, i]
    )
    H_1, Hc_1, G_1, phi1_1, phi2_1, J_ee_1, lam_w1, lam_cf1, lam_b1, psi1_1, psi2_1 = (
        robot.dynamics(
            xc[:, i], vc[:, i], q[:, i], dq[:, i], us[:, i], lam[:, i], dx_finger[:, i]
        )
    )
    H_2, Hc_2, G_2, phi1_2, phi2_2, J_ee_2, lam_w2, lam_cf2, lam_b2, psi1_2, psi2_2 = (
        robot.dynamics(
            xc[:, i + 1],
            vc[:, i + 1],
            q[:, i + 1],
            dq[:, i + 1],
            us[:, i + 1],
            lam[:, i + 1],
            dx_finger[:, i + 1],
        )
    )
    solver.subject_to(xc[0, i] < 0)
    solver.subject_to(x[0, 2] < 0)
    solver.subject_to(xc[0, i] - x[0, 2] >= 0)
    solver.subject_to(d - robot.circle["radius"] >= 0)

    solver.subject_to(q[:, i] - q[:, i + 1] + robot.dt * dq[:, i + 1] == 0)
    solver.subject_to(
        H_2 @ (dq[:, i + 1] - dq[:, i])
        + robot.dt * (-J_ee_2.T @ lam_w2[0:2] - us[:, i + 1])
        == 0
    )
    solver.subject_to(xc[:, i] - xc[:, i + 1] + robot.dt * vc[:, i + 1] == 0)

    tau_b2 = (xc[1, i] - x[1, 2]) * lam_b2[0]
    lam_b2 = vertcat(lam_b2[0], lam_b2[1], tau_b2)
    tau_w2 = -(x[0, 2] - xc[0, i]) * lam_w2[1] + (x[1, 2] - xc[1, i]) * lam_w2[0]
    lam_w2 = vertcat(lam_w2[0], lam_w2[1], tau_w2)
    solver.subject_to(
        Hc_2 @ (vc[:, i + 1] - vc[:, i]) + robot.dt * (-G_2 - lam_b2 - lam_w2) == 0
    )

    solver.subject_to(
        (xc[0, i + 1] - xc[0, i]) + robot.circle["radius"] * vc[2, i + 1] * robot.dt
        <= eps
    )
    solver.subject_to(
        (xc[0, i + 1] - xc[0, i]) + robot.circle["radius"] * vc[2, i + 1] * robot.dt
        >= -eps
    )

    lam_x1p, lam_x1m, lam_z1 = lam[0, i], lam[1, i], lam[2, i]
    lam_xcp, lam_xcm, lam_zc = lam[3, i], lam[4, i], lam[5, i]

    solver.subject_to(phi1_1 >= 0)
    solver.subject_to([lam_z1 >= 0, lam_x1p >= 0, lam_x1m >= 0, gammas[0, i] >= 0])
    solver.subject_to(robot.mu * lam_z1 - lam_x1p - lam_x1m >= 0)

    solver.subject_to(phi2_1 >= 0)
    solver.subject_to([lam_zc >= 0, lam_xcp >= 0, lam_xcm >= 0, gammas[1, i] >= 0])
    solver.subject_to(robot.mu * lam_zc - lam_xcp - lam_xcm >= 0)

    solver.subject_to(gammas[0, i] + psi1_1 >= 0)
    solver.subject_to(gammas[0, i] - psi1_1 >= 0)
    solver.subject_to(gammas[1, i] + psi2_1 >= 0)
    solver.subject_to(gammas[1, i] - psi2_1 >= 0)

    solver.subject_to(phi1_1 * lam_z1 <= eps)
    solver.subject_to(phi1_1 * lam_z1 >= -eps)
    solver.subject_to(phi2_1 * lam_zc <= eps)
    solver.subject_to(phi2_1 * lam_zc >= -eps)

    solver.subject_to((robot.mu * lam_z1 - lam_x1p - lam_x1m) * gammas[0, i] <= eps)
    solver.subject_to((robot.mu * lam_z1 - lam_x1p - lam_x1m) * gammas[0, i] >= -eps)
    solver.subject_to((robot.mu * lam_zc - lam_xcp - lam_xcm) * gammas[1, i] <= eps)
    solver.subject_to((robot.mu * lam_zc - lam_xcp - lam_xcm) * gammas[1, i] >= -eps)

    solver.subject_to((gammas[0, i] + psi1_1[0]) * lam_x1p <= eps)
    solver.subject_to((gammas[0, i] + psi1_1[0]) * lam_x1p >= -eps)
    solver.subject_to((gammas[0, i] - psi1_1[0]) * lam_x1m <= eps)
    solver.subject_to((gammas[0, i] - psi1_1[0]) * lam_x1m >= -eps)

    solver.subject_to((gammas[1, i] + psi2_1[0]) * lam_xcp <= eps)
    solver.subject_to((gammas[1, i] + psi2_1[0]) * lam_xcp >= -eps)
    solver.subject_to((gammas[1, i] - psi2_1[0]) * lam_xcm <= eps)
    solver.subject_to((gammas[1, i] - psi2_1[0]) * lam_xcm >= -eps)

    solver.subject_to(dx_finger[:, i + 1] >= 0)
    solver.subject_to(dx_finger[:, i] >= 0)
    solver.subject_to(
        (dx_finger[:, i] - dx_finger[:, i + 1])
        - robot.circle["radius"] * vc[2, i + 1] * robot.dt
        == 0
    )
    solver.subject_to(phi1_1 * (dx_finger[:, i + 1] - dx_finger[:, i]) <= eps)
    solver.subject_to(phi1_1 * (dx_finger[:, i + 1] - dx_finger[:, i]) >= -eps)

    x2_var[:, i] = x[:, 1]
    x3_var[:, i] = x[:, 2]
    dis[:, i] = phi1_1
    # Cost
    if i < 150:
        cost += (x_goal - xc[0:2, i]).T @ Q @ (x_goal - xc[0:2, i]) + dis[
            :, i
        ].T @ Qd @ dis[:, i]
    else:
        cost += (x_goal - xc[0:2, i]).T @ Qt @ (x_goal - xc[0:2, i]) + dq[
            :, i
        ].T @ Rq @ dq[:, i]

# Set cost and solver
solver.minimize(cost)
p_opts = {"expand": True}
s_opts = {
    "max_iter": 3000,
    "tol": 1e-4,
    "acceptable_tol": 1e-5,
    "constr_viol_tol": 1e-5,
    "acceptable_iter": 5,
    "nlp_scaling_method": "none",
    "warm_start_init_point": "yes",
}
solver.solver("ipopt", p_opts, s_opts)

# Solve
try:
    sol = solver.solve_limited()
except RuntimeError:
    sol = solver.debug

# Extract solution
qsol = sol.value(q)
dqsol = sol.value(dq)
usol = sol.value(us)
lam_result = sol.value(lam)
xcsol = sol.value(xc)
vcsol = sol.value(vc)
gamma_sol = sol.value(gammas)
dx_finger_sol = sol.value(dx_finger)

sol_data = {
    "qsol": qsol,
    "dqsol": dqsol,
    "usol": usol,
    "lam": lam_result,
    "xcsol": xcsol,
    "vcsol": vcsol,
    "gamma_sol": gamma_sol,
    "dx_finger_sol": dx_finger_sol,
    "x_goal": np.array(x_goal.full()).flatten(),
    "circle_center": np.array(circle_center.full()).flatten(),
}
np.savez("finger_solution.npz", **sol_data)

print("Solution saved to finger_solution.npz")

data = np.load("finger_solution.npz", allow_pickle=True)

qsol = data["qsol"]
xcsol = data["xcsol"]
dx_finger_sol = data["dx_finger_sol"]
x_goal = data["x_goal"]
circle_center = data["circle_center"]
