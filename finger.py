from casadi import (
    SX,
    vertcat,
    Function,
    cos,
    sin,
    sqrt,
    atan2,
    dot,
    DM,
    jacobian,
    jtimes,
)


class Finger:
    def __init__(self, circle_center, finger_center):
        self.n_contact = 2
        self.dof = 3
        self.dims = 2
        self.arm = {
            "mass": DM([1, 1, 1]),
            "length": DM([0.45, 0.25, 0.26]),
            "origin": finger_center,
        }
        self.circle = {"mass": 0.11, "center": circle_center, "radius": 0.075}
        self.gravity = -9.81
        self.mu = 0.1
        self.dt = 0.01

        self.set_physics()

    def set_physics(self):
        # symbolic variables for finger
        q1, q2, q3 = SX.sym("q1"), SX.sym("q2"), SX.sym("q3")
        dq1, dq2, dq3 = SX.sym("dq1"), SX.sym("dq2"), SX.sym("dq3")
        u1, u2, u3 = SX.sym("u1"), SX.sym("u2"), SX.sym("u3")
        # for circle
        xc, yc, thc = SX.sym("xc"), SX.sym("yc"), SX.sym("thc")
        vxc, vyc, wc = SX.sym("vxc"), SX.sym("vyc"), SX.sym("wc")

        lam = SX.sym("lambda", 6, 1)
        dx_finger = SX.sym("dx_finger")  # only for continous rolling

        q = vertcat(q1, q2, q3)
        dq = vertcat(dq1, dq2, dq3)
        u = vertcat(u1, u2, u3)
        XC = vertcat(xc, yc, thc)
        VC = vertcat(vxc, vyc, wc)

        # forward kinematics
        origin = self.arm["origin"]
        l1 = self.arm["length"][0]
        l2 = self.arm["length"][1]
        l3 = self.arm["length"][2]

        x = SX.zeros(2, self.dof)
        x[0, 0] = origin[0] + l1 * cos(q1)
        x[1, 0] = origin[1] + l1 * sin(q1)
        x[0, 1] = x[0, 0] + l2 * cos(q1 + q2)
        x[1, 1] = x[1, 0] + l2 * sin(q1 + q2)
        x[0, 2] = x[0, 1] + (0.15 + dx_finger) * cos(q1 + q2 + q3)
        x[1, 2] = x[1, 1] + (0.15 + dx_finger) * sin(q1 + q2 + q3)

        x3_act = x[0, 1] + l3 * cos(q1 + q2 + q3)
        y3_act = x[1, 1] + l3 * sin(q1 + q2 + q3)
        x_act = vertcat(x3_act, y3_act)

        # angular pose
        a = SX.zeros(3, self.dof)
        temp = DM.eye(self.dof)
        for i in range(self.dof):
            for j in range(i):
                temp[i, j] = 1
        a[2, :] = temp @ q

        # Jacobians
        J_ee_func = Function("J_ee", [q, dx_finger], [jacobian(x[:, 2], q)])
        J_ee = J_ee_func(q, dx_finger)
        dx = jtimes(x, q, dq)
        da = jtimes(a, q, dq)

        # inertia matrix H finger
        H = SX.zeros(self.dof, self.dof)
        for i in range(self.dof):
            J_l = jacobian(x[:, i], q)
            J_a = jacobian(a[:, i], q)
            I = SX.zeros(3, 3)
            I[2, 2] = (1 / 12) * self.arm["mass"][i] * self.arm["length"][i] ** 2
            H += self.arm["mass"][i] * (J_l.T @ J_l) + (J_a.T @ I @ J_a)

        # circle inertia (Disc)
        H_c = SX.zeros(3, 3)
        H_c[0, 0] = self.circle["mass"]
        H_c[1, 1] = self.circle["mass"]
        H_c[2, 2] = 0.5 * self.circle["mass"] * self.circle["radius"] ** 2
        G_c = vertcat(0, self.circle["mass"] * self.gravity, 0)

        # contact modeling
        x2, y2 = x[0, 1], x[1, 1]
        diff_x = x[0, 2] - xc
        diff_y = x[1, 2] - yc
        theta_contact = atan2(diff_y, diff_x)

        r_c2w = SX(2, 2)
        r_c2w[0, 0] = cos(theta_contact)
        r_c2w[0, 1] = sin(theta_contact)
        r_c2w[1, 0] = -sin(theta_contact)
        r_c2w[1, 1] = cos(theta_contact)

        lam_c = SX.zeros(3, 1)
        lam_c[0] = lam[0] - lam[1]
        lam_c[1] = lam[2]
        lam_c[2] = 0

        lam_w = r_c2w @ lam_c[0:2]
        lam_w = vertcat(lam_w, 0)

        lam_b = SX.zeros(2, 1)
        lam_b[0] = lam[3] - lam[4]
        lam_b[1] = lam[5]

        phi1 = sqrt(diff_x**2 + diff_y**2) - self.circle["radius"]
        phi2 = yc - self.circle["radius"] - 10 * 0.08380

        normal_vec_c = r_c2w @ SX.ones(2, 1)

        ee_vel_b = J_ee @ dq
        r_arm2w = SX(2, 2)
        r_arm2w[0, 0] = cos(q1 + q2 + q3)
        r_arm2w[0, 1] = -sin(q1 + q2 + q3)
        r_arm2w[1, 0] = sin(q1 + q2 + q3)
        r_arm2w[1, 1] = cos(q1 + q2 + q3)

        ee_vel_w = r_arm2w @ ee_vel_b
        ee_vel_proj = dot(ee_vel_w, normal_vec_c) * normal_vec_c
        ee_vel_orth_w = ee_vel_w - ee_vel_proj
        ee_vel_orth_ref = r_c2w.T @ ee_vel_orth_w

        psi1 = ee_vel_orth_ref[0] + wc * self.circle["radius"] + vxc
        psi2 = vxc - wc * self.circle["radius"]

        t = ((xc - x2) * (x3_act - x2) + (yc - y2) * (y3_act - y2)) / (
            (x3_act - x2) ** 2 + (y3_act - y2) ** 2
        )
        t_clamp = SX.fmax(0, SX.fmin(1, t))
        xp = x2 + t_clamp * (x3_act - x2)
        yp = y2 + t_clamp * (y3_act - y2)
        d = sqrt((xp - xc) ** 2 + (yp - yc) ** 2)
        x_close = vertcat(xp, yp)

        # Store CasADi Functions
        self.kinematics = Function(
            "Kinematics",
            [q, dq, XC, VC, dx_finger],
            [x, dx, a, da, normal_vec_c, d, x_close, x_act],
        )
        self.dynamics = Function(
            "Dynamics",
            [XC, VC, q, dq, u, lam, dx_finger],
            [H, H_c, G_c, phi1, phi2, J_ee, lam_w, lam_c, lam_b, psi1, psi2],
        )
