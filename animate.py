import numpy as np
from casadi import DM
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
from meshcat.animation import Animation


def update_finger(vis, anim, frame_idx, robot, q, xc, dx_finger_val):
    x_val = robot.kinematics(
        DM(q), DM([0, 0, 0]), DM(xc), DM([0, 0, 0]), DM([dx_finger_val])
    )[0]
    x_val = np.array(x_val.full(), dtype=float)

    origin = np.asarray(robot.arm["origin"], dtype=float).reshape(2, 1)
    pts = np.hstack([origin, x_val])
    pts -= origin
    pts3 = np.vstack([pts, np.zeros((1, pts.shape[1]))])

    M = pts3.shape[1]
    for j in range(1, M):
        p = pts3[:, j]
        T = tf.translation_matrix([float(p[0]), float(p[1]), float(p[2])])
        with anim.at_frame(vis[f"finger/joint/{j-1}"], frame_idx) as f:
            f.set_transform(T)

    radius = float(robot.circle["radius"]) / 2
    z_axis = np.array([0.0, 0.0, 1.0])
    finger_length = np.array(robot.arm["length"].full()).ravel()

    for j in range(0, M - 1):
        p0 = pts3[:, j]
        p1 = pts3[:, j + 1]
        v = p1 - p0
        L = float(np.linalg.norm(v))
        base = f"finger/capsule/{j}"

        dir = v / L
        c = float(np.clip(np.dot(z_axis, dir), -1.0, 1.0))
        if np.isclose(c, 1.0):
            R = np.eye(4)
        elif np.isclose(c, -1.0):
            R = tf.rotation_matrix(np.pi, [1, 0, 0])
        else:
            axis = np.cross(z_axis, dir)
            axis /= np.linalg.norm(axis)
            angle = np.arccos(c)
            R = tf.rotation_matrix(angle, axis)

        if j == (M - 2):
            L_visual = float(finger_length[-1])
            v_visual = dir * L_visual
            p1_visual = p0 + v_visual
            T_parent = tf.translation_matrix(((p0 + p1_visual) / 2.0)) @ R
            LB = max(0.0, L_visual - 2.0 * radius)
        else:
            T_parent = tf.translation_matrix(((p0 + p1) / 2.0)) @ R
            LB = max(0.0, L - 2.0 * radius)

        z_off = (LB / 2.0) + radius
        S_body = np.diag([radius, radius, LB, 1.0])
        S_sph = np.diag([radius, radius, radius, 1.0])

        with anim.at_frame(vis[base], frame_idx) as f:
            f.set_transform(T_parent)
        with anim.at_frame(vis[f"{base}/body"], frame_idx) as f:
            f.set_transform(tf.rotation_matrix(np.pi / 2, [1, 0, 0]) @ S_body)
        with anim.at_frame(vis[f"{base}/end0"], frame_idx) as f:
            f.set_transform(tf.translation_matrix([0.0, 0.0, -z_off]) @ S_sph)
        with anim.at_frame(vis[f"{base}/end1"], frame_idx) as f:
            f.set_transform(tf.translation_matrix([0.0, 0.0, z_off]) @ S_sph)


def setup_scene(vis, robot, circle_center, dx):
    radius = float(robot.circle["radius"]) / 2
    circle = g.Cylinder(radius * 2, radius)
    mat_circle = g.MeshPhongMaterial(color=0x87CEEB, opacity=0.7)

    vis["circle"].set_object(circle, mat_circle)
    vis["indicator"].set_object(g.Sphere(0.005), g.MeshPhongMaterial(color=0x000000))

    circle_init_T = (
        tf.translation_matrix([circle_center[0], circle_center[1], 0.0])
        @ tf.rotation_matrix(0, [0, 0, 1])
        @ tf.rotation_matrix(np.pi / 2, [1, 0, 0])
    )
    indicator_T = circle_init_T @ tf.translation_matrix([-radius * 0.8, radius, 0.0])
    vis["circle"].set_transform(circle_init_T)
    vis["indicator"].set_transform(indicator_T)

    cam_tf = (
        tf.translation_matrix([0.0, 0.0, 45.0])
        @ tf.rotation_matrix(0.5 * np.pi, [0, 0, 1])
        @ tf.rotation_matrix(-np.pi / 2.5, [0, 1, 0])
    )

    n_links = 3
    finger_length = np.array(robot.arm["length"].full()).ravel()

    mat_links = g.MeshPhongMaterial(color=0x333333, opacity=0.95)
    mat_end_effector = g.MeshPhongMaterial(color=0xFF9999, opacity=0.95)

    for s in range(n_links):
        base = f"finger/capsule/{s}"
        if s == n_links - 1:
            vis[f"{base}/body"].set_object(
                g.Cylinder(finger_length[s], radius), mat_end_effector
            )
            vis[f"{base}/end0"].set_object(g.Sphere(radius), mat_links)
            vis[f"{base}/end1"].set_object(g.Sphere(radius), mat_end_effector)
        else:
            vis[f"{base}/body"].set_object(
                g.Cylinder(finger_length[s], radius), mat_links
            )
            vis[f"{base}/end0"].set_object(g.Sphere(radius), mat_links)
            vis[f"{base}/end1"].set_object(g.Sphere(radius), mat_links)
        vis[f"{base}/body"].set_transform(tf.rotation_matrix(np.pi / 2, [1, 0, 0]))

    vis["/Grid"].delete()
    vis["/Axes"].delete()  # FIXME : This doesn't seem to work.
    vis["/Background"].set_property("visible", True)
    vis["/Background"].set_property("top_color", [0.98, 0.98, 0.98])
    vis["/Background"].set_property("bottom_color", [0.98, 0.98, 0.98])
    vis["/Cameras/default"].set_transform(cam_tf)
    vis["/Cameras/default/rotated/<object>"].set_property("zoom", 50)


def play_trajectory(
    vis,
    qsol,
    xcsol,
    dx_finger_sol,
    robot,
    circle_center_first,
    offsets=(0.0, 0.0),
    dt=0.05,
):
    anim = Animation()
    anim.default_framerate = int(round(1.0 / dt))
    N = qsol.shape[1]
    x_offset, y_offset = offsets

    radius = float(robot.circle["radius"]) / 2
    Rx90 = tf.rotation_matrix(np.pi / 2, [1, 0, 0])
    xc0 = xcsol[:, 0]
    for i in range(0, N):
        frame_idx = i
        update_finger(
            vis, anim, frame_idx, robot, qsol[:, i], xcsol[:, i], dx_finger_sol[i]
        )
        cx, cy, yaw = map(float, xcsol[:, i])
        cx += x_offset
        cy += y_offset
        T_circle = (
            tf.translation_matrix([cx, cy, 0.0])
            @ tf.rotation_matrix(yaw, [0, 0, 1])
            @ Rx90
        )
        T_indicator = T_circle @ tf.translation_matrix([-radius * 0.8, radius, 0.0])
        with anim.at_frame(vis["circle"], frame_idx) as f:
            f.set_transform(T_circle)
        with anim.at_frame(vis["indicator"], frame_idx) as k:
            k.set_transform(T_indicator)
    vis.set_animation(anim)


if __name__ == "__main__":
    from finger import Finger

    data = np.load("finger_solution.npz", allow_pickle=True)
    qsol = data["qsol"]
    xcsol = data["xcsol"]
    dx_finger_sol = data["dx_finger_sol"].reshape(-1)
    x_goal = data["x_goal"].copy()
    circle_center_np = data["circle_center"].copy()
    circle_center_DM = DM(circle_center_np)
    finger_center_DM = DM([0, 0.95])
    robot = Finger(circle_center_DM, finger_center_DM)

    finger_origin = np.array(robot.arm["origin"], dtype=float).flatten()
    x_offset, y_offset = -finger_origin

    x_goal[0] += x_offset
    x_goal[1] += y_offset
    circle_center_np[0] += x_offset
    circle_center_np[1] += y_offset

    np.set_printoptions(precision=12, suppress=True)
    vis = meshcat.Visualizer().open()
    setup_scene(vis, robot, circle_center_np, dx_finger_sol[0])

    offsets = (x_offset, y_offset)
    play_trajectory(
        vis,
        qsol,
        xcsol,
        dx_finger_sol,
        robot,
        circle_center_np,
        offsets,
        0.05,
    )
