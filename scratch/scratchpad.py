# K = 4.0
# R = 7.98
# alpha_eye = [6.21616, -6.21626]
# beta_eye = 0.29797849
# theta_cam = 5.0
# kappa_cam = 4.66150654


    params = (7.8, 4.75, 5, 1.5, 0.1, 0.1)
    bounds = ((4, 12),
              (2, 8),
              (0, 10),
              (0, 5),
              (-5, 5),
              (-5, 5))
    args = (glints[0][0:200],
            ppos[0][0:200],
            targets[0:200])
    kc1 = minimize(calibrate, params, args=args,
                            bounds=bounds,
                            method='SLSQP', tol=1e-2)
    print(kc1)


def k_qsp(k_c, o, uvw, b, rk):
    """k_q, k_s, k_p (2.33, 2.34, 2.35)"""
    num = k_c * np.dot(o - uvw, b) \
        - np.sqrt(k_c**2 * np.dot(o - uvw, b)**2 \
                  - LA.norm(o - uvw)**2 * (k_c**2 - rk**2))
    denom = LA.norm(o - uvw)**2
    return (num / denom)

def glints_qsp(o, uvw, kqsp):
    """pupil center p, reflections s & q (2.10, 2.8, 2.4)"""
    return o + kqsp * (o - uvw)

def curvaturecenter_c(o, b, kc):
    """center of corneal curvature c (2.29)"""
    return o + kc * b

def find_min_distance(params, eye1, gaze1, eye2, gaze2):
    """Find the minimum distance between left & right visual axes"""
    t1, t2 = params
    vec1 = eye1 + t1 * (gaze1 - eye1)
    vec2 = eye2 + t2 * (gaze2 - eye2)
    return LA.norm(vec1 - vec2)

def solve_kc_qc(kc, u, w, o, l, m, b, kr):
    """substitute q, c and solve for k_c (2.2)"""
    kq = k_qsp(kc, o, u, b, kr)
    q = glints_qsp(o, u, kq)
    c = curvaturecenter_c(o, b, kc)

    lhs = np.dot(l - q, q - c) * LA.norm(o - q)
    rhs = np.dot(o - q, q - c) * LA.norm(l - q)
    return lhs - rhs

def solve_kc_sc(kc, u, w, o, l, m, b, kr):
    """substitute s, c and solve for k_c (2.6)"""
    ks = k_qsp(kc, o, w, b, kr)
    s = glints_qsp(o, w, ks)
    c = curvaturecenter_c(o, b, kc)

    lhs = np.dot(m - s, s - c) * LA.norm(o - s)
    rhs = np.dot(o - s, s - c) * LA.norm(m - s)
    return lhs - rhs

def calibrate(params, *args):
    r, k, alpha, beta, theta, kappa = params
    glints, pupils, targets = args

    # determine coordinate transformation parameters
    kcam = k_cam(phi_cam, theta)
    ic0 = i_cam_0(np.array([0, 1, 0]), kcam)
    jc0 = j_cam_0(kcam, ic0)
    icam = i_cam(ic0, jc0, kappa)
    jcam = j_cam(ic0, jc0, kappa)
    ijkcam = np.array([icam, jcam, kcam])

    # transform image to camera coordinates
    glint = [to_ccs(np.mean(glints, axis=1)[0], c_center, p_pitch),
             to_ccs(np.mean(glints, axis=1)[1], c_center, p_pitch)]
    pupil = to_ccs(np.mean(pupils, axis=0), c_center, p_pitch)
    glint[0] = to_wcs(ijkcam, glint[0], t_trans)
    glint[1] = to_wcs(ijkcam, glint[1], t_trans)
    pupil = to_wcs(ijkcam, pupil, t_trans)

    # bnorm vector
    bnorm = b_norm(source[0], source[1], glint[0], glint[1], nodal_point)

    # obtain k_c for both glints
    args = (glint[0],
            glint[1],
            nodal_point,
            source[0],
            source[1],
            bnorm,
            r)
    kc1 = fsolve(solve_kc_qc, 0, args=args)
    kc2 = fsolve(solve_kc_sc, kc1[0], args=args)

    # use mean as result
    kc = (kc1[0] + kc2[0]) / 2

    # calculate c and p from kc
    kp = k_qsp(kc, nodal_point, pupil, bnorm, k)
    c_res = curvaturecenter_c(nodal_point, bnorm, kc)
    p_res = glints_qsp(nodal_point, pupil, kp)

    # optical axis is vector given by c & p
    o_ax = p_res - c_res
    o_ax_norm = o_ax / LA.norm(o_ax)

    # calculate phi_eye and theta_eye from c_res and p_res
    val = (np.abs(p_res[1] - c_res[1])) / k
    phi_eye = np.arcsin(val)
    theta_eye = -np.arctan((p_res[0] - c_res[0]) /
                           (p_res[2] - c_res[2]))

    # calculate k_g from pan and tilt angles
    k_g = c_res[2] / (np.cos(phi_eye + beta) * np.cos(theta_eye + alpha))

    # calculate gaze point g from k_g
    x1 = np.cos(phi_eye + beta) * np.sin(np.deg2rad(theta_eye + alpha))
    x2 = np.sin(np.deg2rad(phi_eye + beta))
    x3 = -np.cos(np.deg2rad(phi_eye + beta)) * np.cos(np.deg2rad(theta_eye + alpha))
    gazepoint = c_res + k_g * np.array([x1, x2, x3])

    target = np.array([np.mean(targets, axis=0)[0], np.mean(targets, axis=0)[1], 0])
    return LA.norm(target - gazepoint)

def main(rng, innerplots=False, outerplots=True, savefig=False):
    if rng > 1:
        sys.stdout = DevNull()

    # initialize solution arrays
    glints_wcs = [[] for _ in (0, 1)]
    pupils_wcs = [[] for _ in (0, 1)]
    gazepoints = [[] for _ in (0, 1)]
    mindists = [[] for _ in (0, 1)]
    c_res = [[] for _ in (0, 1)]
    p_res = [[] for _ in (0, 1)]

    # determine coordinate transformation parameters
    kcam = k_cam(phi_cam, theta_cam)
    ic0 = i_cam_0(np.array([0, 1, 0]), kcam)
    jc0 = j_cam_0(kcam, ic0)
    icam = i_cam(ic0, jc0, kappa_cam)
    jcam = j_cam(ic0, jc0, kappa_cam)
    ijkcam = np.array([icam, jcam, kcam])

    for i in tqdm(range(rng), ncols=80):  # images to scan
        for j in (0, 1):                  # left, right
            # transform image to camera coordinates
            glint = [to_ccs(glints[j][0][i], c_center, p_pitch),
                     to_ccs(glints[j][1][i], c_center, p_pitch)]
            pupil = to_ccs(ppos[j][i], c_center, p_pitch)

            glint[0] = to_wcs(ijkcam, glint[0], t_trans)
            glint[1] = to_wcs(ijkcam, glint[1], t_trans)
            pupil = to_wcs(ijkcam, pupil, t_trans)

            glints_wcs[j].append([glint[0], glint[1]])
            pupils_wcs[j].append(pupil)

            # bnorm vector
            bnorm = b_norm(source[0], source[1], glint[0], glint[1], nodal_point)

            # obtain k_c for both glints
            args = (glint[0],
                    glint[1],
                    nodal_point,
                    source[0],
                    source[1],
                    bnorm,
                    R)
            kc1 = fsolve(solve_kc_qc, 0, args=args)
            kc2 = fsolve(solve_kc_sc, kc1[0], args=args)

            # use mean as result
            kc = (kc1[0] + kc2[0]) / 2

            # calculate c and p from kc
            kp = k_qsp(kc, nodal_point, pupil, bnorm, K)
            c_res[j].append(curvaturecenter_c(nodal_point, bnorm, kc))
            p_res[j].append(glints_qsp(nodal_point, pupil, kp))

            # optical axis is vector given by c & p
            o_ax = p_res[j][i] - c_res[j][i]
            o_ax_norm = o_ax / LA.norm(o_ax)

            # calculate phi_eye and theta_eye from c_res and p_res
            phi_eye = np.arcsin((p_res[j][i][1] - c_res[j][i][1]) / K)
            theta_eye = -np.arctan((p_res[j][i][0] - c_res[j][i][0]) /
                                   (p_res[j][i][2] - c_res[j][i][2]))

            # calculate k_g from pan and tilt angles
            k_g = c_res[j][i][2] / (np.cos(phi_eye + beta_eye) * np.cos(theta_eye + alpha_eye[j]))

            # calculate gaze point g from k_g
            x1 = np.cos(phi_eye + beta_eye) * np.sin(np.deg2rad(theta_eye + alpha_eye[j]))
            x2 = np.sin(np.deg2rad(phi_eye + beta_eye))
            x3 = -np.cos(np.deg2rad(phi_eye + beta_eye)) * np.cos(np.deg2rad(theta_eye + alpha_eye[j]))
            gazepoints[j].append(c_res[j][i] + k_g * np.array([x1, x2, x3]))

        # calculate shortest distance between visual axes ("intersection")
        mindist = minimize(find_min_distance, (1, 1),
                                  args=(c_res[0][i], gazepoints[0][i], c_res[1][i], gazepoints[1][i]),
                                  bounds=((-100, 100),
                                          (-100, 100)),
                                  method='SLSQP',
                                  tol=1e-5)

        for j in (0, 1):
            solp = c_res[j][i] + (gazepoints[j][i] - c_res[j][i]) * mindist.x[j]
            mindists[j].append(solp)

        if innerplots:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(*np.array(source).T, c='k')
            ax.scatter(*nodal_point.T, c='k', label='Nodal point')
            ax.scatter(*c_res[0][i].T, c='b', label='Center of corneal curvature (left)')
            ax.scatter(*c_res[1][i].T, c='r', label='Center of corneal curvature (right)')
            ax.scatter(*gazepoints[0][i], c='b', marker='x', label='Gaze point (left)')
            ax.scatter(*gazepoints[1][i], c='r', marker='x', label='Gaze point (right)')
            ax.scatter(*np.array(mindists[0][i]).T, c='y', label='Closest point (left)')
            ax.scatter(*np.array(mindists[1][i]).T, c='y', label='Closest point (right)')
            ax.scatter(*np.unique(targets, axis=0).T, [0] * 9, c='k', label='Nodal point')
            ax.plot(*np.array((c_res[0][i], gazepoints[0][i])).T, c='b', linestyle='--')
            ax.plot(*np.array((c_res[1][i], gazepoints[1][i])).T, c='r', linestyle='--')
            ax.auto_scale_xyz([-500, 500], [-500, 500], [-1000, 0])
            ax.view_init(elev=110, azim=-90)
            plt.title('Out-of-camera view')
            ax.set_xlabel('x (mm)')
            ax.set_ylabel('y (mm)')
            ax.set_zlabel('z (mm)')
            plt.legend()
            plt.tight_layout()
            if savefig:
                plt.savefig('./plots/out_of_camera_{:04d}.png'.format(i))
            else:
                plt.show()
            plt.close()

    if outerplots:
        mindists[0] = np.array(mindists[0])
        mindists[1] = np.array(mindists[1])
        mindists = mindists[0] + 0.5 * (mindists[1] - mindists[0])
        for j in (0, 1):
            gazepoints[j] = np.array(gazepoints[j])
            gazepoints[j] = gazepoints[j][~np.isnan(gazepoints[j]).any(axis=1)]
            gazepoints[j] = gazepoints[j][abs(gazepoints[j][:, 0]) < 2000]
            gazepoints[j] = gazepoints[j][abs(gazepoints[j][:, 1]) < 2000]
            mindists = mindists[~np.isnan(mindists).any(axis=1)]
            mindists = mindists[abs(mindists[:, 0]) < 100]
            mindists = mindists[abs(mindists[:, 1]) < 100]
            mindists = mindists[abs(mindists[:, 2]) < 100]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*gazepoints[0].T, label='Gaze (left)')
        ax.scatter(*gazepoints[1].T, label='Gaze (right)')
        plt.title('Gaze points for {} frames'.format(len(gazepoints[0])))
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_zlabel('z (mm)')
        plt.legend()
        if savefig:
            plt.savefig('./plots/total_gaze_3d.png')
        else:
            plt.show()
        plt.close()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*mindists.T, label='Gaze intersect')
        plt.title('Gaze intersect for {} frames'.format(len(gazepoints[0])))
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_zlabel('z (mm)')
        plt.legend()
        if savefig:
            plt.savefig('./plots/total_intersect_3d.png')
        else:
            plt.show()
        plt.close()

        fig, ax = plt.subplots()
        ax.scatter([*gazepoints[0].T][0], [*gazepoints[0].T][1], label='Gaze (left)')
        ax.scatter([*gazepoints[1].T][0], [*gazepoints[1].T][1], label='Gaze (right)')
        plt.title('xy projection of gaze points for {} frames'.format(len(gazepoints[0])))
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        plt.legend()
        if savefig:
            plt.savefig('./plots/total_gaze_2d.png')
        else:
            plt.show()
        plt.close()

        fig, ax = plt.subplots()
        ax.scatter([*mindists.T][0], [*mindists.T][1])
        plt.title('xy projection of gaze intersect for {} frames'.format(len(gazepoints[0])))
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        if savefig:
            plt.savefig('./plots/total_intersect_2d.png')
        else:
            plt.show()
        plt.close()
