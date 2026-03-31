import math
import warnings

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "Noto Sans CJK SC",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def calculate_stress(
    epsilonc,
    epsilons1,
    epsilons2,
    fc,
    ecu,
    e0,
    ft,
    et0,
    etu,
    fy1,
    Es1,
    ey1,
    fsu1,
    esu1,
    Esh1,
    fy2,
    Es2,
    ey2,
    fsu2,
    esu2,
    Esh2,
):
    epsilonc = np.asarray(epsilonc, dtype=float)
    epsilons1 = np.asarray(epsilons1, dtype=float)
    epsilons2 = np.asarray(epsilons2, dtype=float)

    # 混凝土应力
    sigmac = np.zeros_like(epsilonc)
    idx_c1 = (epsilonc >= 0) & (epsilonc <= e0)
    r1 = epsilonc[idx_c1] / e0
    sigmac[idx_c1] = fc * (1.55 * r1 - 1.2 * r1**4 + 0.65 * r1**5)

    idx_c2 = (epsilonc > e0) & (epsilonc <= ecu)
    r2 = epsilonc[idx_c2] / e0
    sigmac[idx_c2] = fc * r2 / (6 * (r2 - 1) ** 2 + r2)

    et = np.abs(epsilonc)
    idx_t1 = (epsilonc < 0) & (et <= et0)
    rt1 = et[idx_t1] / et0
    sigmac[idx_t1] = -ft * (1.17 * rt1 + 0.65 * rt1**2 - 0.83 * rt1**3)

    idx_t2 = (epsilonc < 0) & (et > et0) & (et < etu)
    rt2 = et[idx_t2] / et0
    sigmac[idx_t2] = -ft * rt2 / np.sqrt(5.5 * (rt2 - 1) ** 2.2 + rt2)

    # 受压钢筋应力(1)
    sigmas1 = np.zeros_like(epsilons1)
    id1 = (epsilons1 >= 0) & (epsilons1 <= ey1)
    sigmas1[id1] = Es1 * epsilons1[id1]
    id2 = (epsilons1 > ey1) & (epsilons1 <= esu1)
    sigmas1[id2] = fy1 + Esh1 * (epsilons1[id2] - ey1)
    id3 = (epsilons1 < 0) & (np.abs(epsilons1) <= ey1)
    sigmas1[id3] = Es1 * epsilons1[id3]
    id4 = (epsilons1 < 0) & (np.abs(epsilons1) > ey1)
    sigmas1[id4] = -fy1

    # 受拉钢筋应力(2)
    sigmas2 = np.zeros_like(epsilons2)
    id1 = (epsilons2 >= 0) & (epsilons2 <= ey2)
    sigmas2[id1] = Es2 * epsilons2[id1]
    id2 = epsilons2 > ey2
    sigmas2[id2] = fy2
    id3 = (epsilons2 < 0) & (np.abs(epsilons2) <= ey2)
    sigmas2[id3] = Es2 * epsilons2[id3]
    id4 = (epsilons2 < 0) & (np.abs(epsilons2) > ey2) & (np.abs(epsilons2) <= esu2)
    sigmas2[id4] = -(fy2 + Esh2 * (np.abs(epsilons2[id4]) - ey2))
    # 受拉筋断裂后，拉伸应力按 0 MPa 处理（受拉为负应变）
    id5 = (epsilons2 < 0) & (np.abs(epsilons2) > esu2)
    sigmas2[id5] = 0.0

    return sigmac, sigmas1, sigmas2


def calculate_axial_force(
    x,
    phi,
    h,
    k,
    ys1,
    ys2,
    As1,
    As2,
    b,
    fc,
    ecu,
    e0,
    ft,
    et0,
    etu,
    fy1,
    Es1,
    ey1,
    fsu1,
    esu1,
    Esh1,
    fy2,
    Es2,
    ey2,
    fsu2,
    esu2,
    Esh2,
):
    hi = h / k
    epsilon0 = (x - h / 2) * phi
    yi = np.linspace(-h / 2 + hi / 2, h / 2 - hi / 2, k)

    epsilonc = epsilon0 + yi * phi
    epsilons1 = epsilon0 + ys1 * phi
    epsilons2 = epsilon0 + ys2 * phi

    sigmac, sigmas1, sigmas2 = calculate_stress(
        epsilonc,
        np.array([epsilons1]),
        np.array([epsilons2]),
        fc,
        ecu,
        e0,
        ft,
        et0,
        etu,
        fy1,
        Es1,
        ey1,
        fsu1,
        esu1,
        Esh1,
        fy2,
        Es2,
        ey2,
        fsu2,
        esu2,
        Esh2,
    )

    sigmac_at_s1, _, _ = calculate_stress(
        np.array([epsilons1]),
        np.array([0.0]),
        np.array([0.0]),
        fc,
        ecu,
        e0,
        ft,
        et0,
        etu,
        fy1,
        Es1,
        ey1,
        fsu1,
        esu1,
        Esh1,
        fy2,
        Es2,
        ey2,
        fsu2,
        esu2,
        Esh2,
    )
    sigmac_at_s2, _, _ = calculate_stress(
        np.array([epsilons2]),
        np.array([0.0]),
        np.array([0.0]),
        fc,
        ecu,
        e0,
        ft,
        et0,
        etu,
        fy1,
        Es1,
        ey1,
        fsu1,
        esu1,
        Esh1,
        fy2,
        Es2,
        ey2,
        fsu2,
        esu2,
        Esh2,
    )

    Nz = (
        np.sum(sigmac * (hi * b))
        + (sigmas1[0] - sigmac_at_s1[0]) * As1
        + (sigmas2[0] - sigmac_at_s2[0]) * As2
    )
    return float(Nz)


def calculate_moment(
    x,
    phi,
    h,
    k,
    ys1,
    ys2,
    As1,
    As2,
    b,
    fc,
    ecu,
    e0,
    ft,
    et0,
    etu,
    fy1,
    Es1,
    ey1,
    fsu1,
    esu1,
    Esh1,
    fy2,
    Es2,
    ey2,
    fsu2,
    esu2,
    Esh2,
):
    hi = h / k
    epsilon0 = (x - h / 2) * phi
    yi = np.linspace(-h / 2 + hi / 2, h / 2 - hi / 2, k)

    epsilonc = epsilon0 + yi * phi
    epsilons1 = epsilon0 + ys1 * phi
    epsilons2 = epsilon0 + ys2 * phi

    sigmac, sigmas1, sigmas2 = calculate_stress(
        epsilonc,
        np.array([epsilons1]),
        np.array([epsilons2]),
        fc,
        ecu,
        e0,
        ft,
        et0,
        etu,
        fy1,
        Es1,
        ey1,
        fsu1,
        esu1,
        Esh1,
        fy2,
        Es2,
        ey2,
        fsu2,
        esu2,
        Esh2,
    )

    sigmac_at_s1, _, _ = calculate_stress(
        np.array([epsilons1]),
        np.array([0.0]),
        np.array([0.0]),
        fc,
        ecu,
        e0,
        ft,
        et0,
        etu,
        fy1,
        Es1,
        ey1,
        fsu1,
        esu1,
        Esh1,
        fy2,
        Es2,
        ey2,
        fsu2,
        esu2,
        Esh2,
    )
    sigmac_at_s2, _, _ = calculate_stress(
        np.array([epsilons2]),
        np.array([0.0]),
        np.array([0.0]),
        fc,
        ecu,
        e0,
        ft,
        et0,
        etu,
        fy1,
        Es1,
        ey1,
        fsu1,
        esu1,
        Esh1,
        fy2,
        Es2,
        ey2,
        fsu2,
        esu2,
        Esh2,
    )

    Mz = (
        np.sum(sigmac * (hi * b) * yi)
        + (sigmas1[0] - sigmac_at_s1[0]) * As1 * ys1
        + (sigmas2[0] - sigmac_at_s2[0]) * As2 * ys2
    )
    return float(Mz)


def main():
    # 一、基本几何参数
    b = 150.0
    h = 200.0
    cover = 26.0
    h0 = h - cover

    # 受压、受拉纵筋到截面形心的距离（以截面几何中心为原点，向上为正）
    ys1 = (h - 2 * cover) / 2
    ys2 = -(h - 2 * cover) / 2

    As1 = 2 * math.pi * (12 / 2) ** 2
    As2 = 2 * math.pi * (12 / 2) ** 2
    k = 100

    # 二、混凝土材料参数
    fc = 90.0
    fcu = fc / 0.88
    ft = 2.14 * math.sqrt(fc) - 12.8
    Ec = (1.272 * 10 ** -17 * fc**8.28 + 43.39) * 1000
    e0 = (377 * math.sqrt(fc) - 923) * 1e-6
    ecu = 3 * e0
    et0 = 22.9 * ft * 1e-6
    etu = 0.015

    # 三、钢筋材料参数
    fy1 = 1002.0
    Es1 = 43.2e3
    ey1 = fy1 / Es1
    fsu1 = 1002.0
    esu1 = 0.023
    Esh1 = (fsu1 - fy1) / (esu1 - ey1)

    fy2 = 1002.0
    Es2 = 43.2e3
    ey2 = fy2 / Es2
    fsu2 = 1002.0
    esu2 = 0.023
    Esh2 = (fsu2 - fy2) / (esu2 - ey2)

    # 四、计算控制参数
    dphi = 1.0e-6
    n = 0.1
    N = n * fc * b * h

    # 五、初始化结果数组及监控变量
    phi = [0.0]
    Mz = [0.0]

    # 六、主计算循环
    max_steps = 50000
    tolN = 1.0

    x_saved = 0.0
    eck = 0.0
    epsilons2_val = 0.0

    _ = (h0, fcu, Ec)  # 保留与原脚本一致的中间变量定义

    for _j in range(2, max_steps + 1):
        phi_j = phi[-1] + dphi
        phi.append(phi_j)

        # 1. 扫描寻找变号区间
        x_max = 5 * h
        dx = 1.0
        x1 = 0.0
        Nz1 = calculate_axial_force(
            x1,
            phi_j,
            h,
            k,
            ys1,
            ys2,
            As1,
            As2,
            b,
            fc,
            ecu,
            e0,
            ft,
            et0,
            etu,
            fy1,
            Es1,
            ey1,
            fsu1,
            esu1,
            Esh1,
            fy2,
            Es2,
            ey2,
            fsu2,
            esu2,
            Esh2,
        )
        F1 = Nz1 - N
        found_bracket = False
        x2 = x1

        if abs(F1) <= tolN:
            x = x1
            found_bracket = True
        else:
            while x2 < x_max:
                x2 += dx
                Nz2 = calculate_axial_force(
                    x2,
                    phi_j,
                    h,
                    k,
                    ys1,
                    ys2,
                    As1,
                    As2,
                    b,
                    fc,
                    ecu,
                    e0,
                    ft,
                    et0,
                    etu,
                    fy1,
                    Es1,
                    ey1,
                    fsu1,
                    esu1,
                    Esh1,
                    fy2,
                    Es2,
                    ey2,
                    fsu2,
                    esu2,
                    Esh2,
                )
                F2 = Nz2 - N
                if F1 * F2 <= 0:
                    found_bracket = True
                    break
                x1 = x2
                F1 = F2

        if not found_bracket:
            warnings.warn("未找到轴力平衡变号区间，可能截面已失效，计算停止。")
            break

        # 2. 二分法求解受压区高度 x
        iter_count = 0
        max_iter = 100
        if abs(F1) > tolN:
            while iter_count < max_iter:
                x = (x1 + x2) / 2
                Nz = calculate_axial_force(
                    x,
                    phi_j,
                    h,
                    k,
                    ys1,
                    ys2,
                    As1,
                    As2,
                    b,
                    fc,
                    ecu,
                    e0,
                    ft,
                    et0,
                    etu,
                    fy1,
                    Es1,
                    ey1,
                    fsu1,
                    esu1,
                    Esh1,
                    fy2,
                    Es2,
                    ey2,
                    fsu2,
                    esu2,
                    Esh2,
                )
                Fm = Nz - N
                if abs(Fm) <= tolN:
                    break
                if F1 * Fm <= 0:
                    x2 = x
                else:
                    x1 = x
                    F1 = Fm
                iter_count += 1

        # 3. 计算本步的应变与弯矩
        eck = x * phi_j
        epsilons2_val = (x - h / 2 + ys2) * phi_j

        Mz_j = calculate_moment(
            x,
            phi_j,
            h,
            k,
            ys1,
            ys2,
            As1,
            As2,
            b,
            fc,
            ecu,
            e0,
            ft,
            et0,
            etu,
            fy1,
            Es1,
            ey1,
            fsu1,
            esu1,
            Esh1,
            fy2,
            Es2,
            ey2,
            fsu2,
            esu2,
            Esh2,
        )
        Mz.append(Mz_j)
        x_saved = x

        # 当混凝土压应变达到1.0倍的极限压应变时停止计算
        if eck >= 1.0 * ecu:
            print("\n=>混凝土顶部达到极限压应变！")
            break

    print("============== 计算完成 ==============")
    print(f"最终受压区高度 x = {x_saved} mm")
    print(f"最大边缘压应变 e_c = {eck}")
    print(f"受拉钢筋拉应变 e_s = {abs(epsilons2_val)}")
    print(f"极限曲率 phi_u = {phi[-1]} 1/mm")
    print(f"极限弯矩 M_u = {Mz[-1] / 1e6} kN·m")

    plt.figure()
    plt.plot(np.array(phi) * 1000, np.array(Mz) / 1e6, linewidth=2, color="b")
    plt.grid(True)
    plt.xlabel("曲率 φ (1/m)")
    plt.ylabel("弯矩 M_z (kN·m)")
    plt.title("矩形配筋混凝土截面弯矩-曲率曲线")
    plt.show()


if __name__ == "__main__":
    main()
