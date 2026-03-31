"""
Title: Axial Force-Moment-Curvature Calculation Model for SFCB Reinforced UHPC Rectangular Sections
Environment: Python 3.8+
Author: Zhiwen Zhang
Description: This script performs the numerical analysis of axial force, moment, and curvature for rectangular UHPC sections reinforced with Steel-FRP Composite Bars (SFCB) based on cross-sectional equilibrium and material constitutive models.
"""

import math
import warnings
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

def calculate_stress(
    epsilonc, epsilons1, epsilons2, fc, ecu, e0, ft, et0, etu,
    fy1, Es1, ey1, fsu1, esu1, Esh1, fy2, Es2, ey2, fsu2, esu2, Esh2,
    concrete_type="UHPC"
):
    epsilonc = np.asarray(epsilonc, dtype=float)
    epsilons1 = np.asarray(epsilons1, dtype=float)
    epsilons2 = np.asarray(epsilons2, dtype=float)

    sigmac = np.zeros_like(epsilonc)
    
    if concrete_type == "UHPC":
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
    else:
        a = max(1.5, 2.4 - 0.0125 * fc)
        alpha_c = max(0.157 * (fc ** 0.785) - 0.905, 0.5)
        
        idx_c1 = (epsilonc >= 0) & (epsilonc <= e0)
        x_c1 = epsilonc[idx_c1] / e0
        sigmac[idx_c1] = fc * (a * x_c1 + (3 - 2*a) * x_c1**2 + (a - 2) * x_c1**3)
        
        idx_c2 = (epsilonc > e0) & (epsilonc <= ecu)
        x_c2 = epsilonc[idx_c2] / e0
        sigmac[idx_c2] = fc * (x_c2 / (alpha_c * (x_c2 - 1)**2 + x_c2))
        
        alpha_t = 0.312 * (ft ** 2)
        et = np.abs(epsilonc)
        
        idx_t1 = (epsilonc < 0) & (et <= et0)
        x_t1 = et[idx_t1] / et0
        sigmac[idx_t1] = -ft * (1.2 * x_t1 - 0.2 * x_t1**6)
        
        idx_t2 = (epsilonc < 0) & (et > et0) & (et < etu)
        x_t2 = et[idx_t2] / et0
        sigmac[idx_t2] = -ft * (x_t2 / (alpha_t * (x_t2 - 1)**1.7 + x_t2))

    sigmas1 = np.zeros_like(epsilons1)
    id1 = (epsilons1 >= 0) & (epsilons1 <= ey1)
    sigmas1[id1] = Es1 * epsilons1[id1]
    id2 = (epsilons1 > ey1) & (epsilons1 <= esu1)
    sigmas1[id2] = fy1 + Esh1 * (epsilons1[id2] - ey1)
    id3 = (epsilons1 < 0) & (np.abs(epsilons1) <= ey1)
    sigmas1[id3] = Es1 * epsilons1[id3]
    id4 = (epsilons1 < 0) & (np.abs(epsilons1) > ey1)
    sigmas1[id4] = -fy1

    sigmas2 = np.zeros_like(epsilons2)
    id1 = (epsilons2 >= 0) & (epsilons2 <= ey2)
    sigmas2[id1] = Es2 * epsilons2[id1]
    id2 = epsilons2 > ey2
    sigmas2[id2] = fy2
    id3 = (epsilons2 < 0) & (np.abs(epsilons2) <= ey2)
    sigmas2[id3] = Es2 * epsilons2[id3]
    id4 = (epsilons2 < 0) & (np.abs(epsilons2) > ey2) & (np.abs(epsilons2) <= esu2)
    sigmas2[id4] = -(fy2 + Esh2 * (np.abs(epsilons2[id4]) - ey2))
    
    id5 = (epsilons2 < 0) & (np.abs(epsilons2) > esu2)
    sigmas2[id5] = 0.0

    return sigmac, sigmas1, sigmas2

def calculate_axial_force(
    x, phi, h, k, ys1, ys2, As1, As2, b, fc, ecu, e0, ft, et0, etu,
    fy1, Es1, ey1, fsu1, esu1, Esh1, fy2, Es2, ey2, fsu2, esu2, Esh2, concrete_type="UHPC"
):
    hi = h / k
    epsilon0 = (x - h / 2) * phi
    yi = np.linspace(-h / 2 + hi / 2, h / 2 - hi / 2, k)

    epsilonc = epsilon0 + yi * phi
    epsilons1 = epsilon0 + ys1 * phi
    epsilons2 = epsilon0 + ys2 * phi

    sigmac, sigmas1, sigmas2 = calculate_stress(
        epsilonc, np.array([epsilons1]), np.array([epsilons2]),
        fc, ecu, e0, ft, et0, etu, fy1, Es1, ey1, fsu1, esu1, Esh1,
        fy2, Es2, ey2, fsu2, esu2, Esh2, concrete_type
    )

    sigmac_at_s1, _, _ = calculate_stress(
        np.array([epsilons1]), np.array([0.0]), np.array([0.0]),
        fc, ecu, e0, ft, et0, etu, fy1, Es1, ey1, fsu1, esu1, Esh1,
        fy2, Es2, ey2, fsu2, esu2, Esh2, concrete_type
    )
    sigmac_at_s2, _, _ = calculate_stress(
        np.array([epsilons2]), np.array([0.0]), np.array([0.0]),
        fc, ecu, e0, ft, et0, etu, fy1, Es1, ey1, fsu1, esu1, Esh1,
        fy2, Es2, ey2, fsu2, esu2, Esh2, concrete_type
    )

    Nz = (
        np.sum(sigmac * (hi * b))
        + (sigmas1[0] - sigmac_at_s1[0]) * As1
        + (sigmas2[0] - sigmac_at_s2[0]) * As2
    )
    return float(Nz)

def calculate_moment(
    x, phi, h, k, ys1, ys2, As1, As2, b, fc, ecu, e0, ft, et0, etu,
    fy1, Es1, ey1, fsu1, esu1, Esh1, fy2, Es2, ey2, fsu2, esu2, Esh2, concrete_type="UHPC"
):
    hi = h / k
    epsilon0 = (x - h / 2) * phi
    yi = np.linspace(-h / 2 + hi / 2, h / 2 - hi / 2, k)

    epsilonc = epsilon0 + yi * phi
    epsilons1 = epsilon0 + ys1 * phi
    epsilons2 = epsilon0 + ys2 * phi

    sigmac, sigmas1, sigmas2 = calculate_stress(
        epsilonc, np.array([epsilons1]), np.array([epsilons2]),
        fc, ecu, e0, ft, et0, etu, fy1, Es1, ey1, fsu1, esu1, Esh1,
        fy2, Es2, ey2, fsu2, esu2, Esh2, concrete_type
    )

    sigmac_at_s1, _, _ = calculate_stress(
        np.array([epsilons1]), np.array([0.0]), np.array([0.0]),
        fc, ecu, e0, ft, et0, etu, fy1, Es1, ey1, fsu1, esu1, Esh1,
        fy2, Es2, ey2, fsu2, esu2, Esh2, concrete_type
    )
    sigmac_at_s2, _, _ = calculate_stress(
        np.array([epsilons2]), np.array([0.0]), np.array([0.0]),
        fc, ecu, e0, ft, et0, etu, fy1, Es1, ey1, fsu1, esu1, Esh1,
        fy2, Es2, ey2, fsu2, esu2, Esh2, concrete_type
    )

    Mz = (
        np.sum(sigmac * (hi * b) * yi)
        + (sigmas1[0] - sigmac_at_s1[0]) * As1 * ys1
        + (sigmas2[0] - sigmac_at_s2[0]) * As2 * ys2
    )
    return float(Mz), float(sigmas1[0]), float(sigmas2[0])

class MNPHIGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Axial Force-Moment-Curvature Calculation Model for SFCB Reinforced UHPC Rectangular Sections")
        
        self.root.geometry("800x950")
        
        style = ttk.Style()
        default_font = ("Arial", 12)
        bold_font = ("Arial", 12, "bold")
        
        style.configure(".", font=default_font)
        style.configure("TLabel", font=default_font)
        style.configure("TButton", font=default_font)
        style.configure("TCheckbutton", font=default_font)
        style.configure("TLabelframe.Label", font=bold_font)
        
        self.results_df = None
        
        self.create_widgets()
        
    def create_widgets(self):
        self.params = {
            "b": tk.StringVar(value="150"),
            "h": tk.StringVar(value="200"),
            "cover": tk.StringVar(value="26"),
            
            "concrete_type": tk.StringVar(value="UHPC"),
            "fc": tk.StringVar(value="90"),
            "ecu": tk.StringVar(value="0.008"),
            "n": tk.StringVar(value="0.0"),
            
            "As1": tk.StringVar(value="226"),
            "Es1": tk.StringVar(value="200000"),
            "fy1": tk.StringVar(value="400"),
            "fsu1": tk.StringVar(value="560"),
            "esu1": tk.StringVar(value="0.2"),
            
            "As2": tk.StringVar(value="339"),
            "Es2": tk.StringVar(value="200000"),
            "fy2": tk.StringVar(value="400"),
            "fsu2": tk.StringVar(value="560"),
            "esu2": tk.StringVar(value="0.2"),
        }
        
        top_frame = ttk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=15, pady=10)
        
        frame_geom = ttk.LabelFrame(top_frame, text="Geometric Parameters", padding=(10, 5))
        frame_geom.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
        frame_geom.columnconfigure(1, weight=1)
        geom_labels = [
            ("Section Width (mm):", "b"),
            ("Section Height (mm):", "h"),
            ("Cover Thickness (mm):", "cover"),
        ]
        for i, (text, key) in enumerate(geom_labels):
            ttk.Label(frame_geom, text=text).grid(row=i, column=0, sticky=tk.W, pady=5)
            ttk.Entry(frame_geom, textvariable=self.params[key], font=("Arial", 12)).grid(row=i, column=1, pady=5, padx=5, sticky=tk.EW)
            
        frame_conc = ttk.LabelFrame(top_frame, text="Concrete Parameters", padding=(10, 5))
        frame_conc.grid(row=0, column=1, padx=10, pady=5, sticky="nsew")
        frame_conc.columnconfigure(1, weight=1)
        
        ttk.Label(frame_conc, text="Concrete Type:").grid(row=0, column=0, sticky=tk.W, pady=5)
        cb = ttk.Combobox(frame_conc, textvariable=self.params["concrete_type"], values=["UHPC", "NSC"], state="readonly", font=("Arial", 12))
        cb.grid(row=0, column=1, pady=5, padx=5, sticky=tk.EW)
        cb.bind("<<ComboboxSelected>>", self.on_concrete_type_change)
        
        conc_labels = [
            ("Compressive Strength (MPa):", "fc"),
            ("Ultimate Comp. Strain:", "ecu"),
            ("Axial Load Ratio:", "n"),
        ]
        for i, (text, key) in enumerate(conc_labels, start=1):
            ttk.Label(frame_conc, text=text).grid(row=i, column=0, sticky=tk.W, pady=5)
            ttk.Entry(frame_conc, textvariable=self.params[key], font=("Arial", 12)).grid(row=i, column=1, pady=5, padx=5, sticky=tk.EW)
            
        frame_s1 = ttk.LabelFrame(top_frame, text="Compression Rebar Parameters", padding=(10, 5))
        frame_s1.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        frame_s1.columnconfigure(1, weight=1)
        s1_labels = [
            ("Area (mm²):", "As1"),
            ("Elastic Modulus (MPa):", "Es1"),
            ("Yield Strength (MPa):", "fy1"),
            ("Ultimate Strength (MPa):", "fsu1"),
            ("Rupture Strain:", "esu1"),
        ]
        for i, (text, key) in enumerate(s1_labels):
            ttk.Label(frame_s1, text=text).grid(row=i, column=0, sticky=tk.W, pady=5)
            ttk.Entry(frame_s1, textvariable=self.params[key], font=("Arial", 12)).grid(row=i, column=1, pady=5, padx=5, sticky=tk.EW)
            
        frame_s2 = ttk.LabelFrame(top_frame, text="Tension Rebar Parameters", padding=(10, 5))
        frame_s2.grid(row=1, column=1, padx=10, pady=5, sticky="nsew")
        frame_s2.columnconfigure(1, weight=1)
        s2_labels = [
            ("Area (mm²):", "As2"),
            ("Elastic Modulus (MPa):", "Es2"),
            ("Yield Strength (MPa):", "fy2"),
            ("Ultimate Strength (MPa):", "fsu2"),
            ("Rupture Strain:", "esu2"),
        ]
        for i, (text, key) in enumerate(s2_labels):
            ttk.Label(frame_s2, text=text).grid(row=i, column=0, sticky=tk.W, pady=5)
            ttk.Entry(frame_s2, textvariable=self.params[key], font=("Arial", 12)).grid(row=i, column=1, pady=5, padx=5, sticky=tk.EW)
            
        top_frame.columnconfigure(0, weight=1)
        top_frame.columnconfigure(1, weight=1)
        
        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        bottom_left = ttk.Frame(bottom_frame)
        bottom_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        bottom_right = ttk.Frame(bottom_frame)
        bottom_right.pack(side=tk.RIGHT, fill=tk.Y, padx=20)
        
        self.calc_btn = ttk.Button(bottom_right, text="▶ Start Calculation", command=self.start_calculation)
        self.calc_btn.pack(fill=tk.X, pady=(0, 10), ipady=8)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(bottom_right, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        self.status_label = ttk.Label(bottom_right, text="Ready")
        self.status_label.pack(fill=tk.X, pady=(0, 20))
        
        export_frame = ttk.LabelFrame(bottom_right, text="Export Options", padding=(10, 10))
        export_frame.pack(fill=tk.BOTH, expand=True)
        
        self.export_vars = {
            "Curvature (1/m)": tk.BooleanVar(value=True),
            "Moment (kN·m)": tk.BooleanVar(value=True),
            "Comp. Zone Height (mm)": tk.BooleanVar(value=True),
            "Concrete Edge Comp. Strain": tk.BooleanVar(value=True),
            "Tension Rebar Strain": tk.BooleanVar(value=True),
            "Comp. Rebar Stress (MPa)": tk.BooleanVar(value=False),
            "Tension Rebar Stress (MPa)": tk.BooleanVar(value=False),
        }
        
        for name, var in self.export_vars.items():
            ttk.Checkbutton(export_frame, text=name, variable=var).pack(anchor=tk.W, pady=5)
            
        self.export_btn = ttk.Button(bottom_right, text="💾 Export to Excel", command=self.export_excel, state=tk.DISABLED)
        self.export_btn.pack(fill=tk.X, pady=20, ipady=8)
        
        self.fig, self.ax = plt.subplots(figsize=(4, 4))
        self.ax.set_xlabel("Curvature (1/m)", fontsize=12)
        self.ax.set_ylabel("Moment (kN·m)", fontsize=12)
        self.ax.set_title("Moment-Curvature Curve", fontsize=14)
        self.ax.tick_params(axis='both', which='major', labelsize=11)
        self.ax.grid(True)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=bottom_left)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def on_concrete_type_change(self, event):
        ctype = self.params["concrete_type"].get()
        if ctype == "NSC":
            self.params["ecu"].set("0.0033")
        else:
            try:
                fc = float(self.params["fc"].get())
                e0 = (377 * math.sqrt(fc) - 923) * 1e-6
                self.params["ecu"].set(str(round(3 * e0, 6)))
            except:
                self.params["ecu"].set("0.01")
                
    def start_calculation(self):
        self.calc_btn.config(state=tk.DISABLED)
        self.export_btn.config(state=tk.DISABLED)
        self.progress_var.set(0)
        self.status_label.config(text="Calculating...")
        self.results_df = None
        self.ax.clear()
        self.ax.set_xlabel("Curvature (1/m)", fontsize=12)
        self.ax.set_ylabel("Moment (kN·m)", fontsize=12)
        self.ax.set_title("Moment-Curvature Curve", fontsize=14)
        self.ax.tick_params(axis='both', which='major', labelsize=11)
        self.ax.grid(True)
        self.canvas.draw()
        
        threading.Thread(target=self.run_analysis, daemon=True).start()
        
    def run_analysis(self):
        try:
            b = float(self.params["b"].get())
            h = float(self.params["h"].get())
            cover = float(self.params["cover"].get())
            
            concrete_type = self.params["concrete_type"].get()
            fc = float(self.params["fc"].get())
            ecu = float(self.params["ecu"].get())
            n = float(self.params["n"].get())
            
            As1 = float(self.params["As1"].get())
            Es1 = float(self.params["Es1"].get())
            fy1 = float(self.params["fy1"].get())
            fsu1 = float(self.params["fsu1"].get())
            esu1 = float(self.params["esu1"].get())
            
            As2 = float(self.params["As2"].get())
            Es2 = float(self.params["Es2"].get())
            fy2 = float(self.params["fy2"].get())
            fsu2 = float(self.params["fsu2"].get())
            esu2 = float(self.params["esu2"].get())
            
            h0 = h - cover
            ys1 = (h - 2 * cover) / 2
            ys2 = -(h - 2 * cover) / 2
            k = 1000
            
            if concrete_type == "UHPC":
                ft = 2.14 * math.sqrt(fc) - 12.8
                e0 = (377 * math.sqrt(fc) - 923) * 1e-6
                et0 = 22.9 * ft * 1e-6
                etu = 0.015
            else:
                ft = 0.26 * (fc ** (2/3))
                e0 = (700 + 172 * math.sqrt(fc)) * 1e-6
                et0 = 65 * (ft ** 0.54) * 1e-6
                etu = 0.005
            
            ey1 = fy1 / Es1
            ey2 = fy2 / Es2
            Esh1 = (fsu1 - fy1) / (esu1 - ey1) if esu1 != ey1 else 0
            Esh2 = (fsu2 - fy2) / (esu2 - ey2) if esu2 != ey2 else 0
            
            dphi = 1.0e-6
            N = n * fc * b * h
            
            phi_list = [0.0]
            Mz_list = [0.0]
            x_list = [0.0]
            eck_list = [0.0]
            es_list = [0.0]
            sigmas1_list = [0.0]
            sigmas2_list = [0.0]
            
            max_steps = 50000
            tolN = 1.0
            
            for j in range(2, max_steps + 1):
                phi_j = phi_list[-1] + dphi
                
                x_max = 5 * h
                dx = 1.0
                x1 = 0.0
                Nz1 = calculate_axial_force(
                    x1, phi_j, h, k, ys1, ys2, As1, As2, b, fc, ecu, e0, ft, et0, etu,
                    fy1, Es1, ey1, fsu1, esu1, Esh1, fy2, Es2, ey2, fsu2, esu2, Esh2, concrete_type
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
                            x2, phi_j, h, k, ys1, ys2, As1, As2, b, fc, ecu, e0, ft, et0, etu,
                            fy1, Es1, ey1, fsu1, esu1, Esh1, fy2, Es2, ey2, fsu2, esu2, Esh2, concrete_type
                        )
                        F2 = Nz2 - N
                        if F1 * F2 <= 0:
                            found_bracket = True
                            break
                        x1 = x2
                        F1 = F2
                        
                if not found_bracket:
                    break
                    
                iter_count = 0
                max_iter = 100
                if abs(F1) > tolN:
                    while iter_count < max_iter:
                        x = (x1 + x2) / 2
                        Nz = calculate_axial_force(
                            x, phi_j, h, k, ys1, ys2, As1, As2, b, fc, ecu, e0, ft, et0, etu,
                            fy1, Es1, ey1, fsu1, esu1, Esh1, fy2, Es2, ey2, fsu2, esu2, Esh2, concrete_type
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
                        
                eck = x * phi_j
                epsilons2_val = (x - h / 2 + ys2) * phi_j
                
                Mz_j, s1_j, s2_j = calculate_moment(
                    x, phi_j, h, k, ys1, ys2, As1, As2, b, fc, ecu, e0, ft, et0, etu,
                    fy1, Es1, ey1, fsu1, esu1, Esh1, fy2, Es2, ey2, fsu2, esu2, Esh2, concrete_type
                )
                
                phi_list.append(phi_j)
                Mz_list.append(Mz_j)
                x_list.append(x)
                eck_list.append(eck)
                es_list.append(epsilons2_val)
                sigmas1_list.append(s1_j)
                sigmas2_list.append(s2_j)
                
                progress = min(100, (eck / (1.0 * ecu)) * 100)
                self.root.after(0, self.progress_var.set, progress)
                
                if eck >= 1.0 * ecu:
                    break
                    
            data = {
                "Curvature (1/m)": np.array(phi_list) * 1000,
                "Moment (kN·m)": np.array(Mz_list) / 1e6,
                "Comp. Zone Height (mm)": x_list,
                "Concrete Edge Comp. Strain": eck_list,
                "Tension Rebar Strain": es_list,
                "Comp. Rebar Stress (MPa)": sigmas1_list,
                "Tension Rebar Stress (MPa)": sigmas2_list
            }
            self.results_df = pd.DataFrame(data)
            
            self.root.after(0, self.plot_results)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"An error occurred during calculation:\n{str(e)}"))
            self.root.after(0, self.reset_ui)
            
    def plot_results(self):
        if self.results_df is not None and not self.results_df.empty:
            self.ax.plot(
                self.results_df["Curvature (1/m)"], 
                self.results_df["Moment (kN·m)"], 
                color='lightcoral',
                linestyle='-',
                linewidth=1.5,
                marker='o',
                markerfacecolor='none',
                markeredgecolor='lightcoral',
                markeredgewidth=1.5,
                markersize=4
            )
            self.ax.set_xlabel("Curvature (1/m)", fontsize=12)
            self.ax.set_ylabel("Moment (kN·m)", fontsize=12)
            self.ax.set_title("Moment-Curvature Curve", fontsize=14)
            self.ax.tick_params(axis='both', which='major', labelsize=11)
            self.ax.grid(True)
            self.canvas.draw()
            
            self.status_label.config(text="Calculation Completed!")
            self.export_btn.config(state=tk.NORMAL)
        else:
            self.status_label.config(text="Calculation Failed or No Data.")
            
        self.calc_btn.config(state=tk.NORMAL)
        self.progress_var.set(100)
        
    def reset_ui(self):
        self.calc_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Ready")
        self.progress_var.set(0)
        
    def export_excel(self):
        if self.results_df is None or self.results_df.empty:
            messagebox.showwarning("Warning", "No data to export. Please run calculation first!")
            return
            
        selected_columns = []
        for col, var in self.export_vars.items():
            if var.get():
                selected_columns.append(col)
                
        if not selected_columns:
            messagebox.showwarning("Warning", "Please select at least one parameter to export!")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel File", "*.xlsx")],
            title="Save Calculation Results"
        )
        
        if file_path:
            try:
                export_df = self.results_df[selected_columns]
                export_df.to_excel(file_path, index=False)
                messagebox.showinfo("Success", f"Data successfully exported to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = MNPHIGUI(root)
    root.mainloop()