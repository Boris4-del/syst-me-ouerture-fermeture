#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interface graphique pour l'étude dynamique d'un train d'atterrissage commandé par vérin.
Permet à l'utilisateur de saisir les paramètres géométriques et physiques,
de lancer la simulation numérique et d'afficher les courbes d'évolution de l'angle
et de la vitesse angulaire.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, messagebox
import sys


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Simulation dynamique - Train d'atterrissage")
        self.geometry("1200x700")
        self.resizable(True, True)

        # Style
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Variables des paramètres (avec valeurs par défaut)
        self.var_L = tk.DoubleVar(value=1.0)
        self.var_e = tk.DoubleVar(value=0.1)
        self.var_d = tk.DoubleVar(value=0.5)
        self.var_a = tk.DoubleVar(value=0.8)
        self.var_h = tk.DoubleVar(value=0.2)
        self.var_m1 = tk.DoubleVar(value=50.0)
        self.var_IG1 = tk.DoubleVar(value=10.0)
        self.var_F = tk.DoubleVar(value=5000.0)
        self.var_g = tk.DoubleVar(value=9.81)
        self.var_tmax = tk.DoubleVar(value=2.0)

        # Création de l'interface
        self.create_widgets()

        # Simulation par défaut au démarrage
        self.simuler()

    def create_widgets(self):
        # Frame principal
        main_frame = ttk.Frame(self, padding="5")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Frame gauche pour les paramètres
        left_frame = ttk.LabelFrame(main_frame, text="Paramètres du système", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5)

        # Création des lignes de saisie
        row = 0
        params = [
            ("L (m) :", self.var_L),
            ("e (m) :", self.var_e),
            ("d (m) :", self.var_d),
            ("a (m) :", self.var_a),
            ("h (m) :", self.var_h),
            ("m1 (kg) :", self.var_m1),
            ("IG1 (kg.m²) :", self.var_IG1),
            ("F (N) :", self.var_F),
            ("g (m/s²) :", self.var_g),
            ("Temps max (s) :", self.var_tmax)
        ]

        for label, var in params:
            ttk.Label(left_frame, text=label).grid(row=row, column=0, sticky=tk.W, pady=3)
            entry = ttk.Entry(left_frame, textvariable=var, width=15)
            entry.grid(row=row, column=1, padx=5, pady=3)
            row += 1

        # Boutons de contrôle
        btn_frame = ttk.Frame(left_frame)
        btn_frame.grid(row=row, column=0, columnspan=2, pady=15)

        ttk.Button(btn_frame, text="Lancer simulation", command=self.simuler).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Réinitialiser", command=self.reset).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Quitter", command=self.quit).pack(side=tk.LEFT, padx=5)

        # Frame droite pour les graphiques
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Création des figures matplotlib
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.fig.subplots_adjust(hspace=0.4)

        # Sous-graphique pour l'angle
        self.ax1 = self.fig.add_subplot(211)
        self.ax1.set_xlabel("Temps (s)")
        self.ax1.set_ylabel("θ (rad)")
        self.ax1.grid(True)
        self.ax1.set_title("Évolution de l'angle du bras")

        # Sous-graphique pour la vitesse angulaire
        self.ax2 = self.fig.add_subplot(212)
        self.ax2.set_xlabel("Temps (s)")
        self.ax2.set_ylabel("dθ/dt (rad/s)")
        self.ax2.grid(True)
        self.ax2.set_title("Évolution de la vitesse angulaire")

        # Intégration dans tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Barre d'état
        self.status = ttk.Label(self, text="Prêt", relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def reset(self):
        """Remet les valeurs par défaut."""
        self.var_L.set(1.0)
        self.var_e.set(0.1)
        self.var_d.set(0.5)
        self.var_a.set(0.8)
        self.var_h.set(0.2)
        self.var_m1.set(50.0)
        self.var_IG1.set(10.0)
        self.var_F.set(5000.0)
        self.var_g.set(9.81)
        self.var_tmax.set(2.0)
        self.simuler()

    def simuler(self):
        """Lance la simulation avec les paramètres courants."""
        try:
            # Récupération des valeurs
            L = self.var_L.get()
            e = self.var_e.get()
            d = self.var_d.get()
            a = self.var_a.get()
            h = self.var_h.get()
            m1 = self.var_m1.get()
            IG1 = self.var_IG1.get()
            F = self.var_F.get()
            g = self.var_g.get()
            tmax = self.var_tmax.get()

            # Vérifications de base
            if L <= 0 or e < 0 or d < 0 or m1 <= 0 or IG1 <= 0 or tmax <= 0:
                raise ValueError("Certains paramètres doivent être positifs (L, m1, IG1, tmax > 0).")

            # Calcul du moment d'inertie total
            IO = IG1 + m1 * d ** 2

            # Définition des fonctions
            def u(theta):
                X = L * np.cos(theta) + e * np.sin(theta) - a
                Y = L * np.sin(theta) - e * np.cos(theta) + h
                return np.sqrt(X ** 2 + Y ** 2)

            def moment_verin(theta):
                # Éviter division par zéro (u > 0 car distances non nulles)
                return (F / u(theta)) * ((h * e + a * L) * np.sin(theta) - (a * e - h * L) * np.cos(theta))

            def f(t, y):
                theta, theta_dot = y
                dtheta_dot = (-m1 * g * d * np.cos(theta) + moment_verin(theta)) / IO
                return [theta_dot, dtheta_dot]

            # Résolution numérique
            t_eval = np.linspace(0, tmax, 300)
            sol = solve_ivp(f, [0, tmax], [0, 0], method='RK45', t_eval=t_eval)

            if not sol.success:
                raise RuntimeError("La résolution numérique a échoué.")

            # Mise à jour des graphiques
            self.ax1.clear()
            self.ax1.plot(sol.t, sol.y[0], 'b-', linewidth=2)
            self.ax1.set_xlabel("Temps (s)")
            self.ax1.set_ylabel("θ (rad)")
            self.ax1.grid(True)
            self.ax1.set_title("Évolution de l'angle du bras")

            self.ax2.clear()
            self.ax2.plot(sol.t, sol.y[1], 'b-', linewidth=2)
            self.ax2.set_xlabel("Temps (s)")
            self.ax2.set_ylabel("dθ/dt (rad/s)")
            self.ax2.grid(True)
            self.ax2.set_title("Évolution de la vitesse angulaire")

            self.canvas.draw()

            # Mise à jour de la barre d'état
            self.status.config(text=f"Simulation terminée (tmax={tmax}s, {len(sol.t)} points)")

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la simulation :\n{str(e)}")
            self.status.config(text="Erreur")


if __name__ == "__main__":
    app = Application()
    app.mainloop()