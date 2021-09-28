#!/usr/bin/env python
"""
Aerodynamics of a sphere
------------------------

This script implements a simulation of a spherical projectile such as a soccer
ball, with air resistance.

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

version = 2.0


def reynolds_number(v, rho, eta, D):
    """
    Reynolds number as a function of velocity (`v`), diameter of the ball (`D`)
    and the air parameters: density (`rho`) and viscosity (`eta`).
    """
    return (rho*D*v)/eta


def drag_coefficient(re):
    """
    Drag coefficient in function of Reynolds number (`re`), for a spherical
    object obtained on Morrison 2016.
    """

    # Breaking into terms
    t1 = 24/re
    t2 = 2.6*(re/5.0)/(1 + (re/5.0)**1.52)
    t3 = 0.411*(re/2.63e5)**-7.94/(1+(re/2.63e5)**-8.00)
    t4 = 0.25*(re/10**6)/(1 + (re/10**6))

    return t1 + t2 + t3 + t4


def Fd(vx, vy, rho, eta, D):
    """
    Drag force as used in Morrison 2016:

    F_d = - (0.5 * C_d * rho * pi*(D/2)^2 * v^2) ṽ
    """

    v = np.sqrt(vx**2 + vy**2)

    Re = reynolds_number(v, rho, eta, D)
    Cd = drag_coefficient(Re)

    Fdx = - 0.5 * Cd * rho * np.pi*(D/2)**2 * vx**2
    Fdy = - 0.5 * Cd * rho * np.pi*(D/2)**2 * vy**2

    return [Fdx, Fdy]


def Fm(vx, vy, omega, rho, D):
    """
    Magnus force with angular velocity (`omega`) defined on the direction
    perpendicular to the xy plane so that it appears on the plane. Using the
    approximation on Aguiar & Rubini (2004):

    Fm = 0.5 * Cm * rho * pi * (D/2)^2 * D/2 * W <vec> V

    Where:
        W  = (  0,   0, omega)
        V  = (vx , vy , 0)
        Cm = 1
    """

    Fmx = 0.5 * rho * np.pi*(D/2)**2 * (D/2) * (-omega*vy)
    Fmy = 0.5 * rho * np.pi*(D/2)**2 * (D/2) * (omega*vx)

    return [Fmx, Fmy]


def simu(v0, theta_d, simu_pars,
         save_results=False, vizualize_results=True, ideal=False):
    """
    Main simulation.
    """

    theta = np.radians(theta_d)  # radians

    #  Decomposing initial movement
    v0x = v0*np.cos(theta)    # m/s
    v0y = v0*np.sin(theta)    # m/s

    Re0 = reynolds_number(v0, simu_pars["rho"], simu_pars["eta"], simu_pars["D"])
    Cd0 = drag_coefficient(Re0)

    Fdx0, Fdy0 = Fd(v0x, v0y, simu_pars["rho"], simu_pars["eta"], simu_pars["D"])    # N
    Fmx0, Fmy0 = Fm(v0x, v0y, simu_pars["omega"], simu_pars["rho"], simu_pars["D"])  # N

    a0x = (Fdx0 + Fmx0)/simu_pars["M"]      # m/s^2
    a0y = (Fdy0 + Fmy0)/simu_pars["M"] + simu_pars["g"]  # m/s^2

    #  Idealized case

    tt = np.arange(simu_pars["t0"], simu_pars["tf"], simu_pars["dt"])  # s

    x_a = v0x*tt
    y_a = v0y*tt + 0.5*simu_pars["g"]*tt**2

    #  Euler's Method

    #    Initializing vectors

    Re = np.zeros(tt.shape)
    Cd = np.zeros(tt.shape)

    x = np.zeros(tt.shape)
    y = np.zeros(tt.shape)

    vx = np.zeros(tt.shape)
    vy = np.zeros(tt.shape)
    v = np.zeros(tt.shape)

    axx = np.zeros(tt.shape)
    ay = np.zeros(tt.shape)
    a = np.zeros(tt.shape)

    #    Setting initial conditions

    Re[0] = Re0
    Cd[0] = Cd0

    x[0] = 0
    y[0] = 0

    vx[0] = v0x
    vy[0] = v0y
    v[0] = v0

    axx[0] = a0x
    ay[0] = a0y
    a[0] = np.sqrt(a0x**2 + a0y**2)

    for i in range(1, len(x)):
        # Evolving in space
        x[i] = x[i - 1] + vx[i - 1]*simu_pars["dt"]
        y[i] = y[i - 1] + vy[i - 1]*simu_pars["dt"]

        # Evolving velocity
        vx[i] = vx[i - 1] + axx[i - 1]*simu_pars["dt"]
        vy[i] = vy[i - 1] + ay[i - 1]*simu_pars["dt"]

        # Calculating new forces
        Fdx, Fdy = Fd(vx[i], vy[i], simu_pars["rho"], simu_pars["eta"], simu_pars["D"])
        Fmx, Fmy = Fm(vx[i], vy[i], simu_pars["omega"], simu_pars["rho"], simu_pars["D"])

        # Evolving acceleration
        axx[i] = (Fdx + Fmx)/simu_pars["M"]
        ay[i] = (Fdy + Fmy)/simu_pars["M"] + simu_pars["g"]

        # Stuff to plot later
        v[i]  = np.sqrt(vx[i]**2 + vy[i]**2)
        a[i]  = np.sqrt(axx[i]**2 + ay[i]**2)
        Re[i] = reynolds_number(v[i], simu_pars["rho"], simu_pars["eta"], simu_pars["D"])
        Cd[i] = drag_coefficient(Re[i])

    #  Estimating trajectory parameters

    #    Maximum height
    y_max_index = y.argmax()
    y_max = y[y_max_index]
    y_max_x = x[y_max_index]

    # print(f"Point of max: (x, y) = ({y_max_x:0.2f}, {y_max:0.2f}) m")

    #    Point of return

    y_return_index = np.where(y <= 0)[0][1]
    y_return = y[y_return_index]
    y_return_x = x[y_return_index]
    y_return_t = tt[y_return_index]

    # print(f"Return time: {y_return_t:0.2f} s")
    # print(f"Horizontal range: {y_return_x:0.2f} m")

    # Packaging data for returning

    results = {
                "time": tt,
                "x_pos": x,
                "y_pos": y,
                "x_vel": vx,
                "y_vel": vy,
                "speed": v,
                "x_acc": axx,
                "y_acc": ay,
                "accel": a,
                "Re": Re,
                "Cd": Cd,
                "x_ideal": x_a,
                "y_ideal": y_a
            }

    results = pd.DataFrame(results)

    trajectory_pars = {
            "Initial angle": theta_d,
            "Initial speed": v0,
            "Horizontal reach": y_return_x,
            "Maximum height": y_max,
            "Time of flight": y_return_t
           }


    return [trajectory_pars, results]


if __name__ == "__main__":

    #  Options

    save_results = False

    results_root = Path("./results")
    save_folder = results_root / Path(f"results_v{version}")
    save_plots = save_folder / Path("plots")
    save_tables = save_folder / Path("tables")

    #  Initial Conditions

    v0 = 29.1       # m/s
    theta_d = 17.7  # degrees
    g = -9.81       # m/s^2

    #  Projectile parameters
    M = 0.454    # kg
    D = 0.222    # m
    omega = 6.8  # rad/s

    #  Air parameters
    eta = 1.83e-5  # kg/(m s)
    rho = 1.05     # kg/m^3

    #  Time interval

    t0 = 0     # s
    tf = 5     # s
    dt = 1e-4  # s

    #  Encapsulating parameters in a dict so i can save them easily later
    simulation_parameters = {
            "g": g, "M": M, "D": D, "omega": omega, "eta": eta, "rho": rho,
            "t0": t0, "tf": tf, "dt": dt
            }

    pars, res = simu(v0, theta_d, simulation_parameters)

    #  Plotting Results

    fig, ax = plt.subplots(dpi=150)

    mask = (res.y_pos >= 0)

    ax.set_title("trajectory")
    ax.plot(res.x_ideal[mask], res.y_ideal[mask], label="ideal case")
    ax.plot(res.x_pos[mask], res.y_pos[mask], label="air drag + magnus")
    ax.legend()
    ax.set_ylim(0)
    ax.set_xlabel("x position / [m]")
    ax.set_ylabel("y position / [m]")

    fig.tight_layout()

    fig.show()

    # Saving results

    if save_results:
        # Checking if folders are in place
        if not save_plots.exists():
            os.makedirs(save_plots)

        if not save_tables.exists():
            os.makedirs(save_tables)

        # Filenames
        file_stem = f"launch_v{version}_v0={v0}_theta={theta_d}_dt={dt}"

        with open(f"{save_folder}/{file_stem}.log", "w") as log:

            log.write("Simulation parameters:\n\n")

            for i in simulation_parameters:
                log.write(f"{i} = {simulation_parameters[i]}\n")

            log.write(f"Simulation results:\n\n"
                      f"Max height       = {y_max:6.2f} m\n"
                      f"Horizontal reach = {y_return_x:6.2f} m\n"
                      f"Flight time      = {y_return_t:6.2f} s")

        # Putting results in a single table

        # Saving
        res.to_csv(save_tables / (file_stem + ".csv"), index=False)
        fig.savefig(save_plots / (file_stem + ".png"), dpi=150)
