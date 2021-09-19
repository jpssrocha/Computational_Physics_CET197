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


if __name__ == "__main__":

    #  Options

    save_results = False

    results_root = Path("./results")
    save_folder = results_root / Path(f"results_v{version}")
    save_plots = save_folder / Path("plots")
    save_tables = save_folder / Path("tables")

    #  Initial Conditions

    v0 = 29.1                  # m/s
    theta_d = 17.7                  # degrees
    theta = np.radians(theta_d)  # radians
    g = -9.81               # m/s^2

    #  Projectile parameters
    M = 0.454   # kg
    D = 0.222   # m
    omega = 6.8  # rad/s

    #  Air parameters
    eta = 1.83e-5  # kg/(m s)
    rho = 1.05    # kg/m^3

    #  Time interval

    t0 = 0      # s
    tf = 5     # s
    dt = 1e-4  # s

    #  Decomposing movement

    v0x = v0*np.cos(theta)    # m/s
    v0y = v0*np.sin(theta)    # m/s

    Re0 = reynolds_number(v0, rho, eta, D)
    Cd0 = drag_coefficient(Re0)

    Fdx0, Fdy0 = Fd(v0x, v0y, rho, eta, D)
    Fmx0, Fmy0 = Fm(v0x, v0y, omega, rho, D)

    a0x = (Fdx0 + Fmx0)/M
    a0y = (Fdy0 + Fmy0)/M + g

    #  Idealized case

    tt = np.arange(t0, tf, dt)

    x_a = v0x*tt
    y_a = v0y*tt + 0.5*g*tt**2

    #  Euler's Method

    #    Initializing vectors

    Re = np.zeros(tt.shape)
    Cd = np.zeros(tt.shape)

    x = np.zeros(tt.shape)
    y = np.zeros(tt.shape)

    vx = np.zeros(tt.shape)
    vy = np.zeros(tt.shape)
    v  = np.zeros(tt.shape)

    axx = np.zeros(tt.shape)
    ay = np.zeros(tt.shape)
    a  = np.zeros(tt.shape)

    #    Setting initial conditions

    Re[0] = Re0
    Cd[0] = Cd0

    x[0] = 0
    y[0] = 0

    vx[0] = v0x
    vy[0] = v0y
    v[0] = v0

    axx[0] = a0x
    ay[0]  = a0y
    a[0]   = np.sqrt(a0x**2 + a0y**2)

    for i in range(1, len(x)):
        # Evolving in space
        x[i] = x[i - 1] + vx[i - 1]*dt
        y[i] = y[i - 1] + vy[i - 1]*dt

        # Evolving velocity
        vx[i] = vx[i - 1] + axx[i - 1]*dt
        vy[i] = vy[i - 1] + ay[i - 1]*dt

        # Calculating new forces
        Fdx, Fdy = Fd(vx[i], vy[i], rho, eta, D)
        Fmx, Fmy = Fm(vx[i], vy[i], omega, rho, D)

        # Evolving acceleration
        axx[i] = (Fdx + Fmx)/M
        ay[i] = (Fdy + Fmy)/M + g

        # Stuff to plot later
        v[i]  = np.sqrt(vx[i]**2 + vy[i]**2)
        a[i]  = np.sqrt(axx[i]**2 + ay[i]**2)
        Re[i] = reynolds_number(v[i], rho, eta, D)
        Cd[i] = drag_coefficient(Re[i])

    #  Estimating trajectory parameters

    #    Maximum height
    y_max_index = y.argmax()
    y_max = y[y_max_index]
    y_max_x = x[y_max_index]

    print(f"Point of max: (x, y) = ({y_max_x:0.2f}, {y_max:0.2f}) m")

    #    Point of return

    y_return_index = np.where(y <= 0)[0][1]
    y_return = y[y_return_index]
    y_return_x = x[y_return_index]
    y_return_t = tt[y_return_index]

    print(f"Return time: {y_return_t:0.2f} s")
    print(f"Horizontal range: {y_return_x:0.2f} m")

    #  Plotting Results

    fig, ax = plt.subplots(dpi=150)

    # fig.suptitle(f"Ideal case vs Air resistance (dt={dt})")

    mask = (y >= 0)

    ax.set_title("Trajectory")
    ax.plot(x_a[mask], y_a[mask], label="Ideal case")
    ax.plot(x[mask], y[mask], label="Air drag + Magnus")
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

            log.write(f"Max height       = {y_max:6.2f} m\n"
                      f"Horizontal reach = {y_return_x:6.2f} m\n"
                      f"Flight time      = {y_return_t:6.2f} s")

        # Putting results in a single table
        data_dict = {"time": tt, "x_pos": x, "y_pos": y}
        data_table = pd.DataFrame(data_dict)

        # Saving
        data_table.to_csv(save_tables / (file_stem + ".csv"), index=False)
        fig.savefig(save_plots / (file_stem + ".png"), dpi=150)
