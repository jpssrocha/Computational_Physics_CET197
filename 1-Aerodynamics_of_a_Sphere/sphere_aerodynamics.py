#!/usr/bin/env python
"""

Aerodynamics of a ball
----------------------


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
    Drag coefficient in function of reynolds number (`re`), for a spherical
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

    Fdx = 0.5 * Cd * rho * np.pi*(D/2)**2 * vx**2
    Fdy = 0.5 * Cd * rho * np.pi*(D/2)**2 * vy**2

    return (-Fdx, -Fdy)


def Fm(v, rho, eta, D):
    """
    Magnus force
    """

    raise NotImplementedError


if __name__ == "__main__":

    #  Options

    save_results = False

    results_root = Path("./results")
    save_folder = results_root / Path(f"results_v{version}")
    save_plots = save_folder / Path("plots")
    save_tables = save_folder / Path("tables")

    #  Initial Conditions

    v0 = 30                  # m/s
    theta_d = 30                  # degrees
    theta = np.radians(theta_d)  # radians
    g = -9.81               # m/s^2

    #  Projectile parameters
    M = 0.454  # kg
    D = 0.222  # m

    #  Air parameters
    eta = 1.83e-5  # kg/(m s)
    rho = 1.224    # kg/m^3

    #  Time interval

    t0 = 0      # s
    tf = 10     # s
    dt = 0.001  # s

    #  Decomposing movement

    v0x = v0*np.cos(theta)    # m/s
    v0y = v0*np.sin(theta)    # m/s

    Fdx, Fdy = Fd(v0x, v0y, rho, eta, D)

    a0x = Fdx/M
    a0y = Fdy/M + g

    #  Analytical solution

    tt = np.arange(t0, tf, dt)

    x_a = v0x*tt
    y_a = v0y*tt + 0.5*g*tt**2

    mask = y_a >= 0  # For plotting

    #  Euler's Method

    #    Initializing vectors

    ax = np.zeros(tt.shape)
    ay = np.zeros(tt.shape)

    vx = np.zeros(tt.shape)
    vy = np.zeros(tt.shape)

    x = np.zeros(tt.shape)
    y = np.zeros(tt.shape)

    #    Setting initial conditions

    x[0] = 0
    y[0] = 0

    vx[0] = v0x
    vy[0] = v0y

    ax[0] = a0x
    ay[0] = a0y

    for i in range(1, len(x)):
        # Evolving in space
        x[i] = x[i - 1] + vx[i - 1]*dt
        y[i] = y[i - 1] + vy[i - 1]*dt

        # Evolving velocity
        vx[i] = vx[i - 1] + ax[i - 1]*dt
        vy[i] = vy[i - 1] + ay[i - 1]*dt

        # Evolving acceleration
        Fdx, Fxy = Fd(vx[i], vy[i], rho, eta, D)
        ax[i] = Fdx/M
        ay[i] = Fdy/M + g

    #  Estimating trajectory parameters

    #    Maximum height

    y_max_index = np.where(y == y.max())[0][0]
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

    fig.suptitle(f"Ideal case vs Air resistance (dt={dt})")

    ax.set_title("Trajectory")
    ax.plot(x_a[mask], y_a[mask], label="Ideal case")
    ax.plot(x[mask], y[mask], label="Air resistance")
    ax.legend()

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
