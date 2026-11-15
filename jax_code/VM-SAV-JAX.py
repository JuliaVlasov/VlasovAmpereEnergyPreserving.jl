#!/usr/bin/env python3
"""Paper-derived semi-Lagrangian SAV solvers for the examples in Sections 6.1–6.3."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

try:
    import jax
    import jax.numpy as jnp
    from jax import lax
except ImportError as exc:
    raise SystemExit(
        "JAX is required. Install the isolated dependencies with "
        "`python -m pip install -r requirements.txt` from jax_code/."
    ) from exc


# Use double precision for the SAV diagnostics and the Section 6.1 step sweep.
jax.config.update("jax_enable_x64", True)


HISTORY_COLUMNS = (
    "time",
    "gauss_l2",
    "kinetic_energy",
    "electric_energy_ex",
    "electric_energy_ey",
    "magnetic_energy",
    "modified_energy",
    "relative_modified_energy_error",
    "sav_error_ex",
    "sav_error_ey",
)


def spectral_derivative(values: jax.Array, wave_numbers: jax.Array) -> jax.Array:
    return jnp.fft.ifft(1j * wave_numbers * jnp.fft.fft(values)).real


def trapezoid_weights(grid: jax.Array) -> jax.Array:
    weights = jnp.ones_like(grid)
    weights = weights.at[0].set(0.5)
    return weights.at[-1].set(0.5) * (grid[1] - grid[0])


def interp1d_zero(
    query: jax.Array, grid: jax.Array, values: jax.Array
) -> jax.Array:
    index = jnp.clip(jnp.searchsorted(grid, query, side="right") - 1, 0, grid.size - 2)
    fraction = (query - grid[index]) / (grid[index + 1] - grid[index])
    result = values[index] * (1.0 - fraction) + values[index + 1] * fraction
    inside = (query >= grid[0]) & (query <= grid[-1])
    return jnp.where(inside, result, 0.0)


def lagrange_weights(fraction: jax.Array) -> jax.Array:
    return jnp.stack(
        (
            -fraction * (fraction - 1.0) * (fraction - 2.0) / 6.0,
            (fraction + 1.0) * (fraction - 1.0) * (fraction - 2.0) / 2.0,
            -(fraction + 1.0) * fraction * (fraction - 2.0) / 2.0,
            (fraction + 1.0) * fraction * (fraction - 1.0) / 6.0,
        ),
        axis=-1,
    )


def interp2d_zero(
    query_x: jax.Array,
    query_y: jax.Array,
    grid_x: jax.Array,
    grid_y: jax.Array,
    values: jax.Array,
) -> jax.Array:
    """Tensor cubic interpolation inside, bilinear near edges, zero outside."""
    shape = query_x.shape
    qx, qy = query_x.reshape(-1), query_y.reshape(-1)
    ix = jnp.clip(jnp.searchsorted(grid_x, qx, side="right") - 1, 0, grid_x.size - 2)
    iy = jnp.clip(jnp.searchsorted(grid_y, qy, side="right") - 1, 0, grid_y.size - 2)
    tx = (qx - grid_x[ix]) / (grid_x[ix + 1] - grid_x[ix])
    ty = (qy - grid_y[iy]) / (grid_y[iy + 1] - grid_y[iy])

    f00 = values[ix, iy]
    f10 = values[ix + 1, iy]
    f01 = values[ix, iy + 1]
    f11 = values[ix + 1, iy + 1]
    linear = (
        (1.0 - tx) * (1.0 - ty) * f00
        + tx * (1.0 - ty) * f10
        + (1.0 - tx) * ty * f01
        + tx * ty * f11
    )

    offsets = jnp.arange(-1, 3)
    stencil_x = ix[:, None] + offsets[None, :]
    stencil_y = iy[:, None] + offsets[None, :]
    stencil = values[stencil_x[:, :, None], stencil_y[:, None, :]]
    cubic = jnp.einsum(
        "pi,pij,pj->p",
        lagrange_weights(tx),
        stencil,
        lagrange_weights(ty),
    )

    interior = (
        (ix >= 1)
        & (ix <= grid_x.size - 3)
        & (iy >= 1)
        & (iy <= grid_y.size - 3)
    )
    inside_domain = (
        (qx >= grid_x[0])
        & (qx <= grid_x[-1])
        & (qy >= grid_y[0])
        & (qy <= grid_y[-1])
    )
    result = jnp.where(interior, cubic, linear)
    return jnp.where(inside_domain, result, 0.0).reshape(shape)


def solve_poisson(
    density: jax.Array, wave_numbers: jax.Array
) -> jax.Array:
    density_hat = jnp.fft.fft(density - jnp.mean(density))
    field_hat = jnp.where(
        wave_numbers != 0.0,
        density_hat / (1j * jnp.where(wave_numbers == 0.0, 1.0, wave_numbers)),
        0.0,
    )
    return jnp.fft.ifft(field_hat).real


def time_steps(final_time: float, dt: float) -> int:
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a positive finite number")
    if not math.isfinite(final_time) or final_time <= 0.0:
        raise ValueError("final time must be a positive finite number")
    count = round(final_time / dt)
    if count < 1 or not math.isclose(count * dt, final_time, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("final time must be an integer multiple of dt")
    return count


def build_case(
    case: str,
    *,
    nx: int | None,
    nv: int | None,
    nvx: int | None,
    nvy: int | None,
    k: float | None,
    alpha: float | None,
    beta: float | None,
    sigma1: float | None,
    sigma2: float | None,
    v_bounds: tuple[float, float] | None,
    vx_bounds: tuple[float, float] | None,
    vy_bounds: tuple[float, float] | None,
) -> dict[str, Any]:
    if case == "two-stream":
        settings: dict[str, Any] = {
            "nx": 128,
            "nvx": 128,
            "nvy": 1,
            "k": 0.5,
            "alpha": 0.05,
            "beta": 0.0,
            "sigma1": 1.0,
            "sigma2": 1.0,
            "vx_bounds": (-8.0, 8.0),
            "vy_bounds": (-1.0, 1.0),
            "dt": 0.5,
            "final_time": 100.0,
        }
    elif case == "weibel":
        thermal = 0.02 / math.sqrt(2.0)
        settings = {
            "nx": 16,
            "nvx": 47,
            "nvy": 47,
            "k": 1.25,
            "alpha": 0.0,
            "beta": -1e-4,
            "sigma1": thermal,
            "sigma2": math.sqrt(12.0) * thermal,
            "vx_bounds": (-0.2, 0.2),
            "vy_bounds": (-0.5, 0.5),
            "dt": 1.0 / 16.0,
            "final_time": 300.0,
        }
    elif case == "landau-weibel":
        settings = {
            "nx": 47,
            "nvx": 47,
            "nvy": 47,
            "k": 0.5,
            "alpha": 1e-3,
            "beta": -1e-6,
            "sigma1": 1.0,
            "sigma2": math.sqrt(2.0),
            "vx_bounds": (-6.0, 6.0),
            "vy_bounds": (-8.0, 8.0),
            "dt": 0.125,
            "final_time": 50.0,
        }
    else:
        raise ValueError(f"unknown case: {case}")

    settings["nx"] = nx if nx is not None else settings["nx"]
    settings["k"] = k if k is not None else settings["k"]
    settings["alpha"] = alpha if alpha is not None else settings["alpha"]
    settings["beta"] = beta if beta is not None else settings["beta"]
    settings["sigma1"] = sigma1 if sigma1 is not None else settings["sigma1"]
    if sigma2 is not None:
        settings["sigma2"] = sigma2
    elif sigma1 is not None and case == "weibel":
        settings["sigma2"] = math.sqrt(12.0) * settings["sigma1"]
    elif sigma1 is not None and case == "landau-weibel":
        settings["sigma2"] = math.sqrt(2.0) * settings["sigma1"]
    settings["nvx"] = nvx if nvx is not None else (nv if nv is not None else settings["nvx"])
    settings["nvy"] = nvy if nvy is not None else (nv if nv is not None else settings["nvy"])
    if v_bounds is not None:
        settings["vx_bounds"] = v_bounds
    if vx_bounds is not None:
        settings["vx_bounds"] = vx_bounds
    if vy_bounds is not None:
        settings["vy_bounds"] = vy_bounds

    if settings["nx"] < 4 or settings["nvx"] < 4:
        raise ValueError("nx and velocity grid sizes must be at least 4")
    if case != "two-stream" and settings["nvy"] < 4:
        raise ValueError("nvy must be at least 4")
    if settings["k"] <= 0.0:
        raise ValueError("k must be positive")
    return settings


def make_initial_state(
    case: str,
    settings: dict[str, Any],
    sav_shift: float,
) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    nx = settings["nx"]
    k = settings["k"]
    length = 2.0 * math.pi / k
    x = jnp.arange(nx, dtype=jnp.float64) * (length / nx)
    dx = length / nx
    wave_numbers = 2.0 * math.pi * jnp.fft.fftfreq(nx, d=dx)
    vx = jnp.linspace(*settings["vx_bounds"], settings["nvx"], dtype=jnp.float64)
    wx = trapezoid_weights(vx)

    if case == "two-stream":
        profile = vx**2 * jnp.exp(-0.5 * vx**2) / math.sqrt(2.0 * math.pi)
        profile = profile / jnp.sum(profile * wx)
        f = profile[None, :] * (
            1.0 + settings["alpha"] * jnp.cos(k * x[:, None])
        )
        density = jnp.sum(f * wx[None, :], axis=1)
        ex = solve_poisson(density, wave_numbers)
        ey = jnp.zeros_like(ex)
        magnetic = jnp.zeros_like(ex)
        state = {
            "f": f,
            "ex": ex,
            "ey": ey,
            "magnetic": magnetic,
            "rx": jnp.sqrt(0.5 * dx * jnp.sum(ex**2) + sav_shift),
            "ry": jnp.asarray(0.0, dtype=jnp.float64),
        }
        grids = {
            "x": x,
            "vx": vx,
            "wx": wx,
            "wave_numbers": wave_numbers,
            "dx": jnp.asarray(dx),
            "sav_shift": jnp.asarray(sav_shift),
            "mass": jnp.mean(density),
        }
        return state, grids

    vy = jnp.linspace(*settings["vy_bounds"], settings["nvy"], dtype=jnp.float64)
    wy = trapezoid_weights(vy)
    vx_mesh, vy_mesh = jnp.meshgrid(vx, vy, indexing="ij")
    profile = jnp.exp(
        -0.5
        * (
            (vx_mesh / settings["sigma1"]) ** 2
            + (vy_mesh / settings["sigma2"]) ** 2
        )
    )
    profile = profile / jnp.sum(profile * wx[:, None] * wy[None, :])
    spatial_modulation = 1.0 + settings["alpha"] * jnp.cos(k * x)
    f = profile[None, :, :] * spatial_modulation[:, None, None]
    density = jnp.sum(f * wx[None, :, None] * wy[None, None, :], axis=(1, 2))
    ex = solve_poisson(density, wave_numbers)
    ey = jnp.zeros_like(ex)
    magnetic = settings["beta"] * jnp.cos(k * x)
    state = {
        "f": f,
        "ex": ex,
        "ey": ey,
        "magnetic": magnetic,
        "rx": jnp.sqrt(0.5 * dx * jnp.sum(ex**2) + sav_shift),
        "ry": jnp.sqrt(0.5 * dx * jnp.sum(ey**2) + sav_shift),
    }
    grids = {
        "x": x,
        "vx": vx,
        "vy": vy,
        "wx": wx,
        "wy": wy,
        "vx_mesh": vx_mesh,
        "vy_mesh": vy_mesh,
        "wave_numbers": wave_numbers,
        "dx": jnp.asarray(dx),
        "sav_shift": jnp.asarray(sav_shift),
        "mass": jnp.mean(density),
    }
    return state, grids


def make_diagnostics(case: str, grids: dict[str, jax.Array]):
    dx = grids["dx"]
    wave_numbers = grids["wave_numbers"]
    sav_shift = grids["sav_shift"]
    mass = grids["mass"]

    def diagnostics(state: dict[str, jax.Array], time: jax.Array):
        if case == "two-stream":
            density = jnp.sum(state["f"] * grids["wx"][None, :], axis=1)
            kinetic = (
                0.5
                * dx
                * jnp.sum(
                    state["f"] * (grids["vx"] ** 2)[None, :] * grids["wx"][None, :]
                )
            )
        else:
            velocity_weights = grids["wx"][:, None] * grids["wy"][None, :]
            density = jnp.sum(state["f"] * velocity_weights[None, :, :], axis=(1, 2))
            kinetic = (
                0.5
                * dx
                * jnp.sum(
                    state["f"]
                    * (grids["vx_mesh"] ** 2 + grids["vy_mesh"] ** 2)[None, :, :]
                    * velocity_weights[None, :, :]
                )
            )
        ex_energy = 0.5 * dx * jnp.sum(state["ex"] ** 2)
        ey_energy = 0.5 * dx * jnp.sum(state["ey"] ** 2)
        magnetic_energy = 0.5 * dx * jnp.sum(state["magnetic"] ** 2)
        gauss = spectral_derivative(state["ex"], wave_numbers) - (density - mass)
        gauss_l2 = jnp.sqrt(dx * jnp.sum(gauss**2))
        if case == "two-stream":
            modified_energy = kinetic + state["rx"] ** 2
            sav_error_ex = jnp.abs(state["rx"] ** 2 - sav_shift - ex_energy)
            sav_error_ey = jnp.asarray(0.0, dtype=jnp.float64)
        else:
            modified_energy = (
                kinetic + state["rx"] ** 2 + state["ry"] ** 2 + magnetic_energy
            )
            sav_error_ex = jnp.abs(state["rx"] ** 2 - sav_shift - ex_energy)
            sav_error_ey = jnp.abs(state["ry"] ** 2 - sav_shift - ey_energy)
        return jnp.stack(
            (
                time,
                gauss_l2,
                kinetic,
                ex_energy,
                ey_energy,
                magnetic_energy,
                modified_energy,
                jnp.asarray(0.0),
                sav_error_ex,
                sav_error_ey,
            )
        )

    return diagnostics


def make_two_stream_step(grids: dict[str, jax.Array], dt: float):
    wave_numbers = grids["wave_numbers"]
    vx = grids["vx"]
    wx = grids["wx"]
    dx = grids["dx"]
    x_grid = vx

    def transport(state: dict[str, jax.Array], step_dt: float):
        f_hat = jnp.fft.fft(state["f"], axis=0)
        phase = jnp.exp(
            -1j * wave_numbers[:, None] * vx[None, :] * step_dt
        )
        f_new_hat = phase * f_hat
        f_new = jnp.fft.ifft(f_new_hat, axis=0).real
        density_change_hat = jnp.sum(
            (f_new_hat - f_hat) * wx[None, :], axis=1
        )
        ex_hat = jnp.fft.fft(state["ex"])
        safe_k = jnp.where(wave_numbers == 0.0, 1.0, wave_numbers)
        delta_ex_hat = density_change_hat / (1j * safe_k)
        current_hat_zero = jnp.sum(f_hat[0, :] * vx * wx)
        delta_ex_hat = delta_ex_hat.at[0].set(-step_dt * current_hat_zero)
        ex_new = jnp.fft.ifft(ex_hat + delta_ex_hat).real
        return {**state, "f": f_new, "ex": ex_new}

    def electric(state: dict[str, jax.Array], step_dt: float):
        electric = state["ex"]
        g = electric / state["rx"]
        current = jnp.sum(state["f"] * (vx * wx)[None, :], axis=1)
        density = jnp.sum(state["f"] * wx[None, :], axis=1)
        coefficient = dx * jnp.sum(g**2 * density)
        work = dx * jnp.sum(g * current)
        factor = step_dt**2 * coefficient / 8.0
        rx_new = (
            state["rx"] * (1.0 - factor) - 0.5 * step_dt * work
        ) / (1.0 + factor)
        rx_half = 0.5 * (state["rx"] + rx_new)
        shifts = step_dt * rx_half * g
        f_new = jax.vmap(
            lambda row, shift: interp1d_zero(x_grid - shift, x_grid, row)
        )(state["f"], shifts)
        return {**state, "f": f_new, "rx": rx_new}

    def step(state: dict[str, jax.Array], _):
        state = transport(state, 0.5 * dt)
        state = electric(state, dt)
        state = transport(state, 0.5 * dt)
        return state, state

    return step


def make_vm_step(grids: dict[str, jax.Array], dt: float):
    wave_numbers = grids["wave_numbers"]
    vx = grids["vx"]
    vy = grids["vy"]
    wx = grids["wx"]
    wy = grids["wy"]
    dx = grids["dx"]
    vx_mesh = grids["vx_mesh"]
    vy_mesh = grids["vy_mesh"]

    def kinetic_ampere(state: dict[str, jax.Array], step_dt: float):
        f_hat = jnp.fft.fft(state["f"], axis=0)
        phase = jnp.exp(-1j * wave_numbers[:, None, None] * vx[None, :, None] * step_dt)
        f_new_hat = phase * f_hat
        f_new = jnp.fft.ifft(f_new_hat, axis=0).real
        density_change_hat = jnp.sum(
            (f_new_hat - f_hat) * wx[None, :, None] * wy[None, None, :],
            axis=(1, 2),
        )
        ex_hat = jnp.fft.fft(state["ex"])
        safe_k = jnp.where(wave_numbers == 0.0, 1.0, wave_numbers)
        delta_ex_hat = density_change_hat / (1j * safe_k)
        current_hat_zero = jnp.sum(
            f_hat[0, :, :]
            * vx[:, None]
            * wx[:, None]
            * wy[None, :]
        )
        delta_ex_hat = delta_ex_hat.at[0].set(-step_dt * current_hat_zero)
        ex_new = jnp.fft.ifft(ex_hat + delta_ex_hat).real

        time_phase = jnp.exp(
            -0.5j * wave_numbers[:, None] * vx[None, :] * step_dt
        )
        time_average = jnp.sinc(
            wave_numbers[:, None] * vx[None, :] * step_dt / (2.0 * math.pi)
        )
        jy_integral_hat = step_dt * jnp.sum(
            f_hat
            * time_phase[:, :, None]
            * time_average[:, :, None]
            * vy[None, None, :]
            * wx[None, :, None]
            * wy[None, None, :],
            axis=(1, 2),
        )
        ey_new = jnp.fft.ifft(jnp.fft.fft(state["ey"]) - jy_integral_hat).real
        return {**state, "f": f_new, "ex": ex_new, "ey": ey_new}

    def electric_sav(state: dict[str, jax.Array], step_dt: float):
        density = jnp.sum(
            state["f"] * wx[None, :, None] * wy[None, None, :],
            axis=(1, 2),
        )
        current_x = jnp.sum(
            state["f"]
            * vx[None, :, None]
            * wx[None, :, None]
            * wy[None, None, :],
            axis=(1, 2),
        )
        current_y = jnp.sum(
            state["f"]
            * vy[None, None, :]
            * wx[None, :, None]
            * wy[None, None, :],
            axis=(1, 2),
        )
        gx = state["ex"] / state["rx"]
        gy = state["ey"] / state["ry"]
        derivative_gy = spectral_derivative(gy, wave_numbers)
        ax = dx * jnp.sum(gx**2 * density)
        work_x = dx * jnp.sum(gx * current_x)
        factor_x = step_dt**2 * ax / 8.0
        rx_new = (
            state["rx"] * (1.0 - factor_x) - 0.5 * step_dt * work_x
        ) / (1.0 + factor_x)

        ay_density = dx * jnp.sum(gy**2 * density)
        ay_derivative = dx * jnp.sum(derivative_gy**2)
        work_y = dx * jnp.sum(gy * current_y)
        magnetic_coupling = dx * jnp.sum(derivative_gy * state["magnetic"])
        factor_y = step_dt**2 * (ay_density + ay_derivative) / 8.0
        ry_new = (
            state["ry"] * (1.0 - factor_y)
            - 0.5 * step_dt * work_y
            + 0.5 * step_dt * magnetic_coupling
        ) / (1.0 + factor_y)

        rx_half = 0.5 * (state["rx"] + rx_new)
        ry_half = 0.5 * (state["ry"] + ry_new)
        shift_x = step_dt * rx_half * gx
        shift_y = step_dt * ry_half * gy
        magnetic_new = (
            state["magnetic"] - step_dt * ry_half * derivative_gy
        )

        def translate_slice(row, sx, sy):
            return interp2d_zero(
                vx_mesh - sx,
                vy_mesh - sy,
                vx,
                vy,
                row,
            )

        f_new = jax.vmap(translate_slice)(state["f"], shift_x, shift_y)
        return {
            **state,
            "f": f_new,
            "magnetic": magnetic_new,
            "rx": rx_new,
            "ry": ry_new,
        }

    def magnetic_rotation(state: dict[str, jax.Array], step_dt: float):
        angle = step_dt * state["magnetic"]
        cosine = jnp.cos(angle)
        sine = jnp.sin(angle)
        query_x = (
            vx_mesh[None, :, :] * cosine[:, None, None]
            + vy_mesh[None, :, :] * sine[:, None, None]
        )
        query_y = (
            -vx_mesh[None, :, :] * sine[:, None, None]
            + vy_mesh[None, :, :] * cosine[:, None, None]
        )

        def rotate_slice(row, qx, qy):
            return interp2d_zero(qx, qy, vx, vy, row)

        f_new = jax.vmap(rotate_slice)(state["f"], query_x, query_y)
        ey_new = state["ey"] - step_dt * spectral_derivative(
            state["magnetic"], wave_numbers
        )
        return {**state, "f": f_new, "ey": ey_new}

    def step(state: dict[str, jax.Array], _):
        state = magnetic_rotation(state, 0.5 * dt)
        state = electric_sav(state, 0.5 * dt)
        state = kinetic_ampere(state, dt)
        state = electric_sav(state, 0.5 * dt)
        state = magnetic_rotation(state, 0.5 * dt)
        return state, state

    return step


def run_simulation(
    case: str,
    settings: dict[str, Any],
    dt: float,
    final_time: float,
    sav_shift: float,
) -> tuple[dict[str, jax.Array], dict[str, jax.Array], jax.Array]:
    if sav_shift <= 0.0 or not math.isfinite(sav_shift):
        raise ValueError("SAV shift must be a positive finite number")
    count = time_steps(final_time, dt)
    initial_state, grids = make_initial_state(case, settings, sav_shift)
    diagnostics = make_diagnostics(case, grids)
    initial_history = diagnostics(initial_state, jnp.asarray(0.0))
    initial_energy = initial_history[6]

    def record(state: dict[str, jax.Array], time: jax.Array):
        values = diagnostics(state, time)
        values = values.at[7].set(
            jnp.abs(values[6] - initial_energy) / jnp.maximum(jnp.abs(initial_energy), 1e-300)
        )
        return values

    step = (
        make_two_stream_step(grids, dt)
        if case == "two-stream"
        else make_vm_step(grids, dt)
    )

    def recorded_step(state: dict[str, jax.Array], index: jax.Array):
        new_state, _ = step(state, index)
        return new_state, record(new_state, (index + 1.0) * dt)

    @jax.jit
    def advance_with_history(state: dict[str, jax.Array]):
        return lax.scan(
            recorded_step,
            state,
            xs=jnp.arange(count, dtype=jnp.float64),
        )

    final_state, history = advance_with_history(initial_state)
    history = jnp.concatenate((initial_history[None, :], history), axis=0)
    return final_state, grids, history


def run_final_only(
    case: str,
    settings: dict[str, Any],
    dt: float,
    final_time: float,
    sav_shift: float,
) -> tuple[dict[str, jax.Array], dict[str, jax.Array], jax.Array]:
    if sav_shift <= 0.0 or not math.isfinite(sav_shift):
        raise ValueError("SAV shift must be a positive finite number")
    count = time_steps(final_time, dt)
    initial_state, grids = make_initial_state(case, settings, sav_shift)
    diagnostics = make_diagnostics(case, grids)
    initial_energy = diagnostics(initial_state, jnp.asarray(0.0))[6]
    step = (
        make_two_stream_step(grids, dt)
        if case == "two-stream"
        else make_vm_step(grids, dt)
    )

    @jax.jit
    def advance(state: dict[str, jax.Array]):
        def body(index: jax.Array, current: dict[str, jax.Array]):
            next_state, _ = step(current, index)
            return next_state

        return lax.fori_loop(0, count, body, state)

    final_state = advance(initial_state)
    final_values = diagnostics(final_state, jnp.asarray(final_time))
    final_values = final_values.at[7].set(
        jnp.abs(final_values[6] - initial_energy)
        / jnp.maximum(jnp.abs(initial_energy), 1e-300)
    )
    return final_state, grids, final_values


def write_history(path: Path, history: jax.Array) -> None:
    host_history = jax.device_get(history).tolist()
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(HISTORY_COLUMNS)
        writer.writerows(host_history)


def write_fields(path: Path, state: dict[str, jax.Array], grids: dict[str, jax.Array]) -> None:
    fields = jax.device_get(
        jnp.stack((grids["x"], state["ex"], state["ey"], state["magnetic"]), axis=1)
    ).tolist()
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("x", "Ex", "Ey", "B"))
        writer.writerows(fields)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("two-stream", "weibel", "landau-weibel"),
        default="two-stream",
    )
    parser.add_argument("--nx", type=int)
    parser.add_argument("--nv", type=int, help="Set both velocity grid sizes")
    parser.add_argument("--nvx", type=int)
    parser.add_argument("--nvy", type=int)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--final-time", type=float)
    parser.add_argument("--k", type=float)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--sigma1", type=float)
    parser.add_argument("--sigma2", type=float)
    parser.add_argument("--v-bounds", nargs=2, type=float)
    parser.add_argument("--vx-bounds", nargs=2, type=float)
    parser.add_argument("--vy-bounds", nargs=2, type=float)
    parser.add_argument("--sav-shift", type=float, default=1e-12)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sav-sweep", action="store_true")
    parser.add_argument(
        "--sweep-exponents",
        nargs="+",
        type=int,
        default=(1, 2, 3, 4, 5, 6),
        help="For --sav-sweep, use dt=10**(-p) for each listed p",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = build_case(
        args.case,
        nx=args.nx,
        nv=args.nv,
        nvx=args.nvx,
        nvy=args.nvy,
        k=args.k,
        alpha=args.alpha,
        beta=args.beta,
        sigma1=args.sigma1,
        sigma2=args.sigma2,
        v_bounds=tuple(args.v_bounds) if args.v_bounds else None,
        vx_bounds=tuple(args.vx_bounds) if args.vx_bounds else None,
        vy_bounds=tuple(args.vy_bounds) if args.vy_bounds else None,
    )
    dt = args.dt if args.dt is not None else settings["dt"]
    final_time = args.final_time if args.final_time is not None else settings["final_time"]
    output_dir = args.output_dir or Path("results") / args.case
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.sav_sweep:
        if args.case != "two-stream":
            raise SystemExit("--sav-sweep is available only for the Section 6.1 case")
        sweep_rows = []
        for exponent in args.sweep_exponents:
            if exponent < 1:
                raise SystemExit("sweep exponents must be positive")
            sweep_dt = 10.0 ** (-exponent)
            steps = time_steps(final_time, sweep_dt)
            final_state, grids, final_row = run_final_only(
                args.case, settings, sweep_dt, final_time, args.sav_shift
            )
            del final_state, grids
            final_values = jax.device_get(final_row).tolist()
            sweep_rows.append((sweep_dt, steps, *final_values[1:]))
            print(
                f"dt={sweep_dt:g}: SAV error={final_values[8]:.6e}, "
                f"Gauss residual={final_values[1]:.6e}"
            )
        with (output_dir / "sav_sweep.csv").open(
            "w", newline="", encoding="utf-8"
        ) as output:
            writer = csv.writer(output)
            writer.writerow(
                ("dt", "steps", *HISTORY_COLUMNS[1:])
            )
            writer.writerows(sweep_rows)
        return

    state, grids, history = run_simulation(
        args.case, settings, dt, final_time, args.sav_shift
    )
    write_history(output_dir / "diagnostics.csv", history)
    write_fields(output_dir / "final_fields.csv", state, grids)
    final_row = jax.device_get(history[-1]).tolist()
    print(
        f"{args.case}: {time_steps(final_time, dt)} steps to t={final_time:g}; "
        f"Gauss residual={final_row[1]:.6e}, "
        f"relative modified-energy error={final_row[7]:.6e}, "
        f"SAV errors=({final_row[8]:.6e}, {final_row[9]:.6e})"
    )
    print(f"Diagnostics written to {output_dir}")


if __name__ == "__main__":
    main()
