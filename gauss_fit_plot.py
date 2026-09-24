from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

# --- CONFIGURAÇÕES ---
filename = "/home/matheus/Desktop/estudos_eso137/SIFS/ESO137_fit_mapas.fits"

# Ajuste de velocidade (km/s)
v_corr = -50.0

# --- LIMITES DOS EIXOS EM PARSECS ---
y_lim_pc = (-1800, 1800)
x_lim_pc = None  # ex: (-1500, 1500) ou None para automático

# --- COORDENADAS DO NÚCLEO E ESCALA ESPACIAL ---
x_center = 32.5  # Pixel X do núcleo
y_center = 35.5  # Pixel Y do núcleo

pixel_scale_pc = 58.2  # Tamanho do pixel em parsecs (pc)

# --- UNIDADE DE FLUXO (Ajuste se o cabeçalho do seu FITS for diferente) ---
unit_flux = r"$\mathrm{erg\ s^{-1}\ cm^{-2}\ \AA^{-1}}$"
unit_vel = r"$\mathrm{km\ s^{-1}}$"

# --- CARREGAMENTO DOS DADOS ---
with fits.open(filename) as hdul:
    cube_data = hdul[2].data  # 3ª saída (Dados)
    cube_err = hdul[3].data  # 4ª saída (Erros)

    flux = cube_data[0]
    vel = cube_data[1]
    sig = cube_data[2]

    flux_err = cube_err[0]
    vel_err = cube_err[1]
    sig_err = cube_err[2]

# Aplica a correção de velocidade
vel_corr = vel + v_corr

# --- CONVERTE EIXOS PARA PARSECS RELATIVOS AO NÚCLEO ---
ny, nx = flux.shape
extent_pc = [
    (0 - x_center) * pixel_scale_pc,
    (nx - x_center) * pixel_scale_pc,
    (0 - y_center) * pixel_scale_pc,
    (ny - y_center) * pixel_scale_pc,
]


# Função auxiliar para plotagem com escala em pc, marcação do núcleo e rótulo na colorbar
def plot_with_colorbar(
    ax, data, title, cmap, cbar_label="", norm=None, extent=None, **kwargs
):
    im = ax.imshow(
        data, origin="lower", cmap=cmap, norm=norm, extent=extent, **kwargs
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(r"$\Delta x$ (pc)")
    ax.set_ylabel(r"$\Delta y$ (pc)")

    ax.set_aspect("equal")

    # Plota o núcleo
    ax.scatter(
        0,
        0,
        marker="+",
        color="black",
        s=180,
        linewidths=1.2,
        label="Núcleo",
        zorder=10,
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    cb = ax.figure.colorbar(im, cax=cax)

    # Adiciona o rótulo com a unidade na barra de cores
    if cbar_label:
        cb.set_label(cbar_label, fontsize=10)

    return im


def apply_axis_limits(ax):
    if y_lim_pc is not None:
        ax.set_ylim(y_lim_pc)
    if x_lim_pc is not None:
        ax.set_xlim(x_lim_pc)


# ==========================================================
# 1. FIGURA 1: FLUXO E ERRO DO FLUXO (Escala Logarítmica)
# ==========================================================
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(12, 5.4), sharex=True, sharey=True
)

ax2.tick_params(labelleft=True)

flux_log_safe = np.where(flux > 0, flux, np.nan)

plot_with_colorbar(
    ax1,
    flux_log_safe,
    "Fluxo [O III] $\lambda 5007$Å (Comp 1)",
    "viridis",
    cbar_label=f"Fluxo ({unit_flux})",
    norm=LogNorm(),
    extent=extent_pc,
    vmin=None,
    vmax=None,
)
plot_with_colorbar(
    ax2,
    flux_err,
    "Erro do Fluxo",
    "magma",
    cbar_label=f"Erro ({unit_flux})",
    extent=extent_pc,
    vmin=None,
    vmax=None,
)

apply_axis_limits(ax1)

plt.tight_layout()
plt.show()

# ==========================================================
# 2. FIGURA 2: VELOCIDADE E ERRO DA VELOCIDADE
# ==========================================================
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(12, 5.4), sharex=True, sharey=True
)
ax2.tick_params(labelleft=True)

plot_with_colorbar(
    ax1,
    vel_corr,
    "Velocidade [O III] $\lambda 5007$Å (Comp 1)",
    "RdBu_r",
    cbar_label=f"$v$ ({unit_vel})",
    extent=extent_pc,
    vmin=-300,
    vmax=300,
)
plot_with_colorbar(
    ax2,
    vel_err,
    "Erro da Velocidade",
    "magma",
    cbar_label=f"Erro ({unit_vel})",
    extent=extent_pc,
    vmin=None,
    vmax=None,
)

apply_axis_limits(ax1)

plt.tight_layout()
plt.show()

# ==========================================================
# 3. FIGURA 3: SIGMA E ERRO DO SIGMA
# ==========================================================
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(12, 5.4), sharex=True, sharey=True
)
ax2.tick_params(labelleft=True)

plot_with_colorbar(
    ax1,
    sig,
    r"Dispersão ($\sigma$) [O III] $\lambda 5007$Å (Comp 1)",
    "turbo",
    cbar_label=f"$\sigma$ ({unit_vel})",
    extent=extent_pc,
    vmin=None,
    vmax=None,
)
plot_with_colorbar(
    ax2,
    sig_err,
    "Erro do Sigma",
    "magma",
    cbar_label=f"Erro ({unit_vel})",
    extent=extent_pc,
    vmin=None,
    vmax=None,
)

apply_axis_limits(ax1)

plt.tight_layout()
plt.show()
