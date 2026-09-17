import os
import sys
import warnings
import logging
import multiprocessing as mp
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
import astropy.units as u
import pyspeckit
from tqdm import tqdm


def gaussian_sum(x, pars):
    """Soma de N componentes gaussianas onde pars = [a1, c1, s1, a2, c2, s2, ...]"""
    total = np.zeros_like(x, dtype=float)
    for i in range(0, len(pars), 3):
        amp, cen, sig = pars[i], pars[i + 1], pars[i + 2]
        total += amp * np.exp(-0.5 * ((x - cen) / sig) ** 2)
    return total


# ==============================================================================
# FUNÇÃO TRABALHADORA PARALELA (N COMPONENTES)
# ==============================================================================
def process_single_spaxel(coords, spec_cube, wave_crop, snr_thresh, run_mc, n_iter_mc,
                           guesses, tied_pars, lim_min, min_pars, fit_range, exclude_range):
    warnings.filterwarnings('ignore')
    logging.getLogger('pyspeckit').setLevel(logging.ERROR)
    logging.getLogger('astropy').setLevel(logging.ERROR)
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

    y, x = coords
    spec_crop = spec_cube[:, y, x]

    if np.isnan(spec_crop).any() or np.isinf(spec_crop).any():
        return (y, x, None, None)

    bg_std = np.std(spec_crop[:10])
    if bg_std == 0 or (np.max(spec_crop) / bg_std) < snr_thresh:
        return (y, x, None, None)

    try:
        xarr = pyspeckit.units.SpectroscopicAxis(wave_crop * u.AA)
        sp = pyspeckit.Spectrum(data=spec_crop, xarr=xarr, unit='Flux')

        sp.baseline(
            exclude=exclude_range,
            order=1,
            subtract=False,
            highlight_fitregion=False
        )

        sp.specfit(
            fittype='gaussian',
            guesses=guesses,
            xmin=fit_range[0],
            xmax=fit_range[1],
            subtract_baseline=True,
            limitedmin=lim_min,
            minpars=min_pars,
            tied=tied_pars
        )

        pars = list(sp.specfit.modelpars)
        raw_errs = list(sp.specfit.modelerrs)

        if run_mc:
            fit_total = sp.baseline.basespec + gaussian_sum(wave_crop, pars)
            noise_std = np.std(spec_crop - fit_total)

            mc_results = []
            for _ in range(n_iter_mc):
                synth = fit_total + np.random.normal(0, noise_std, len(spec_crop))
                sp_mc = pyspeckit.Spectrum(data=synth, xarr=xarr, unit='Flux')
                sp_mc.baseline(exclude=exclude_range, order=1, subtract=False, highlight_fitregion=False)
                try:
                    sp_mc.specfit(
                        fittype='gaussian',
                        guesses=pars,
                        xmin=fit_range[0],
                        xmax=fit_range[1],
                        subtract_baseline=True,
                        limitedmin=lim_min,
                        minpars=min_pars,
                        tied=tied_pars
                    )
                    mc_results.append(list(sp_mc.specfit.modelpars))
                except Exception:
                    continue

            errs = np.std(mc_results, axis=0) if len(mc_results) > 5 else raw_errs
        else:
            errs = raw_errs

        return (y, x, pars, errs)

    except Exception:
        return (y, x, None, None)


# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================
if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    filename = '/home/matheus/Desktop/AGN/ESO137G34/ESO_corrigido.fits'
    with fits.open(filename) as hdul:
        header = hdul[0].header
        data = hdul[0].data

    n_wave, ny, nx = data.shape
    crpix3 = header.get('CRPIX3', 1)
    crval3 = header.get('CRVAL3', 0)
    cdelt3 = header.get('CD3_3', header.get('CDELT3', 1))
    wavelength = crval3 + (np.arange(n_wave) + 1 - crpix3) * cdelt3

    # ==========================================================================
    # CONFIGURAÇÃO DE N COMPONENTES
    # ==========================================================================
    # Exemplo: 3 componentes em torno da região de H-alpha + [N II]
    CROP_WAVE = [6500, 6620]
    FIT_RANGE = [6530, 6600]
    EXCLUDE_RANGE = [6540, 6595]

    # Lista de dicionários para definir N componentes
    # Cada componente possui: [amp_guess, cen_guess, sig_guess, tied_amp, tied_cen, tied_sig]
    COMPONENTS = [
        # Comp 1: [N II] 6584 (Livre)
        {'guess': [10.0, 6583.4, 2.0], 'tied': ['', '', '']},
        # Comp 2: [N II] 6548 (Amarrado ao [N II] 6584)
        {'guess': [3.3, 6548.0, 2.0],  'tied': ['p[0] / 3.05', 'p[1] - 35.4', 'p[2]']},
        # Comp 3: H-alpha 6563 (Centro e sigma livres)
        {'guess': [15.0, 6562.8, 2.5], 'tied': ['', '', '']}
    ]

    N_COMPONENTS = len(COMPONENTS)
    
    # Construção automática dos vetores de entrada
    GUESSES = []
    TIED_PARS = []
    LIM_MIN = []
    MIN_PARS = []

    for comp in COMPONENTS:
        GUESSES.extend(comp['guess'])
        TIED_PARS.extend(comp['tied'])
        LIM_MIN.extend([True, False, True])  # Amplitudes >= 0 e Sigmas >= 0.1
        MIN_PARS.extend([0.0, 0.0, 0.1])

    crop_mask = (wavelength >= CROP_WAVE[0]) & (wavelength <= CROP_WAVE[1])
    wave_crop = wavelength[crop_mask]
    spec_cube_crop = data[crop_mask, :, :]

    MONTE_CARLO = False
    N_ITER_MC = 50
    SNR_THRESHOLD = 3.0
    NUM_CORES = mp.cpu_count()

    print(f"Iniciando ajuste para {N_COMPONENTS} componentes ({NUM_CORES} cores)...")

    spaxel_coords = [(y, x) for y in range(ny) for x in range(nx)]

    worker_func = partial(
        process_single_spaxel,
        spec_cube=spec_cube_crop,
        wave_crop=wave_crop,
        snr_thresh=SNR_THRESHOLD,
        run_mc=MONTE_CARLO,
        n_iter_mc=N_ITER_MC,
        guesses=GUESSES,
        tied_pars=TIED_PARS,
        lim_min=LIM_MIN,
        min_pars=MIN_PARS,
        fit_range=FIT_RANGE,
        exclude_range=EXCLUDE_RANGE
    )

    with mp.Pool(processes=NUM_CORES) as pool:
        results = list(tqdm(
            pool.imap(worker_func, spaxel_coords, chunksize=10),
            total=len(spaxel_coords),
            desc="Processando Spaxels"
        ))

    # ==========================================================================
    # RECONSTRUÇÃO DOS MAPAS E EXPORTAÇÃO FITS
    # ==========================================================================
    n_pars = 3 * N_COMPONENTS
    param_cube = np.full((n_pars, ny, nx), np.nan)
    error_cube = np.full((n_pars, ny, nx), np.nan)

    for y, x, pars, errs in results:
        if pars is not None and errs is not None:
            param_cube[:, y, x] = pars
            error_cube[:, y, x] = errs

    header_out = header.copy()
    for kw in ['CRVAL3', 'CRPIX3', 'CDELT3', 'CD3_3', 'CTYPE3', 'CUNIT3']:
        header_out.remove(kw, ignore_missing=True, remove_all=True)

    plane_names = []
    for i in range(1, N_COMPONENTS + 1):
        plane_names.extend([f'amp{i}', f'cen{i}', f'sig{i}'])

    for idx, name in enumerate(plane_names, start=1):
        header_out[f'PLANE{idx}'] = (name, f'Parametro do plano {idx}')

    primary_hdu = fits.PrimaryHDU(data=param_cube, header=header_out)
    primary_hdu.header['EXTNAME'] = 'PARAMETERS'
    error_hdu = fits.ImageHDU(data=error_cube, header=header_out, name='ERRORS')

    hdul_out = fits.HDUList([primary_hdu, error_hdu])
    hdul_out.writeto('fit_result_n_comp.fits', overwrite=True)
    print("Processamento concluído com sucesso!")

    # ==========================================================================
    # VISUALIZAÇÃO INTERATIVA DINÂMICA
    # ==========================================================================
    # Calcula fluxo total somando todas as N componentes
    total_flux_map = np.zeros((ny, nx))
    for i in range(N_COMPONENTS):
        amp = param_cube[3 * i]
        sig = param_cube[3 * i + 2]
        flux_i = np.sqrt(2 * np.pi) * amp * sig
        total_flux_map += np.nan_to_num(flux_i)

    total_flux_map[np.isnan(param_cube[0])] = np.nan
    pos_flux_map = np.where(total_flux_map > 0, total_flux_map, np.nan)

    valid_data = pos_flux_map[np.isfinite(pos_flux_map)]
    vmin, vmax = np.percentile(valid_data, [1, 99]) if len(valid_data) > 0 else (1e-3, 1.0)
    if vmin <= 0: vmin = np.min(valid_data[valid_data > 0])

    fig, (ax_map, ax_spec) = plt.subplots(1, 2, figsize=(15, 6))

    im = ax_map.imshow(pos_flux_map, origin='lower', cmap='viridis', norm=LogNorm(vmin=vmin, vmax=vmax))
    ax_map.set_title(f'Fluxo Integrado Total ({N_COMPONENTS} Comps)\n[Clique em um spaxel]')
    plt.colorbar(im, ax=ax_map, format='%.1e')

    marker, = ax_map.plot([], [], 'rx', markersize=10, markeredgewidth=2)

    def onclick(event):
        if event.inaxes != ax_map: return
        x_pix, y_pix = int(round(event.xdata)), int(round(event.ydata))
        if not (0 <= x_pix < nx and 0 <= y_pix < ny): return

        marker.set_data([x_pix], [y_pix])
        ax_spec.clear()
        spec_spaxel = spec_cube_crop[:, y_pix, x_pix]
        ax_spec.plot(wave_crop, spec_spaxel, color='black', alpha=0.5, drawstyle='steps-mid', label='Observado')

        spaxel_pars = param_cube[:, y_pix, x_pix]

        if not np.isnan(spaxel_pars[0]):
            mask_base = (wave_crop < EXCLUDE_RANGE[0]) | (wave_crop > EXCLUDE_RANGE[1])
            poly = np.polyfit(wave_crop[mask_base], spec_spaxel[mask_base], deg=1)
            baseline = np.polyval(poly, wave_crop)

            ax_spec.plot(wave_crop, baseline, 'k--', alpha=0.4, label='Contínuo')

            colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta']
            for i in range(N_COMPONENTS):
                p_sub = spaxel_pars[3 * i : 3 * i + 3]
                g_i = gaussian_sum(wave_crop, p_sub)
                color = colors[i % len(colors)]
                ax_spec.plot(wave_crop, baseline + g_i, '--', color=color, label=f'Comp {i+1}')

            total_model = baseline + gaussian_sum(wave_crop, spaxel_pars)
            ax_spec.plot(wave_crop, total_model, 'r-', linewidth=1.8, label='Modelo Total')
            ax_spec.set_title(f'Spaxel ({x_pix}, {y_pix})')
        else:
            ax_spec.set_title(f'Spaxel ({x_pix}, {y_pix}) — Sem Ajuste')

        ax_spec.set_xlabel(r'Comprimento de Onda ($\AA$)')
        ax_spec.set_ylabel('Fluxo')
        ax_spec.legend(fontsize=8, loc='upper right')
        ax_spec.grid(True, linestyle=':', alpha=0.6)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout()
    plt.show()
