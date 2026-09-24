import os
import sys
import warnings
import logging
import multiprocessing as mp
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.widgets import RadioButtons
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.io import fits
import astropy.units as u
import pyspeckit
from tqdm import tqdm


# Constante física: velocidade da luz em km/s
C_KMS = 299792.458


def gaussian_sum(x, pars):
    """Soma de N componentes gaussianas ignorando componentes NaN/descartadas."""
    total = np.zeros_like(x, dtype=float)
    for i in range(0, len(pars), 3):
        amp, cen, sig = pars[i], pars[i + 1], pars[i + 2]
        if np.isnan(amp) or np.isnan(cen) or np.isnan(sig):
            continue
        total += amp * np.exp(-0.5 * ((x - cen) / sig) ** 2)
    return total


def sort_components_by_amplitude(pars, errs=None):
    """
    Ordena os trios de parâmetros [amp, cen, sig] e erros correspondentes 
    em ordem decrescente de amplitude (Amp_1 >= Amp_2 >= ... >= Amp_N).
    """
    if pars is None:
        return pars, errs

    n_comps = len(pars) // 3

    if errs is not None and len(errs) == len(pars):
        comp_tuples = [
            (pars[i * 3 : (i + 1) * 3], errs[i * 3 : (i + 1) * 3])
            for i in range(n_comps)
        ]
        comp_tuples.sort(key=lambda item: item[0][0] if not np.isnan(item[0][0]) else -np.inf, reverse=True)

        sorted_pars = [p for comp, _ in comp_tuples for p in comp]
        sorted_errs = [e for _, err in comp_tuples for e in err]
        return sorted_pars, sorted_errs
    else:
        comp_pars = [pars[i * 3 : (i + 1) * 3] for i in range(n_comps)]
        comp_pars.sort(key=lambda comp: comp[0] if not np.isnan(comp[0]) else -np.inf, reverse=True)
        sorted_pars = [p for comp in comp_pars for p in comp]
        return sorted_pars, errs


# ==============================================================================
# FUNÇÕES TRABALHADORAS PARALELAS
# ==============================================================================
def fit_single_spaxel_initial(coords, spec_cube, wave_crop, snr_thresh,
                              guesses, tied_pars, lim_min, min_pars, lim_max, max_pars,
                              fit_range, exclude_range, amp_rms_factor=3.0, sort_components=False):
    """Realiza o ajuste principal do spaxel observado."""
    warnings.filterwarnings('ignore')
    logging.getLogger('pyspeckit').setLevel(logging.ERROR)
    logging.getLogger('astropy').setLevel(logging.ERROR)
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

    y, x = coords
    spec_crop = spec_cube[:, y, x]

    if np.isnan(spec_crop).any() or np.isinf(spec_crop).any():
        return (y, x, None, None, None, None)

    bg_std = np.std(spec_crop[:10])
    if bg_std == 0 or (np.max(spec_crop) / bg_std) < snr_thresh:
        return (y, x, None, None, None, None)

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
            limitedmax=lim_max,
            maxpars=max_pars,
            tied=tied_pars
        )

        pars = list(sp.specfit.modelpars)
        raw_errs = list(sp.specfit.modelerrs)

        fit_total = sp.baseline.basespec + gaussian_sum(wave_crop, pars)
        residuals = spec_crop - fit_total
        noise_std = np.std(residuals)

        # Validação de componentes: Amp > amp_rms_factor * RMS
        for i in range(len(pars) // 3):
            amp = pars[3 * i]
            if amp < amp_rms_factor * noise_std:
                pars[3 * i : 3 * i + 3] = [np.nan, np.nan, np.nan]
                if raw_errs is not None and len(raw_errs) == len(pars):
                    raw_errs[3 * i : 3 * i + 3] = [np.nan, np.nan, np.nan]

        if np.all(np.isnan(pars)):
            return (y, x, None, None, None, None)

        if sort_components:
            pars, raw_errs = sort_components_by_amplitude(pars, raw_errs)

        return (y, x, pars, raw_errs, fit_total, noise_std)

    except Exception:
        return (y, x, None, None, None, None)


def fit_single_spaxel_mc(item, wave_crop, guesses_init, tied_pars, lim_min, min_pars, lim_max, max_pars,
                          fit_range, exclude_range):
    """Executa 1 iteração de Monte Carlo para 1 spaxel de forma robusta."""
    coords, fit_total, noise_std, best_pars = item
    y, x = coords

    if fit_total is None or best_pars is None:
        return (y, x, None)

    warnings.filterwarnings('ignore')
    logging.getLogger('pyspeckit').setLevel(logging.ERROR)
    logging.getLogger('astropy').setLevel(logging.ERROR)
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

    try:
        mc_guesses = [g_init if np.isnan(p) else p for p, g_init in zip(best_pars, guesses_init)]

        synth = fit_total + np.random.normal(0, noise_std, len(fit_total))
        xarr = pyspeckit.units.SpectroscopicAxis(wave_crop * u.AA)
        sp_mc = pyspeckit.Spectrum(data=synth, xarr=xarr, unit='Flux')
        sp_mc.baseline(exclude=exclude_range, order=1, subtract=False, highlight_fitregion=False)

        sp_mc.specfit(
            fittype='gaussian',
            guesses=mc_guesses,
            xmin=fit_range[0],
            xmax=fit_range[1],
            subtract_baseline=True,
            limitedmin=lim_min,
            minpars=min_pars,
            limitedmax=lim_max,
            maxpars=max_pars,
            tied=tied_pars
        )
        mc_pars = list(sp_mc.specfit.modelpars)

        return (y, x, mc_pars)
    except Exception:
        return (y, x, None)


# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================
if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    filename = '/home/matheus/Desktop/estudos_eso137/SIFS/ESO137_calibrado_fluxo.fits'
    output_filename = '/home/matheus/Desktop/estudos_eso137/SIFS/ESO137_fit_mapas.fits'

    with fits.open(filename) as hdul:
        header = hdul[0].header
        data = hdul[0].data

    n_wave, ny, nx = data.shape
    crpix3 = header.get('CRPIX3', 1)
    crval3 = header.get('CRVAL3', 0)
    cdelt3 = header.get('CD3_3', header.get('CDELT3', 1))
    wavelength = crval3 + (np.arange(n_wave) + 1 - crpix3) * cdelt3

    # ==========================================================================
    # CONFIGURAÇÃO DE FÍSICA E AJUSTE
    # ==========================================================================
    LAMBDA_REST = 5051  # Comprimento de onda de repouso em Å (ex: [O III])

    CROP_WAVE = [5020, 5100]
    FIT_RANGE = [5025, 5095]
    EXCLUDE_RANGE = [5030, 5080]

    AMP_RMS_FACTOR = 3.0
    SORT_BY_AMPLITUDE = True

    MONTE_CARLO = True
    N_ITER_MC = 500
    SNR_THRESHOLD = 3.0
    NUM_CORES = 12

    COMPONENTS = [
        {
            'guess': [2.5e-16, 5050.0, 2.0],
            'tied': ['', '', ''],
            'limits': [(0.0, None), (5030.0, 5070.0), (1.0, None)]
        },
        {
            'guess': [1.5e-16, 5050.0, 3.0],
            'tied': ['', '', ''],
            'limits': [(0.0, None), (5030.0, 5070.0), (1.0, None)]
        },
    ]

    N_COMPONENTS = len(COMPONENTS)

    GUESSES, TIED_PARS = [], []
    LIM_MIN, MIN_PARS = [], []
    LIM_MAX, MAX_PARS = [], []

    for comp in COMPONENTS:
        GUESSES.extend(comp['guess'])
        TIED_PARS.extend(comp['tied'])

        limits = comp.get('limits', [(0.0, None), (None, None), (0.1, None)])
        for low, high in limits:
            if low is not None:
                LIM_MIN.append(True)
                MIN_PARS.append(float(low))
            else:
                LIM_MIN.append(False)
                MIN_PARS.append(0.0)

            if high is not None:
                LIM_MAX.append(True)
                MAX_PARS.append(float(high))
            else:
                LIM_MAX.append(False)
                MAX_PARS.append(0.0)

    crop_mask = (wavelength >= CROP_WAVE[0]) & (wavelength <= CROP_WAVE[1])
    wave_crop = wavelength[crop_mask]
    spec_cube_crop = data[crop_mask, :, :]

    print("Ajuste de cubo iniciado\n")
    print("# ===== Parâmetros do ajuste ===== #\n")
    print(f"Comprimento de onda de repouso (Lambda0): {LAMBDA_REST} Å")
    print(f"Ajuste para {N_COMPONENTS} componentes")
    print(f"Núcleos utilizados: {NUM_CORES}")
    print(f"Fator de corte RMS: Amp > {AMP_RMS_FACTOR:.1f}x RMS")
    print(f"Ordenação por amplitude (Amp1 > Amp2...): {'Ativada' if SORT_BY_AMPLITUDE else 'Desativada'}")
    print(f"Erro por Monte Carlo: {'Ativado' if MONTE_CARLO else 'Desativado'}")

    spaxel_coords = [(y, x) for y in range(ny) for x in range(nx)]

    # 1. Ajuste Inicial do Cubo
    worker_func = partial(
        fit_single_spaxel_initial,
        spec_cube=spec_cube_crop,
        wave_crop=wave_crop,
        snr_thresh=SNR_THRESHOLD,
        guesses=GUESSES,
        tied_pars=TIED_PARS,
        lim_min=LIM_MIN,
        min_pars=MIN_PARS,
        lim_max=LIM_MAX,
        max_pars=MAX_PARS,
        fit_range=FIT_RANGE,
        exclude_range=EXCLUDE_RANGE,
        amp_rms_factor=AMP_RMS_FACTOR,
        sort_components=SORT_BY_AMPLITUDE
    )

    with mp.Pool(processes=NUM_CORES) as pool:
        desc_init = "Processando Spaxels" if not MONTE_CARLO else "Processando Spaxels (Ajuste Inicial)"
        initial_results = list(tqdm(
            pool.imap(worker_func, spaxel_coords, chunksize=10),
            total=len(spaxel_coords),
            desc=desc_init
        ))

    n_pars = 3 * N_COMPONENTS
    param_cube = np.full((n_pars, ny, nx), np.nan)
    error_cube = np.full((n_pars, ny, nx), np.nan)

    mc_data_list = []
    for y, x, pars, errs, fit_total, noise_std in initial_results:
        if pars is not None and fit_total is not None:
            param_cube[:, y, x] = pars
            error_cube[:, y, x] = errs
            mc_data_list.append(((y, x), fit_total, noise_std, pars))
        else:
            mc_data_list.append(((y, x), None, None, None))

    # 2. Execução do Monte Carlo
    mc_results_dict = {}
    if MONTE_CARLO:
        with mp.Pool(processes=NUM_CORES) as pool:
            for mc_i in range(1, N_ITER_MC + 1):
                mc_worker = partial(
                    fit_single_spaxel_mc,
                    wave_crop=wave_crop,
                    guesses_init=GUESSES,
                    tied_pars=TIED_PARS,
                    lim_min=LIM_MIN,
                    min_pars=MIN_PARS,
                    lim_max=LIM_MAX,
                    max_pars=MAX_PARS,
                    fit_range=FIT_RANGE,
                    exclude_range=EXCLUDE_RANGE
                )

                is_last = (mc_i == N_ITER_MC)
                mc_results = list(tqdm(
                    pool.imap(mc_worker, mc_data_list, chunksize=10),
                    total=len(mc_data_list),
                    desc=f"Processando Spaxels: (Monte Carlo {mc_i}/{N_ITER_MC})",
                    leave=is_last
                ))

                for y, x, mc_pars in mc_results:
                    if mc_pars is not None:
                        if (y, x) not in mc_results_dict:
                            mc_results_dict[(y, x)] = []
                        mc_results_dict[(y, x)].append(mc_pars)

        # Atualiza os erros das componentes espectrais via percentis MC
        for (y, x), samples in mc_results_dict.items():
            if len(samples) > 5:
                samples_arr = np.array(samples)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    p16, p84 = np.percentile(samples_arr, [16, 84], axis=0)
                    mc_errs = (p84 - p16) / 2.0

                initial_pars = param_cube[:, y, x]
                mc_errs = np.where(np.isnan(initial_pars), np.nan, mc_errs)
                error_cube[:, y, x] = mc_errs

    # ==========================================================================
    # CÁLCULO DOS PARÂMETROS FÍSICOS E PROPAGAÇÃO DE ERROS
    # ==========================================================================
    map_dict = {}

    total_flux_map = np.zeros((ny, nx))
    total_flux_err_sq = np.zeros((ny, nx))
    total_amp_map = np.zeros((ny, nx))
    total_amp_err_sq = np.zeros((ny, nx))

    for i in range(N_COMPONENTS):
        amp = param_cube[3 * i]
        cen = param_cube[3 * i + 1]
        sig = param_cube[3 * i + 2]

        err_amp = error_cube[3 * i]
        err_cen = error_cube[3 * i + 1]
        err_sig = error_cube[3 * i + 2]

        valid_mask = ~np.isnan(amp)

        # 1. Parâmetros Espectrais Puros
        map_dict[f'Amp Comp {i+1}'] = np.where(valid_mask, amp, np.nan)
        map_dict[f'Erro Amp Comp {i+1}'] = np.where(valid_mask, err_amp, np.nan)

        map_dict[f'Centro Comp {i+1}'] = np.where(valid_mask, cen, np.nan)
        map_dict[f'Erro Centro Comp {i+1}'] = np.where(valid_mask, err_cen, np.nan)

        map_dict[f'Sigma Comp {i+1}'] = np.where(valid_mask, sig, np.nan)
        map_dict[f'Erro Sigma Comp {i+1}'] = np.where(valid_mask, err_sig, np.nan)

        # 2. Conversões Físicas
        flux_comp = np.sqrt(2 * np.pi) * amp * sig
        vel_comp = C_KMS * (cen - LAMBDA_REST) / LAMBDA_REST
        disp_comp = C_KMS * sig / LAMBDA_REST

        # 3. Propagação de Erros
        if MONTE_CARLO and len(mc_results_dict) > 0:
            err_flux_comp = np.full((ny, nx), np.nan)
            err_vel_comp = np.full((ny, nx), np.nan)
            err_disp_comp = np.full((ny, nx), np.nan)

            for (y_pix, x_pix), samples in mc_results_dict.items():
                if len(samples) > 5 and not np.isnan(amp[y_pix, x_pix]):
                    samples_arr = np.array(samples)
                    amp_s = samples_arr[:, 3 * i]
                    cen_s = samples_arr[:, 3 * i + 1]
                    sig_s = samples_arr[:, 3 * i + 2]

                    flux_s = np.sqrt(2 * np.pi) * amp_s * sig_s
                    vel_s = C_KMS * (cen_s - LAMBDA_REST) / LAMBDA_REST
                    disp_s = C_KMS * sig_s / LAMBDA_REST

                    p16_f, p84_f = np.percentile(flux_s, [16, 84])
                    p16_v, p84_v = np.percentile(vel_s, [16, 84])
                    p16_d, p84_d = np.percentile(disp_s, [16, 84])

                    err_flux_comp[y_pix, x_pix] = (p84_f - p16_f) / 2.0
                    err_vel_comp[y_pix, x_pix] = (p84_v - p16_v) / 2.0
                    err_disp_comp[y_pix, x_pix] = (p84_d - p16_d) / 2.0
        else:
            err_flux_comp = flux_comp * np.sqrt(
                np.maximum(0, (err_amp / np.where(amp == 0, np.nan, amp))**2 + 
                              (err_sig / np.where(sig == 0, np.nan, sig))**2)
            )
            err_vel_comp = C_KMS * err_cen / LAMBDA_REST
            err_disp_comp = C_KMS * err_sig / LAMBDA_REST

        map_dict[f'Fluxo Comp {i+1}'] = np.where(valid_mask, flux_comp, np.nan)
        map_dict[f'Erro Fluxo Comp {i+1}'] = np.where(valid_mask, err_flux_comp, np.nan)

        map_dict[f'Velocidade Comp {i+1}'] = np.where(valid_mask, vel_comp, np.nan)
        map_dict[f'Erro Velocidade Comp {i+1}'] = np.where(valid_mask, err_vel_comp, np.nan)

        map_dict[f'Dispersão Comp {i+1}'] = np.where(valid_mask, disp_comp, np.nan)
        map_dict[f'Erro Dispersão Comp {i+1}'] = np.where(valid_mask, err_disp_comp, np.nan)

        total_amp_map += np.nan_to_num(amp)
        total_amp_err_sq += np.nan_to_num(err_amp)**2

        total_flux_map += np.nan_to_num(flux_comp)
        total_flux_err_sq += np.nan_to_num(err_flux_comp)**2

    total_amp_map = np.where(total_amp_map > 0, total_amp_map, np.nan)
    total_amp_err_map = np.where(total_amp_err_sq > 0, np.sqrt(total_amp_err_sq), np.nan)

    total_flux_map = np.where(total_flux_map > 0, total_flux_map, np.nan)
    total_flux_err_map = np.where(total_flux_err_sq > 0, np.sqrt(total_flux_err_sq), np.nan)

    map_dict['Amp Total'] = total_amp_map
    map_dict['Erro Amp Total'] = total_amp_err_map

    map_dict['Fluxo Total'] = total_flux_map
    map_dict['Erro Fluxo Total'] = total_flux_err_map

    # ==========================================================================
    # CONSTRUÇÃO E SALVAMENTO DO ARQUIVO FITS COM 4 EXTENSÕES
    # ==========================================================================
    print("\nMontando e salvando o arquivo FITS...")

    # Organiza os cubos dos parâmetros físicos e seus erros (Extensões 3 e 4)
    n_phys = 3 * N_COMPONENTS
    phys_cube = np.full((n_phys, ny, nx), np.nan)
    phys_err_cube = np.full((n_phys, ny, nx), np.nan)

    for i in range(N_COMPONENTS):
        phys_cube[3 * i, :, :] = map_dict[f'Fluxo Comp {i+1}']
        phys_cube[3 * i + 1, :, :] = map_dict[f'Velocidade Comp {i+1}']
        phys_cube[3 * i + 2, :, :] = map_dict[f'Dispersão Comp {i+1}']

        phys_err_cube[3 * i, :, :] = map_dict[f'Erro Fluxo Comp {i+1}']
        phys_err_cube[3 * i + 1, :, :] = map_dict[f'Erro Velocidade Comp {i+1}']
        phys_err_cube[3 * i + 2, :, :] = map_dict[f'Erro Dispersão Comp {i+1}']

    # Extensão 1: Parâmetros do modelo espectral (Primary HDU)
    hdu1 = fits.PrimaryHDU(data=param_cube, header=header.copy())
    hdu1.header['EXTNAME'] = 'PARS_FIT'
    hdu1.header['COMMENT'] = 'Extensao 1: Amp1, Pos1, Sig1, Amp2, Pos2, Sig2, ...'

    # Extensão 2: Erros dos parâmetros do modelo espectral
    hdu2 = fits.ImageHDU(data=error_cube, name='PARS_FIT_ERR')
    hdu2.header['COMMENT'] = 'Extensao 2: Erros de Amp1, Pos1, Sig1, Amp2, Pos2, Sig2, ...'

    # Extensão 3: Parâmetros físicos (Fluxo, Velocidade, Dispersão)
    hdu3 = fits.ImageHDU(data=phys_cube, name='PARS_PHYS')
    hdu3.header['COMMENT'] = 'Extensao 3: Flux1, Vel1, Disp1, Flux2, Vel2, Disp2, ...'

    # Extensão 4: Erros dos parâmetros físicos
    hdu4 = fits.ImageHDU(data=phys_err_cube, name='PARS_PHYS_ERR')
    hdu4.header['COMMENT'] = 'Extensao 4: Erros de Flux1, Vel1, Disp1, Flux2, Vel2, Disp2, ...'

    # Adiciona metadados nos cabeçalhos descrevendo os eixos/camadas
    for i in range(N_COMPONENTS):
        # Extensões 1 e 2
        hdu1.header[f'BAND{3*i+1}'] = f'Amp{i+1}'
        hdu1.header[f'BAND{3*i+2}'] = f'Pos{i+1}'
        hdu1.header[f'BAND{3*i+3}'] = f'Sig{i+1}'

        hdu2.header[f'BAND{3*i+1}'] = f'Amp{i+1}_err'
        hdu2.header[f'BAND{3*i+2}'] = f'Pos{i+1}_err'
        hdu2.header[f'BAND{3*i+3}'] = f'Sig{i+1}_err'

        # Extensões 3 e 4
        hdu3.header[f'BAND{3*i+1}'] = f'Flux{i+1}'
        hdu3.header[f'BAND{3*i+2}'] = f'Vel{i+1}'
        hdu3.header[f'BAND{3*i+3}'] = f'Disp{i+1}'

        hdu4.header[f'BAND{3*i+1}'] = f'Flux{i+1}_err'
        hdu4.header[f'BAND{3*i+2}'] = f'Vel{i+1}_err'
        hdu4.header[f'BAND{3*i+3}'] = f'Disp{i+1}_err'

    hdul_out = fits.HDUList([hdu1, hdu2, hdu3, hdu4])
    hdul_out.writeto(output_filename, overwrite=True)
    print(f"Arquivo FITS salvo com sucesso em: {output_filename}\n")

    # ==========================================================================
    # VISUALIZAÇÃO INTERATIVA DINÂMICA
    # ==========================================================================
    FONT_TITLE = 13
    FONT_RADIO_TITLE = 11
    FONT_RADIO_BTNS = 9
    FONT_INFO_BOX = 9.5
    FONT_AXES_LABELS = 10
    FONT_LEGEND = 8.5

    fig = plt.figure(figsize=(16, 9.5))

    ax_map = fig.add_axes([0.06, 0.48, 0.38, 0.46])
    divider = make_axes_locatable(ax_map)
    ax_cbar = divider.append_axes("right", size="5%", pad=0.05)

    ax_spec = fig.add_axes([0.52, 0.48, 0.44, 0.46])

    comp_options = ['Total'] + [f'Comp {i+1}' for i in range(N_COMPONENTS)]
    param_options = ['Amplitude', 'Centro (Å)', 'Sigma (Å)', 'Fluxo', 'Velocidade (km/s)', 'Dispersão (km/s)']
    type_options = ['Valor', 'Erro']

    ax_radio_comp  = fig.add_axes([0.05, 0.08, 0.11, 0.32], facecolor='#f0f0f0')
    ax_radio_param = fig.add_axes([0.17, 0.08, 0.18, 0.32], facecolor='#f0f0f0')
    ax_radio_type  = fig.add_axes([0.36, 0.08, 0.10, 0.32], facecolor='#f0f0f0')

    ax_radio_comp.set_title('Componente', fontsize=FONT_RADIO_TITLE, fontweight='bold', pad=6)
    ax_radio_param.set_title('Parâmetro', fontsize=FONT_RADIO_TITLE, fontweight='bold', pad=6)
    ax_radio_type.set_title('Exibir', fontsize=FONT_RADIO_TITLE, fontweight='bold', pad=6)

    radio_comp = RadioButtons(ax_radio_comp, comp_options, active=0)
    radio_param = RadioButtons(ax_radio_param, param_options, active=0)
    radio_type = RadioButtons(ax_radio_type, type_options, active=0)

    for r in [radio_comp, radio_param, radio_type]:
        for lbl in r.labels:
            lbl.set_fontsize(FONT_RADIO_BTNS)

    ax_info = fig.add_axes([0.52, 0.08, 0.44, 0.32], facecolor='#f8f9fa')
    ax_info.axis('off')
    ax_info.text(0.5, 0.5, 'Clique em um spaxel no mapa\npara exibir todos os parâmetros físicos', 
                 ha='center', va='center', fontsize=FONT_INFO_BOX, color='gray', transform=ax_info.transAxes)

    def get_norm_and_cmap(label, data):
        valid = data[np.isfinite(data)]
        if len(valid) == 0:
            return Normalize(vmin=0, vmax=1), 'viridis'

        is_error = 'Erro' in label

        if 'Amp' in label or 'Fluxo' in label:
            pos_valid = valid[valid > 0]
            if len(pos_valid) > 0:
                vmin, vmax = np.percentile(pos_valid, [1, 99])
                if vmin == vmax: vmin, vmax = vmin * 0.9, vmax * 1.1
            else:
                vmin, vmax = 1e-18, 1e-15
            cmap = 'inferno' if is_error else 'viridis'
            return LogNorm(vmin=vmin, vmax=vmax), cmap

        elif 'Velocidade' in label:
            if is_error:
                vmin, vmax = np.percentile(valid, [2, 98])
                return Normalize(vmin=vmin, vmax=vmax), 'magma'
            else:
                vmax = np.percentile(np.abs(valid), 98)
                return Normalize(vmin=-vmax, vmax=vmax), 'RdBu_r'

        elif 'Centro' in label:
            vmin, vmax = np.percentile(valid, [2, 98])
            return Normalize(vmin=vmin, vmax=vmax), 'coolwarm'

        else: # Sigma / Dispersao
            vmin, vmax = np.percentile(valid, [2, 98])
            cmap = 'cividis' if is_error else 'plasma'
            return Normalize(vmin=vmin, vmax=vmax), cmap

    init_map_name = 'Amp Total'
    init_data = map_dict[init_map_name]
    init_norm, init_cmap = get_norm_and_cmap(init_map_name, init_data)

    im = ax_map.imshow(init_data, origin='lower', cmap=init_cmap, norm=init_norm)
    ax_map.set_title(f'Mapa: {init_map_name}\n[Clique em um spaxel]', fontsize=FONT_TITLE)

    cbar = fig.colorbar(im, cax=ax_cbar, format='%.1e')
    marker, = ax_map.plot([], [], 'rx', markersize=10, markeredgewidth=2)

    def update_map(_=None):
        global cbar
        selected_comp = radio_comp.value_selected
        selected_param = radio_param.value_selected
        selected_type = radio_type.value_selected

        prefix = "Erro " if selected_type == "Erro" else ""

        if selected_comp == 'Total':
            if selected_param == 'Fluxo':
                key = f"{prefix}Fluxo Total"
            else:
                key = f"{prefix}Amp Total"
        else:
            p_name = selected_param.split(' ')[0] # Extrai 'Centro', 'Sigma', etc.
            key = f"{prefix}{p_name} {selected_comp}"

        if key in map_dict:
            current_data = map_dict[key]
            norm, cmap = get_norm_and_cmap(key, current_data)

            im.set_data(current_data)
            im.set_norm(norm)
            im.set_cmap(cmap)
            
            ax_cbar.clear()
            fmt_str = '%.1e' if ('Amp' in key or 'Fluxo' in key) else '%.1f'
            cbar = fig.colorbar(im, cax=ax_cbar, format=fmt_str)

            ax_map.set_title(f'Mapa: {key}\n[Clique em um spaxel]', fontsize=FONT_TITLE)
            fig.canvas.draw_idle()

    radio_comp.on_clicked(update_map)
    radio_param.on_clicked(update_map)
    radio_type.on_clicked(update_map)

    def onclick(event):
        if event.inaxes != ax_map: return
        x_pix, y_pix = int(round(event.xdata)), int(round(event.ydata))
        if not (0 <= x_pix < nx and 0 <= y_pix < ny): return

        marker.set_data([x_pix], [y_pix])
        ax_spec.clear()
        ax_info.clear()
        ax_info.axis('off')

        spec_spaxel = spec_cube_crop[:, y_pix, x_pix]
        ax_spec.plot(wave_crop, spec_spaxel, color='black', alpha=0.5, drawstyle='steps-mid', label='Observado')

        spaxel_pars = param_cube[:, y_pix, x_pix]

        if not np.all(np.isnan(spaxel_pars)):
            mask_base = (wave_crop < EXCLUDE_RANGE[0]) | (wave_crop > EXCLUDE_RANGE[1])
            poly = np.polyfit(wave_crop[mask_base], spec_spaxel[mask_base], deg=1)
            baseline = np.polyval(poly, wave_crop)

            ax_spec.plot(wave_crop, baseline, 'k--', alpha=0.4, label='Contínuo')

            colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta']
            total_model = baseline + gaussian_sum(wave_crop, spaxel_pars)
            rms_spaxel = np.std(spec_spaxel - total_model)

            info_lines = [
                f"=== SPAXEL ({x_pix}, {y_pix}) ===",
                f"RMS Ruído: {rms_spaxel:.2e} | Lambda0: {LAMBDA_REST} Å",
                "-" * 48
            ]

            for i in range(N_COMPONENTS):
                p_sub = spaxel_pars[3 * i : 3 * i + 3]
                amp, cen, sig = p_sub
                
                if np.isnan(amp):
                    info_lines.append(f"• COMPONENTE {i+1}: Descartada (< {AMP_RMS_FACTOR:.1f}xRMS)")
                    continue

                g_i = gaussian_sum(wave_crop, p_sub)
                color = colors[i % len(colors)]
                ax_spec.plot(wave_crop, baseline + g_i, '--', color=color, label=f'Comp {i+1}')

                # Resgata valores e erros do map_dict
                e_amp  = map_dict[f'Erro Amp Comp {i+1}'][y_pix, x_pix]
                e_cen  = map_dict[f'Erro Centro Comp {i+1}'][y_pix, x_pix]
                e_sig  = map_dict[f'Erro Sigma Comp {i+1}'][y_pix, x_pix]

                flux   = map_dict[f'Fluxo Comp {i+1}'][y_pix, x_pix]
                e_flux = map_dict[f'Erro Fluxo Comp {i+1}'][y_pix, x_pix]

                vel    = map_dict[f'Velocidade Comp {i+1}'][y_pix, x_pix]
                e_vel  = map_dict[f'Erro Velocidade Comp {i+1}'][y_pix, x_pix]

                disp   = map_dict[f'Dispersão Comp {i+1}'][y_pix, x_pix]
                e_disp = map_dict[f'Erro Dispersão Comp {i+1}'][y_pix, x_pix]

                info_lines.append(f"• COMPONENTE {i+1}:")
                info_lines.append(f"  Fluxo : {flux:.2e} ± {e_flux:.2e}")
                info_lines.append(f"  Vel   : {vel:+.1f} ± {e_vel:.1f} km/s (Centro: {cen:.2f} ± {e_cen:.2f} Å)")
                info_lines.append(f"  Disp  : {disp:.1f} ± {e_disp:.1f} km/s (Sigma : {sig:.2f} ± {e_sig:.2f} Å)")
                info_lines.append("")

            ax_spec.plot(wave_crop, total_model, 'r-', linewidth=1.8, label='Modelo Total')
            ax_spec.set_title(f'Spaxel ({x_pix}, {y_pix})', fontsize=FONT_TITLE)

            ax_info.text(
                0.02, 0.96, "\n".join(info_lines),
                transform=ax_info.transAxes,
                verticalalignment='top',
                fontsize=FONT_INFO_BOX,
                family='monospace'
            )
        else:
            ax_spec.set_title(f'Spaxel ({x_pix}, {y_pix}) — Sem Ajuste Válido', fontsize=FONT_TITLE)
            ax_info.text(
                0.05, 0.93, f"=== SPAXEL ({x_pix}, {y_pix}) ===\n\nTodas as componentes foram descartadas.",
                transform=ax_info.transAxes,
                verticalalignment='top',
                fontsize=FONT_INFO_BOX,
                family='monospace'
            )

        ax_spec.set_xlabel(r'Comprimento de Onda ($\AA$)', fontsize=FONT_AXES_LABELS)
        ax_spec.set_ylabel('Fluxo', fontsize=FONT_AXES_LABELS)
        ax_spec.legend(fontsize=FONT_LEGEND, loc='upper right')
        ax_spec.grid(True, linestyle=':', alpha=0.6)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
