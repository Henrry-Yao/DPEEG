#!/usr/bin/env python3
"""
真实 EEG 信号的频段能量与 ERP 波形分析（按受试者）
—— 用于补充验证"低一致性受试者 EEG 信号特征"
"""
import numpy as np
import torch
import sys
from pathlib import Path
from scipy.fft import fft, fftfreq
from scipy.signal import welch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = Path("data/things-eeg/Preprocessed_data_250Hz_whiten")
SUBJECTS = ["sub-%02d" % i for i in range(1, 11)]
TIMESTEPS = (0, 250)
FS = 250  # Hz
SELECTED_CH = ['P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
               'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']
ALL_CHANNELS = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
                'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1',
                'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
                'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
                'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
                'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                'O1', 'Oz', 'O2']

BANDS = {
    "delta (1-4)":  (1, 4),
    "theta (4-8)":  (4, 8),
    "alpha (8-13)": (8, 13),
    "beta (13-30)": (13, 30),
    "gamma (30-80)":(30, 80),
}

selected_idx = [ALL_CHANNELS.index(ch) for ch in SELECTED_CH]


def compute_band_power(eeg_trials: np.ndarray, fs: float) -> dict:
    """
    eeg_trials: (N_trials, n_channels, n_timepoints)
    Returns dict of band_name -> mean relative power
    """
    n_trials, n_ch, n_t = eeg_trials.shape
    # Welch PSD per trial, per channel
    total_power = 0
    band_powers = {b: 0.0 for b in BANDS}

    for trial in eeg_trials:  # (17, 250)
        for ch in range(n_ch):
            freqs, psd = welch(trial[ch], fs=fs, nperseg=min(128, n_t))
            total = np.trapz(psd[(freqs >= 1) & (freqs <= 80)], freqs[(freqs >= 1) & (freqs <= 80)])
            if total <= 0:
                continue
            for name, (lo, hi) in BANDS.items():
                mask = (freqs >= lo) & (freqs < hi)
                band_powers[name] += np.trapz(psd[mask], freqs[mask]) / total

    n_valid = n_trials * n_ch
    return {k: v / n_valid for k, v in band_powers.items()}


def compute_erp(eeg_data: np.ndarray) -> dict:
    """
    eeg_data: (N_concepts, N_trials, n_channels, n_timepoints)
    Returns grand-average ERP, per-channel ERP, peak info
    """
    n_conc, n_trials, n_ch, n_t = eeg_data.shape
    # Grand average across all concepts, trials, channels
    grand_erp = eeg_data.mean(axis=(0, 1, 2))  # (250,)
    # Per-trial variance (noise floor)
    trial_variance = eeg_data.var(axis=(1, 2, 3))  # (N_concepts,) - across trials within concept

    return {
        "grand_erp": grand_erp,
        "mean_amplitude": float(np.abs(grand_erp).mean()),
        "peak_amplitude": float(np.abs(grand_erp).max()),
        "peak_latency_ms": float(np.argmax(np.abs(grand_erp)) * 1000 / FS),
        "mean_trial_variance": float(trial_variance.mean()),
        "trial_variance_std": float(trial_variance.std()),
    }


def main():
    # 已知数据
    consistency = [0.0828, 0.0530, 0.0964, 0.0880, 0.0369,
                   0.0632, 0.0995, 0.0900, 0.0392, 0.0893]
    gen_enc = [9.5, 16.5, 15.5, 9.0, 35.0, 22.5, 12.5, 28.0, 70.5, 19.0]

    results = []
    for si, sub in enumerate(SUBJECTS):
        print("Processing %s ..." % sub, end=" ", flush=True)
        data_path = DATA_DIR / sub / "test.pt"
        loaded = torch.load(data_path, weights_only=False)
        eeg = torch.from_numpy(loaded['eeg']).float()
        # (N_img, N_trials, C_all, T) → select channels + time
        eeg = eeg[:, :, selected_idx, TIMESTEPS[0]:TIMESTEPS[1]]  # (200, 80, 17, 250)

        # 所有试次展开
        all_trials = eeg.reshape(-1, 17, 250).numpy()  # (16000, 17, 250)

        # 频段能量
        band_power = compute_band_power(all_trials, FS)

        # ERP
        erp_info = compute_erp(eeg.numpy())

        results.append({
            "subject": sub,
            "consistency": consistency[si],
            "gen_enc": gen_enc[si],
            "band_power": band_power,
            "erp": erp_info,
        })
        print("alpha=%.3f gamma=%.3f peak_amp=%.4f" % (
            band_power["alpha (8-13)"], band_power["gamma (30-80)"], erp_info["peak_amplitude"]))

    # ── 汇总 ──
    print("\n" + "=" * 100)
    print("%-10s %12s %10s  %8s %8s %8s %8s %8s  %10s %10s %12s" % (
        "Subject", "Consistency", "Gen Enc",
        "Delta", "Theta", "Alpha", "Beta", "Gamma",
        "ERP Amp", "Peak ms", "Trial Var"))
    print("-" * 100)
    for r in results:
        bp = r["band_power"]
        erp = r["erp"]
        print("%-10s %12.4f %10.1f  %8.3f %8.3f %8.3f %8.3f %8.3f  %10.4f %10.1f %12.6f" % (
            r["subject"], r["consistency"], r["gen_enc"],
            bp["delta (1-4)"], bp["theta (4-8)"], bp["alpha (8-13)"],
            bp["beta (13-30)"], bp["gamma (30-80)"],
            erp["peak_amplitude"], erp["peak_latency_ms"], erp["mean_trial_variance"]))
    print("=" * 100)

    # 相关性：频段能量 vs 生成式编码
    print("\n--- Band power vs Generative Encoding (Pearson r) ---")
    for band in BANDS:
        vals = [r["band_power"][band] for r in results]
        r_val = np.corrcoef(vals, gen_enc)[0, 1]
        print("  %s : r = %+.3f" % (band, r_val))

    print("\n--- ERP features vs Generative Encoding (Pearson r) ---")
    for feat in ["peak_amplitude", "peak_latency_ms", "mean_trial_variance"]:
        vals = [r["erp"][feat] for r in results]
        r_val = np.corrcoef(vals, gen_enc)[0, 1]
        print("  %s : r = %+.3f" % (feat, r_val))

    # sub-09 vs others
    sub09 = results[8]
    others = [r for r in results if r["subject"] != "sub-09"]
    print("\n--- sub-09 vs Others ---")
    for band in BANDS:
        ovals = [r["band_power"][band] for r in others]
        print("  %s : sub-09=%.3f  others_mean=%.3f±%.3f" % (
            band, sub09["band_power"][band], np.mean(ovals), np.std(ovals)))
    print("  ERP peak amp : sub-09=%.4f  others=%.4f±%.4f" % (
        sub09["erp"]["peak_amplitude"],
        np.mean([r["erp"]["peak_amplitude"] for r in others]),
        np.std([r["erp"]["peak_amplitude"] for r in others])))
    print("  Trial var    : sub-09=%.6f  others=%.6f±%.6f" % (
        sub09["erp"]["mean_trial_variance"],
        np.mean([r["erp"]["mean_trial_variance"] for r in others]),
        np.std([r["erp"]["mean_trial_variance"] for r in others])))


if __name__ == "__main__":
    main()
