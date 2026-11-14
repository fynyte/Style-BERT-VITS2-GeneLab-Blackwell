import warnings

import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn

warnings.filterwarnings(action="ignore")
MAX_WAV_VALUE = 32768.0


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def _get_hann_window(win_size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    key = (win_size, dtype, device)
    win = hann_window.get(key)
    if win is None:
        win = torch.hann_window(win_size).to(dtype=dtype, device=device)
        hann_window[key] = win
        hann_window[f"{win_size}_{dtype}_{device}"] = win
    return win


def _get_mel_basis(
    sampling_rate: int,
    n_fft: int,
    num_mels: int,
    fmin: float,
    fmax,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    target_dtype = (
        torch.get_autocast_gpu_dtype()
        if device.type == "cuda" and torch.is_autocast_enabled()
        else dtype
    )
    key = (sampling_rate, n_fft, num_mels, fmin, fmax, dtype, target_dtype, device)
    basis = mel_basis.get(key)
    if basis is None:
        base_key = (sampling_rate, n_fft, num_mels, fmin, fmax, dtype, dtype, device)
        base = mel_basis.get(base_key)
        if base is None:
            mel = librosa_mel_fn(
                sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
            )
            base = torch.from_numpy(mel).to(dtype=dtype, device=device)
            mel_basis[base_key] = base
            mel_basis[f"{fmax}_{dtype}_{device}"] = base
        basis = base if target_dtype == dtype else base.to(dtype=target_dtype)
        mel_basis[key] = basis
    return basis


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    win = _get_hann_window(win_size, y.dtype, y.device)
    pad = (n_fft - hop_size) // 2
    y = torch.nn.functional.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=win,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    basis = _get_mel_basis(
        sampling_rate, n_fft, num_mels, fmin, fmax, spec.dtype, spec.device
    )
    spec = torch.matmul(basis, spec)
    spec = spectral_normalize_torch(spec)
    return spec


def mel_spectrogram_torch(
    y,
    n_fft,
    num_mels,
    sampling_rate,
    hop_size,
    win_size,
    fmin,
    fmax,
    center=False,
    *,
    check_range=True,
):
    if check_range:
        if torch.min(y) < -1.0:
            print("min value is ", torch.min(y))
        if torch.max(y) > 1.0:
            print("max value is ", torch.max(y))

    basis = _get_mel_basis(
        sampling_rate, n_fft, num_mels, fmin, fmax, y.dtype, y.device
    )
    win = _get_hann_window(win_size, y.dtype, y.device)

    pad = (n_fft - hop_size) // 2
    y = torch.nn.functional.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=win,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    spec = torch.matmul(basis, spec)
    spec = spectral_normalize_torch(spec)

    return spec
