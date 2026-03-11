import numpy as np
import wave
import csv
import json
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

def _read_wav_mono(path: str) -> Tuple[np.ndarray, int]:
    with wave.open(path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        if n_frames == 0:
            raise ValueError("Empty WAV: no frames.")
        raw = wf.readframes(n_frames)
    if sampwidth == 1:
        dtype = np.uint8
        data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sampwidth == 2:
        dtype = np.int16
        data = np.frombuffer(raw, dtype=dtype).astype(np.float32) / 32768.0
    elif sampwidth in (3, 4):
        dtype = np.int32
        data = np.frombuffer(raw, dtype=dtype).astype(np.float32) / (2**31)
    else:
        raise ValueError(f"Unsupported sampwidth: {sampwidth}")
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    max_abs = float(np.max(np.abs(data))) or 1.0
    data = (data / max_abs).astype(np.float32)
    return data, int(framerate)

def _resample_linear(x: np.ndarray, src_sr: int, tgt_sr: int) -> np.ndarray:
    if src_sr == tgt_sr or x.size == 0:
        return x.astype(np.float32, copy=False)
    ratio = tgt_sr / float(src_sr)
    n_tgt = max(1, int(np.floor(len(x) * ratio)))
    xp = np.linspace(0.0, 1.0, num=len(x), dtype=np.float64)
    xq = np.linspace(0.0, 1.0, num=n_tgt, dtype=np.float64)
    y = np.interp(xq, xp, x.astype(np.float64))
    return y.astype(np.float32)

def _frame_hop(x: np.ndarray, sr: int, frame_ms: float, hop_ms: float):
    frame_size = int(sr * frame_ms / 1000.0)
    hop_size = int(sr * hop_ms / 1000.0)
    if len(x) < frame_size:
        pad = np.zeros(frame_size, dtype=np.float32)
        pad[:len(x)] = x
        yield pad, sr
        return
    for start in range(0, len(x) - frame_size + 1, hop_size):
        yield x[start:start + frame_size], sr

def _hann(n: int) -> np.ndarray:
    return np.hanning(n).astype(np.float32) if n > 1 else np.ones(1, dtype=np.float32)

def _magnitude_spectrum(x: np.ndarray, samplerate: int, fft_size: Optional[int] = None,
                        smooth_len: int = 5, noise_floor_q: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    n = len(x)
    if fft_size is None:
        target = max(512, int(2 ** np.ceil(np.log2(n))))
        fft_size = min(target, 16384)
    if n < fft_size:
        pad = np.zeros(fft_size, dtype=np.float32)
        pad[:n] = x
        xw = pad
    else:
        xw = x[:fft_size]
    win = _hann(len(xw))
    X = np.fft.rfft(xw * win, n=fft_size)
    mag = np.abs(X).astype(np.float32)
    mag[0] = 0.0
    mag = np.log1p(mag)
    kernel = np.ones(smooth_len, dtype=np.float32) / float(smooth_len)
    env = np.convolve(mag, kernel, mode='same') + 1e-6
    mag_w = mag / env
    floor = np.quantile(mag_w, noise_floor_q)
    mag_w = np.clip(mag_w - floor, 0.0, None)
    norm = np.linalg.norm(mag_w) or 1.0
    mag_n = mag_w / norm
    freqs = np.fft.rfftfreq(fft_size, d=1.0 / samplerate).astype(np.float32)
    return mag_n, freqs

def _find_peaks(mag: np.ndarray, freqs: np.ndarray,
                min_hz: float, max_hz: float, top_k: int,
                threshold_quantile: float = 0.90) -> Tuple[np.ndarray, np.ndarray]:
    mask = (freqs >= min_hz) & (freqs <= max_hz)
    cand_mags = mag[mask]
    cand_freqs = freqs[mask]
    if cand_mags.size == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    thresh = float(np.quantile(cand_mags, threshold_quantile))
    idx = np.where(cand_mags >= thresh)[0]
    if idx.size == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    order = np.argsort(cand_mags[idx])[::-1][:top_k]
    sel_mags = cand_mags[idx][order].astype(np.float32)
    sel_freqs = cand_freqs[idx][order].astype(np.float32)
    return sel_freqs, sel_mags

def spectral_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    a = a[:n]
    b = b[:n]
    denom = (np.linalg.norm(a) or 1.0) * (np.linalg.norm(b) or 1.0)
    return float(np.dot(a, b) / denom)

def peak_overlap_score(freqs_a: np.ndarray, freqs_b: np.ndarray, tol_hz: float = 5.0) -> float:
    if len(freqs_a) == 0 or len(freqs_b) == 0:
        return 0.0
    hits = 0
    for fa in freqs_a:
        if np.any(np.abs(freqs_b - fa) <= tol_hz):
            hits += 1
    return hits / max(1, len(freqs_a))

@dataclass
class TwinTrustConfig:
    target_samplerate: int = 16000
    frame_ms: float = 200.0
    hop_ms: float = 100.0
    min_hz: float = 40.0
    max_hz: float = 8000.0
    top_k_peaks: int = 12
    smooth_len: int = 7
    noise_floor_q: float = 0.05
    peak_tol_hz: float = 5.0
    alpha_cosine: float = 0.65
    alpha_peaks: float = 0.35

@dataclass
class SpectralSignature:
    fft_size: int
    samplerate: int
    ref_vector: np.ndarray
    peak_freqs: np.ndarray
    peak_mags: np.ndarray

class TwinFrequencyTrust:
    def __init__(self, signature: SpectralSignature, cfg: Optional[TwinTrustConfig] = None):
        self.sig = signature
        self.cfg = cfg or TwinTrustConfig()

    @staticmethod
    def from_wav_reference(ref_wav: str, cfg: Optional[TwinTrustConfig] = None, ref_frame_ms: float = 400.0) -> "TwinFrequencyTrust":
        cfg = cfg or TwinTrustConfig()
        x, sr = _read_wav_mono(ref_wav)
        x = _resample_linear(x, sr, cfg.target_samplerate)
        frames = list(_frame_hop(x, cfg.target_samplerate, ref_frame_ms, ref_frame_ms))
        n_avg = min(5, len(frames))
        mags = []
        freqs = None
        for i in range(n_avg):
            frame, _ = frames[i]
            mag, freqs = _magnitude_spectrum(frame, cfg.target_samplerate,
                                             smooth_len=cfg.smooth_len,
                                             noise_floor_q=cfg.noise_floor_q)
            mags.append(mag)
        ref_vec = np.mean(np.stack(mags, axis=0), axis=0).astype(np.float32)
        ref_vec = ref_vec / (np.linalg.norm(ref_vec) or 1.0)
        peak_freqs, peak_mags = _find_peaks(ref_vec, freqs, cfg.min_hz, cfg.max_hz, cfg.top_k_peaks)
        sig = SpectralSignature(fft_size=(len(ref_vec) - 1) * 2, samplerate=cfg.target_samplerate,
                                ref_vector=ref_vec, peak_freqs=peak_freqs, peak_mags=peak_mags)
        return TwinFrequencyTrust(sig, cfg)

    def score_frame(self, frame: np.ndarray, samplerate: int) -> Dict[str, float]:
        f = _resample_linear(frame, samplerate, self.cfg.target_samplerate)
        mag, freqs = _magnitude_spectrum(f, self.cfg.target_samplerate, fft_size=self.sig.fft_size,
                                         smooth_len=self.cfg.smooth_len,
                                         noise_floor_q=self.cfg.noise_floor_q)
        cos = spectral_cosine_similarity(mag, self.sig.ref_vector)
        pf, pm = _find_peaks(mag, freqs, self.cfg.min_hz, self.cfg.max_hz, self.cfg.top_k_peaks)
        peak_score = peak_overlap_score(pf, self.sig.peak_freqs, tol_hz=self.cfg.peak_tol_hz)
        trust = self.cfg.alpha_cosine * cos + self.cfg.alpha_peaks * peak_score
        return {"cosine": float(cos), "peak_overlap": float(peak_score), "trust": float(trust)}

    def stream_score_wav(self, wav_path: str) -> List[Dict[str, float]]:
        x, sr = _read_wav_mono(wav_path)
        x = _resample_linear(x, sr, self.cfg.target_samplerate)
        scores = []
        for frame, srr in _frame_hop(x, self.cfg.target_samplerate, self.cfg.frame_ms, self.cfg.hop_ms):
            s = self.score_frame(frame, self.cfg.target_samplerate)
            scores.append(s)
        return scores

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Twin Frequency Trust v2: robust spectral twin detection.")
    parser.add_argument("--ref", required=True, help="Path to reference WAV file.")
    parser.add_argument("--test", required=True, help="Path to test WAV file.")
    parser.add_argument("--csv", type=str, default="", help="Optional CSV output file.")
    args = parser.parse_args()

    cfg = TwinTrustConfig()
    model = TwinFrequencyTrust.from_wav_reference(args.ref, cfg=cfg)
    scores = model.stream_score_wav(args.test)
    print(json.dumps(scores[:10], indent=2))
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_index", "cosine", "peak_overlap", "trust"])
            for i, s in enumerate(scores):
                w.writerow([i, s["cosine"], s["peak_overlap"], s["trust"]])
