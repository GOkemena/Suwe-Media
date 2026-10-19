"""Suwe voiceover standard (approved by Great, 19 Oct 2026, episode 1 "21j light").

Usage: python3 voice_standard.py voice.m4a beat.wav video_no_audio.mp4 out.mp4 [key_pcs]
  key_pcs: comma list of pitch classes for the beat's key (default A minor / C major: 9,11,0,2,4,5,7)

Chain:
 1. Clean: mono, highpass 85Hz, noise reduction, warmth +2.5dB @220Hz, cut harshness -4dB @3.2k and -3.5dB @6.5k,
    compressor, loudnorm.
 2. Light autotune (pyworld): snap toward beat key at 55% strength, notes held ~100ms (median 21 frames),
    smoothed glides (gaussian sigma 3), then blend 80% tuned + 20% original voice.
 3. Voice polish: highpass 90, -2.5dB @300Hz, +2dB @2.5k, de-esser, compressor 3:1, very light echo (0.06), loudnorm -15 LUFS.
 4. Beat: volume 0.40, -4dB @2.2k (room for voice), -2dB @300Hz; ducks under voice (sidechain 2.5:1), fade out last 1.6s.
 5. Every exported video MUST have audible audio. Never post a silent video.
"""
import sys, subprocess, numpy as np, soundfile as sf, pyworld as pw
from scipy.ndimage import median_filter, gaussian_filter1d
voice, beat, video, out = sys.argv[1:5]
scale = [int(p) for p in (sys.argv[5] if len(sys.argv) > 5 else "9,11,0,2,4,5,7").split(",")]
def ff(*a): subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *a], check=True)
ff("-i", voice, "-af", "pan=mono|c0=0.5*c0+0.5*c1,highpass=f=85,afftdn=nf=-28,equalizer=f=220:t=o:w=1.2:g=2.5,"
   "equalizer=f=3200:t=o:w=1.5:g=-4,equalizer=f=6500:t=o:w=1.5:g=-3.5,acompressor=threshold=-20dB:ratio=3:attack=8:release=120:makeup=2,"
   "loudnorm=I=-16:TP=-1.5:LRA=9,aresample=44100", "_clean.wav")
x, sr = sf.read("_clean.wav"); x = (x.mean(1) if x.ndim > 1 else x).astype(np.float64)
f0, tt = pw.harvest(x, sr, f0_floor=60, f0_ceil=400, frame_period=5)
sp = pw.cheaptrick(x, f0, tt, sr); ap = pw.d4c(x, f0, tt, sr)
notes = np.array([m for m in range(30, 80) if m % 12 in scale])
v = f0 > 0; midi = np.zeros_like(f0); midi[v] = 69 + 12 * np.log2(f0[v] / 440)
target = np.zeros_like(f0)
for i in np.where(v)[0]: target[i] = notes[np.argmin(np.abs(notes - midi[i]))]
tm = median_filter(target, size=21)
corr = gaussian_filter1d(np.where(v, np.where(tm > 0, tm, target) - midi, 0), 3)
o = f0.copy(); o[v] = 440 * 2 ** ((midi[v] + 0.55 * corr[v] - 69) / 12)
y = pw.synthesize(o, sp, ap, sr, frame_period=5); n = min(len(y), len(x))
y = 0.8 * y[:n] / np.max(np.abs(y)) + 0.2 * x[:n] / np.max(np.abs(x))
sf.write("_tuned.wav", y / np.max(np.abs(y)) * 0.9, sr)
ff("-i", "_tuned.wav", "-af", "highpass=f=90,equalizer=f=300:t=o:w=1:g=-2.5,equalizer=f=2500:t=o:w=1.2:g=2,deesser=i=0.4,"
   "acompressor=threshold=-22dB:ratio=3:attack=5:release=80:makeup=2,aecho=0.8:0.2:40:0.06,loudnorm=I=-15:TP=-1.5:LRA=6",
   "-ac", "2", "-ar", "44100", "_vx.wav")
dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", video],
                           capture_output=True, text=True).stdout)
ff("-i", "_vx.wav", "-i", beat, "-filter_complex",
   "[1:a]volume=0.40,equalizer=f=2200:t=o:w=1.5:g=-4,equalizer=f=300:t=o:w=1:g=-2[m];"
   "[m][0:a]sidechaincompress=threshold=0.06:ratio=2.5:attack=20:release=350[md];[0:a]apad=pad_dur=5[vv];"
   f"[vv][md]amix=inputs=2:duration=longest:normalize=0,atrim=0:{dur},afade=t=out:st={dur-1.6}:d=1.6,alimiter=limit=0.95", "_mix.wav")
ff("-i", video, "-i", "_mix.wav", "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
   "-shortest", "-movflags", "+faststart", out)
print("done", out)
