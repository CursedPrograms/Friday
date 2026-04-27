#!/usr/bin/env python3

import os
import sys

# Disable torch.compile / TorchDynamo entirely — Triton is not available on
# Windows, so every compile attempt would fail and fall back to eager anyway.
# Setting this before any torch import avoids all compile overhead and noise.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
import time
import tempfile
import subprocess
import requests
import wave
import numpy as np
from rich.console import Console
from rich.panel import Panel
import threading
import json
import math
import socket
import ipaddress
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.markup import escape
import random
import surveillance

import pygame
import psutil

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except Exception as _cv2_err:
    CV2_AVAILABLE = False
    _cv2_import_error = str(_cv2_err)
else:
    _cv2_import_error = ""

try:
    from scapy.all import ARP, Ether, srp, conf as scapy_conf
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

console = Console()

# ==================== PLATFORM ====================
IS_WINDOWS = sys.platform == "win32"

# ==================== PATHS ====================
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR  = os.path.join(BASE_DIR, "audio")
VOICES_DIR = os.path.join(BASE_DIR, "voices")
IMG_DIR    = os.path.join(BASE_DIR, "images")
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
AUDIO_FILE = os.path.join(AUDIO_DIR, "stt.wav")
WAKE_FILE  = os.path.join(AUDIO_DIR, "wake.wav")

# MuseTalk reference video for lipsync
MUSETALK_TALK_VIDEO = os.path.join(VIDEOS_DIR, "musetalk_talk.mp4")
# Sleeping video shown while in sleep state
SLEEPING_VIDEO      = os.path.join(VIDEOS_DIR, "sleeping.mp4")
# MuseTalk output goes here (temp, overwritten each turn)
MUSETALK_OUT_DIR    = os.path.join(BASE_DIR, "musetalk_out")
os.makedirs(MUSETALK_OUT_DIR, exist_ok=True)

# Pre-generated startup lipsync video + audio (generated once, reused every run)
STARTUP_TEXT = "ComCentre online. DREAM is ready. Say Hey DREAM to wake me."
STARTUP_VID  = os.path.join(VIDEOS_DIR, "startup_intro.mp4")
STARTUP_WAV  = os.path.join(AUDIO_DIR,  "startup.wav")

# Lipsync cache: short fixed responses pre-generated on first use
# {text: (wav_path, vid_path)} — populated by _init_lipsync_cache() at startup
_lipsync_cache: dict = {}
_CACHE_YES_WAV = os.path.join(AUDIO_DIR,  "cache_yes.wav")
_CACHE_YES_VID = os.path.join(VIDEOS_DIR, "cache_yes_lipsync.mp4")


def _init_lipsync_cache():
    """Populate _lipsync_cache with any pre-built response videos."""
    entries = [
        ("Yes?",       _CACHE_YES_WAV, _CACHE_YES_VID),
        (STARTUP_TEXT, STARTUP_WAV,    STARTUP_VID),
    ]
    for text, wav, vid in entries:
        if os.path.exists(wav) and os.path.exists(vid):
            _lipsync_cache[text] = (wav, vid)
            console.print(f"[dim]Lipsync cache: {os.path.basename(vid)}[/dim]")


def _save_lipsync_cache(text: str, wav_src: str, vid_src: str,
                        wav_dst: str, vid_dst: str):
    """Copy generated wav+vid to permanent cache paths and register them."""
    import shutil
    try:
        shutil.copy(wav_src, wav_dst)
        shutil.copy(vid_src, vid_dst)
        _lipsync_cache[text] = (wav_dst, vid_dst)
        console.print(f"[green]Cached lipsync:[/green] {os.path.basename(vid_dst)}")
    except Exception as e:
        console.print(f"[yellow]Cache save failed: {e}[/yellow]")

if IS_WINDOWS:
    VENV_DIR  = os.path.join(BASE_DIR, "venv311")
    PIPER_BIN = os.path.join(VENV_DIR, "Scripts", "piper.exe")
else:
    VENV_DIR  = os.path.join(BASE_DIR, "venv")
    PIPER_BIN = os.path.join(VENV_DIR, "bin", "piper")

os.makedirs(AUDIO_DIR, exist_ok=True)

import surveillance

surveillance.start_surveillance()
surveillance.start_motion_detection()

# ==================== CONFIG ====================
OLLAMA_URL     = "http://localhost:11434/api/generate"
MODEL          = "phi3:mini"
SAMPLE_RATE    = 16000
CHANNELS       = 1
RECORD_SECONDS = 16
WAKE_SECONDS   = 3
RMS_THRESHOLD  = 200

FLIRT_IDLE_TIMEOUT = 600   # 10 minutes — flirt attention grab
SLEEP_IDLE_TIMEOUT = 900   # 15 minutes — fall asleep

WAKE_WORDS = [
    "hey dream", "hey, dream", "hi dream", "hi, dream",
    "okay dream", "ok dream", "dream",
]

# Wake words used only while sleeping
SLEEP_WAKE_WORDS = ["wake up", "wake up dream", "wake up, dream"]

WIFI_TRIGGERS = [
    "check wifi", "check the wifi", "wifi scan", "scan wifi",
    "scan the wifi", "who's on the wifi", "who is on the wifi",
    "check network", "network scan", "check connections",
    "what devices", "list devices", "show devices",
]

STATS_TRIGGERS = [
    "system stats", "cpu usage", "ram usage", "memory usage",
    "disk usage", "how's the system", "system status",
    "how are you doing", "check stats", "check the stats",
    "temperature", "cpu temp", "how hot", "system health",
]

with open(os.path.join(BASE_DIR, "config.json")) as f:
    config = json.load(f)

CHAR_NAME     = config["Config"]["DREAM"]["CharName"]
SYSTEM_PROMPT = config["Config"]["DREAM"]["SystemPrompt"].format(name=CHAR_NAME)

print("CharName:", CHAR_NAME)
print("SystemPrompt:", SYSTEM_PROMPT)
console.print(f"[cyan]CUDA available:[/cyan] {CUDA_AVAILABLE} "
              f"({'GPU: ' + torch.cuda.get_device_name(0) if CUDA_AVAILABLE else 'CPU only'})")

# ==================== VOICE MODEL ====================

def find_voice_model():
    if not os.path.isdir(VOICES_DIR):
        return None
    for fname in sorted(os.listdir(VOICES_DIR)):
        if fname.endswith(".onnx"):
            return os.path.join(VOICES_DIR, fname)
    return None

VOICE_MODEL = find_voice_model()

# ==================== PIPER SAMPLE RATE DETECTION ====================

def get_piper_sample_rate():
    if VOICE_MODEL is None:
        return 22050
    json_path = VOICE_MODEL + ".json"
    if os.path.exists(json_path):
        try:
            with open(json_path, encoding="utf-8") as fh:
                cfg = json.load(fh)
            sr = cfg.get("audio", {}).get("sample_rate")
            if sr:
                console.print(f"[dim]Piper voice sample rate: {sr} Hz[/dim]")
                return int(sr)
        except Exception as e:
            console.print(f"[yellow]Could not read piper json config: {e}[/yellow]")
    console.print("[yellow]Piper json config not found — assuming 22050 Hz[/yellow]")
    return 22050

# ==================== SHARED STATE ====================
_state = {
    "value":        "idle",
    "running":      True,
    "wake_active":  False,
    "last_wake_ts": time.time(),
    "flirt_played": False,
    "force_video":  None,
    "sleeping":     False,          # True when in sleep state
    "deep_dream_thread": None,      # background deep dream worker
}

def set_state(s):
    _state["value"] = s

def touch_interaction():
    """Call whenever the user actually interacts — resets all idle timers."""
    _state["last_wake_ts"] = time.time()
    _state["flirt_played"] = False

# ==================== VIDEO POOL HELPERS ====================

def _glob_videos(prefix):
    if not os.path.isdir(VIDEOS_DIR):
        return []
    return sorted(
        os.path.join(VIDEOS_DIR, f)
        for f in os.listdir(VIDEOS_DIR)
        if f.startswith(prefix) and f.endswith(".mp4")
    )

VIDEO_POOLS = {}

def build_video_pools():
    global VIDEO_POOLS
    VIDEO_POOLS = {
        "idle":       _glob_videos("idle"),
        "listening":  _glob_videos("listening"),
        "thinking":   _glob_videos("thinking"),
        "talking":    _glob_videos("talking"),
        "flirtytalk": _glob_videos("flirtytalk"),
        "sleeping":   [SLEEPING_VIDEO] if os.path.exists(SLEEPING_VIDEO) else [],
    }
    for k, v in VIDEO_POOLS.items():
        console.print(f"[dim]  {k}: {len(v)} video(s)[/dim]")

# ==================== DEEP DREAM (background, runs while sleeping) ====================

def _run_deep_dream_background():
    """
    Spawns deep_dream_batch processing while DREAM is asleep.
    Runs in a daemon thread so it stops when the main process exits.
    Will stop as soon as _state["sleeping"] flips back to False.
    """
    try:
        # Import lazily so deep dream deps don't affect startup if missing
        from types import ModuleType
        import importlib.util

        batch_path = os.path.join(BASE_DIR, "deep_dream_batch.py")
        if not os.path.exists(batch_path):
            console.print("[yellow]deep_dream_batch.py not found — skipping deep dream[/yellow]")
            return

        console.print("[magenta]Deep dream started (sleeping)[/magenta]")

        # Build a dreams output directory timestamped to this sleep session
        dream_ts = time.strftime("%Y%m%d_%H%M%S")
        dream_input = os.path.join(BASE_DIR, "output", "dreams", f"dream_{dream_ts}_interpolate")
        os.makedirs(dream_input, exist_ok=True)

        # Run deep_dream_batch as a subprocess so it can use its own TF session
        # and we can terminate it cleanly when woken
        proc = subprocess.Popen(
            [sys.executable, batch_path, dream_input],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True
        )

        while _state["sleeping"] and proc.poll() is None:
            line = proc.stdout.readline()
            if line:
                console.print(f"[dim][DeepDream] {line.rstrip()}[/dim]")
            time.sleep(0.05)

        if proc.poll() is None:
            proc.terminate()
            console.print("[magenta]Deep dream terminated (woke up)[/magenta]")
        else:
            console.print("[magenta]Deep dream finished[/magenta]")

    except Exception as e:
        console.print(f"[red]Deep dream error: {e}[/red]")


def enter_sleep():
    """Transition into sleep state."""
    if _state["sleeping"]:
        return
    console.print("[blue]DREAM is falling asleep...[/blue]")
    _state["sleeping"] = True
    set_state("sleeping")
    # Start deep dream in background
    t = threading.Thread(target=_run_deep_dream_background, daemon=True)
    t.start()
    _state["deep_dream_thread"] = t


def exit_sleep():
    """Wake DREAM back up."""
    if not _state["sleeping"]:
        return
    console.print("[bold green]DREAM is waking up![/bold green]")
    _state["sleeping"] = False
    set_state("idle")
    touch_interaction()

# ==================== MUSETALK LIPSYNC ====================

# MuseTalk models are loaded once at startup (same pattern as musetalk.py)
_musetalk_loaded = False
_mt_vae = _mt_unet = _mt_pe = _mt_audio_processor = _mt_whisper = None
_mt_device = None
_mt_weight_dtype = None
_mt_timesteps = None
_musetalk_lock = threading.Lock()   # prevents concurrent GPU inference

# Reference-video caches — populated once on first run_musetalk() call
_mt_ref_cycle  = None   # (frame_list_cycle, coord_list_cycle, latent_list_cycle)
_mt_ref_fp     = None   # FaceParsing instance (reused across calls)
_mt_ref_fps    = 25.0   # fps of musetalk_talk.mp4
_mt_timesteps = None

def _load_musetalk():
    """Lazy-load MuseTalk models once. Skips gracefully if not installed."""
    global _musetalk_loaded, _mt_vae, _mt_unet, _mt_pe
    global _mt_audio_processor, _mt_whisper
    global _mt_device, _mt_weight_dtype, _mt_timesteps

    if _musetalk_loaded:
        return True

    # scripts/ directory contains the musetalk package
    scripts_dir = os.path.join(BASE_DIR, "scripts")
    if not os.path.isdir(scripts_dir):
        console.print("[yellow]scripts/ not found — lipsync disabled[/yellow]")
        return False
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    models_dir = os.path.join(BASE_DIR, "models")

    try:
        import torch
        from transformers import WhisperModel
        from musetalk.utils.audio_processor import AudioProcessor
        from musetalk.models.vae import VAE
        from musetalk.models.unet import UNet, PositionalEncoding

        _mt_device       = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
        _mt_weight_dtype = torch.float16 if CUDA_AVAILABLE else torch.float32

        # Instantiate directly with absolute paths (bypasses load_all_model's
        # relative-path construction that only works from the project root)
        _mt_vae  = VAE(model_path=os.path.join(models_dir, "sd-vae"))
        _mt_unet = UNet(
            unet_config=os.path.join(models_dir, "musetalkV15", "musetalk.json"),
            model_path=os.path.join(models_dir, "musetalkV15", "unet.pth"),
            device=_mt_device,
        )
        _mt_pe = PositionalEncoding(d_model=384)

        _mt_pe         = _mt_pe.to(_mt_device).to(_mt_weight_dtype)
        _mt_vae.vae    = _mt_vae.vae.to(_mt_device).to(_mt_weight_dtype)
        _mt_unet.model = _mt_unet.model.to(_mt_device).to(_mt_weight_dtype)

        whisper_path = os.path.join(models_dir, "whisper")
        _mt_audio_processor = AudioProcessor(feature_extractor_path=whisper_path)
        _mt_whisper = WhisperModel.from_pretrained(whisper_path)
        _mt_whisper = _mt_whisper.to(device=_mt_device, dtype=_mt_weight_dtype).eval()
        _mt_whisper.requires_grad_(False)

        _mt_timesteps = torch.tensor([0], device=_mt_device)
        console.print(f"[green]OK[/green] MuseTalk models on {_mt_device}")

        # Build reference-video cache while still in this background thread.
        # _musetalk_loaded is only set True once the cache is ready, so
        # run_musetalk() will never be called before this completes.
        _build_ref_cache()

        _musetalk_loaded = True
        console.print("[green]OK[/green] MuseTalk ready (models + reference cache)")
        return True

    except Exception as e:
        console.print(f"[yellow]MuseTalk load failed ({e}) — lipsync disabled[/yellow]")
        import traceback; traceback.print_exc()
        return False


def _build_ref_cache():
    """
    Extract frames from musetalk_talk.mp4, detect face landmarks, and
    VAE-encode all reference frames.  Results are stored in the _mt_ref_*
    module globals and reused on every run_musetalk() call.
    """
    global _mt_ref_cycle, _mt_ref_fp, _mt_ref_fps

    if _mt_ref_cycle is not None:
        return
    if not os.path.exists(MUSETALK_TALK_VIDEO):
        console.print("[yellow]musetalk_talk.mp4 not found — no reference cache[/yellow]")
        return

    try:
        import glob, pickle, imageio
        from musetalk.utils.face_parsing  import FaceParsing
        from musetalk.utils.utils         import get_video_fps
        from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder

        extra_margin     = 10
        input_basename   = os.path.splitext(os.path.basename(MUSETALK_TALK_VIDEO))[0]
        frames_cache_dir = os.path.join(MUSETALK_OUT_DIR, input_basename + "_frames")
        coord_cache_path = os.path.join(MUSETALK_OUT_DIR, input_basename + ".pkl")

        # ── Extract reference frames to disk (used for landmark detection) ──
        if not os.path.isdir(frames_cache_dir) or not os.listdir(frames_cache_dir):
            console.print("[dim]Extracting reference frames...[/dim]")
            os.makedirs(frames_cache_dir, exist_ok=True)
            reader = imageio.get_reader(MUSETALK_TALK_VIDEO)
            for i, im in enumerate(reader):
                imageio.imwrite(f"{frames_cache_dir}/{i:08d}.png", im)

        input_img_list = sorted(glob.glob(
            os.path.join(frames_cache_dir, "*.[jpJP][pnPN]*[gG]")
        ))
        _mt_ref_fps = get_video_fps(MUSETALK_TALK_VIDEO)

        # ── Face landmarks (disk-cached across restarts) ──────────────────
        if os.path.exists(coord_cache_path):
            console.print("[dim]Loading cached face landmarks...[/dim]")
            with open(coord_cache_path, "rb") as f:
                coord_list = pickle.load(f)
            frame_list = read_imgs(input_img_list)
        else:
            console.print("[dim]Detecting face landmarks (one-time, ~30s)...[/dim]")
            coord_list, frame_list = get_landmark_and_bbox(input_img_list, 0)
            with open(coord_cache_path, "wb") as f:
                pickle.dump(coord_list, f)

        # ── VAE-encode all reference crops (GPU, cached in memory) ────────
        console.print("[dim]Encoding reference latents...[/dim]")
        latent_list = []
        for bbox, frame in zip(coord_list, frame_list):
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            y2   = min(y2 + extra_margin, frame.shape[0])
            crop = cv2.resize(frame[y1:y2, x1:x2], (256, 256),
                              interpolation=cv2.INTER_LANCZOS4)
            latent_list.append(_mt_vae.get_latents_for_unet(crop))

        _mt_ref_fp    = FaceParsing(left_cheek_width=90, right_cheek_width=90)
        _mt_ref_cycle = (
            frame_list  + frame_list[::-1],
            coord_list  + coord_list[::-1],
            latent_list + latent_list[::-1],
        )
        console.print("[green]Reference cache ready[/green]"
                      f" ({len(frame_list)} frames)")

    except Exception as e:
        console.print(f"[yellow]Reference cache failed: {e}[/yellow]")
        import traceback; traceback.print_exc()


def run_musetalk(audio_wav_path: str) -> str | None:
    """
    Run MuseTalk inference on audio_wav_path against MUSETALK_TALK_VIDEO.

    Optimisations vs. the naive version:
      • Reference frames, face-coords, VAE latents and FaceParsing are
        computed once and kept in module-level caches (_mt_ref_*).
      • No intermediate PNG files — composite frames are written directly
        to the output video with imageio.
      • No moviepy — audio is played by pygame separately, so the video
        is silent and does not need the heavy moviepy audio-mux step.
    """
    global _mt_ref_cycle, _mt_ref_fp, _mt_ref_fps

    if not os.path.exists(MUSETALK_TALK_VIDEO):
        console.print("[yellow]musetalk_talk.mp4 not found — lipsync skipped[/yellow]")
        return None
    if not _load_musetalk():
        return None

    try:
        import imageio
        import torch
        from musetalk.utils.blending      import get_image
        from musetalk.utils.utils         import datagen
        from musetalk.utils.preprocessing import coord_placeholder

        extra_margin = 10
        batch_size   = 8

        if _mt_ref_cycle is None:
            console.print("[yellow]Reference cache not ready — lipsync skipped[/yellow]")
            return None

        frame_list_cycle, coord_list_cycle, latent_list_cycle = _mt_ref_cycle

        # ── Audio features ────────────────────────────────────────────────────
        whisper_feats, librosa_length = _mt_audio_processor.get_audio_feature(audio_wav_path)
        whisper_chunks = _mt_audio_processor.get_whisper_chunk(
            whisper_feats, _mt_device, _mt_weight_dtype, _mt_whisper, librosa_length,
            fps=_mt_ref_fps,
            audio_padding_length_left=2, audio_padding_length_right=2,
        )

        output_vid = os.path.join(MUSETALK_OUT_DIR, f"speech_{int(time.time())}.mp4")
        writer = imageio.get_writer(
            output_vid, fps=25, codec="libx264",
            quality=7, pixelformat="yuv420p", macro_block_size=None,
        )

        # ── Single streaming pass: infer → composite → write per batch ────────
        # torch.inference_mode() disables autograd graph construction — no
        # gradient tracking overhead during forward passes.
        console.print("[dim]MuseTalk inference...[/dim]")
        fp  = _mt_ref_fp
        idx = 0
        gen = datagen(
            whisper_chunks     = whisper_chunks,
            vae_encode_latents = latent_list_cycle,
            batch_size         = batch_size,
            delay_frame        = 0,
            device             = _mt_device,
        )
        with torch.inference_mode():
            for wb, lb in gen:
                af   = _mt_pe(wb)
                lb   = lb.to(dtype=_mt_weight_dtype)
                pred = _mt_unet.model(lb, _mt_timesteps, encoder_hidden_states=af).sample
                for rf in _mt_vae.decode_latents(pred):
                    bbox      = coord_list_cycle[idx % len(coord_list_cycle)]
                    ori_frame = frame_list_cycle[idx % len(frame_list_cycle)].copy()
                    idx += 1
                    if bbox == coord_placeholder:
                        continue
                    x1, y1, x2, y2 = bbox
                    y2 = min(y2 + extra_margin, ori_frame.shape[0])
                    try:
                        rf_resized = cv2.resize(rf.astype(np.uint8), (x2 - x1, y2 - y1))
                    except Exception:
                        continue
                    composite = get_image(ori_frame, rf_resized, [x1, y1, x2, y2],
                                          mode="jaw", fp=fp)
                    writer.append_data(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
        writer.close()

        console.print(f"[green]MuseTalk →[/green] {os.path.basename(output_vid)}")
        return output_vid

    except Exception as e:
        console.print(f"[red]MuseTalk error: {e}[/red]")
        import traceback; traceback.print_exc()
        return None

# ==================== BANNER ====================

def startup_banner():
    piper_ok = os.path.exists(PIPER_BIN)
    voice_ok = VOICE_MODEL is not None
    console.print(Panel.fit(
        "[bold cyan]ComCentre v2.7[/bold cyan]\n"
        "[dim]DREAM - Local AI Voice Assistant[/dim]\n\n"
        f"[green]LLM:[/green]      {MODEL}\n"
        f"[green]STT:[/green]      Whisper tiny\n"
        f"[green]Wake:[/green]     'Hey DREAM'\n"
        f"[green]Sleep Wake:[/green] 'Wake Up'\n"
        f"[green]Piper:[/green]    {'[green]' + PIPER_BIN + '[/green]' if piper_ok else '[red]NOT FOUND[/red]'}\n"
        f"[green]Voice:[/green]    {'[green]' + os.path.basename(VOICE_MODEL) + '[/green]' if voice_ok else '[red]NOT FOUND[/red]'}\n"
        f"[green]MuseTalk:[/green] {'[green]musetalk_talk.mp4[/green]' if os.path.exists(MUSETALK_TALK_VIDEO) else '[yellow]musetalk_talk.mp4 not found[/yellow]'}\n"
        f"[green]CUDA:[/green]     {'[green]' + (torch.cuda.get_device_name(0) if CUDA_AVAILABLE else '') + '[/green]' if CUDA_AVAILABLE else '[yellow]CPU only[/yellow]'}\n"
        f"[green]Videos:[/green]   {VIDEOS_DIR}\n"
        f"[green]Platform:[/green] {'Windows' if IS_WINDOWS else 'Linux'}",
        border_style="cyan"
    ))
    if not piper_ok:
        console.print(f"[red]❌  Piper not found at: {PIPER_BIN}[/red]")
        if IS_WINDOWS:
            console.print("[yellow]    Run: pip install piper-tts  (inside your venv)[/yellow]")
        else:
            console.print("[yellow]    Run: pip install piper-tts[/yellow]")
        sys.exit(1)
    if not voice_ok:
        console.print(f"[red]❌  No .onnx voice model found in: {VOICES_DIR}[/red]")
        console.print("[yellow]    Download a voice from https://rhasspy.github.io/piper-samples/[/yellow]")
        sys.exit(1)
    console.print(f"[green]OK[/green] Piper ready")
    console.print(f"[green]OK[/green] Voice: {os.path.basename(VOICE_MODEL)}")
    if CV2_AVAILABLE:
        import cv2 as _cv2
        console.print(f"[green]OK[/green] OpenCV {_cv2.__version__}")
    else:
        console.print(f"[yellow]WARN[/yellow] opencv-python not importable: {_cv2_import_error}")

    build_video_pools()

# ==================== AUDIO ====================

def get_mic_samplerate():
    import sounddevice as sd
    return int(sd.query_devices(kind="input")["default_samplerate"])

def _record_clip(filepath, seconds):
    import sounddevice as sd
    from scipy.signal import resample_poly
    try:
        native_rate = get_mic_samplerate()
        audio = sd.rec(int(seconds * native_rate), samplerate=native_rate,
                       channels=CHANNELS, dtype="int16")
        sd.wait()
        audio_flat = audio[:, 0] if audio.ndim > 1 else audio.flatten()
        if native_rate != SAMPLE_RATE:
            g = math.gcd(SAMPLE_RATE, native_rate)
            audio_flat = resample_poly(audio_flat, SAMPLE_RATE // g,
                                       native_rate // g).astype(np.int16)
        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_flat.tobytes())
        return os.path.getsize(filepath) > 500
    except Exception as e:
        console.print(f"[red]Record error: {e}[/red]")
        return False

def check_audio_levels(filepath=None):
    fp = filepath or AUDIO_FILE
    if not os.path.exists(fp):
        return False
    try:
        with wave.open(fp, "rb") as wf:
            audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        return int(np.max(np.abs(audio))) > RMS_THRESHOLD
    except Exception:
        return False

# ==================== WHISPER ====================

_whisper_model = None

def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        console.print("[dim]Loading Whisper (CPU — keeps GPU free for MuseTalk)...[/dim]")
        import whisper
        _whisper_model = whisper.load_model("tiny", device="cpu")
        console.print("[green]OK[/green] Whisper ready (device=cpu)")
    return _whisper_model

def transcribe_file(filepath):
    try:
        result = get_whisper().transcribe(filepath, language="en", fp16=False)
        return result["text"].strip()
    except Exception as e:
        console.print(f"[red]STT error: {e}[/red]")
        return ""

# ==================== WAKE WORD LOOP ====================

def listen_for_wake_word():
    """
    Listens for wake words.
    - When sleeping: only reacts to SLEEP_WAKE_WORDS ("wake up")
    - When awake:    reacts to WAKE_WORDS ("hey dream") and wifi triggers
    """
    if not _state["sleeping"]:
        console.print("[dim]Listening for wake word 'Hey DREAM'...[/dim]")
    else:
        console.print("[dim]Sleeping — listening for 'Wake Up'...[/dim]")

    _state["wake_active"] = True
    if not _state["sleeping"]:
        set_state("idle")

    while _state["running"]:
        if not _record_clip(WAKE_FILE, WAKE_SECONDS):
            time.sleep(0.2)
            continue

        if not check_audio_levels(WAKE_FILE):
            continue

        text = transcribe_file(WAKE_FILE).lower().strip()
        if not text:
            continue

        console.print(f"[dim]Wake check: '{text}'[/dim]")

        # --- Sleeping mode: only "wake up" counts ---
        if _state["sleeping"]:
            if any(w in text for w in SLEEP_WAKE_WORDS):
                console.print("[bold green]Wake-up word detected — waking DREAM![/bold green]")
                _state["wake_active"] = False
                exit_sleep()
                return "__WAKE__"
            # Ignore everything else while sleeping
            continue

        # --- Awake mode ---
        if any(w in text for w in WAKE_WORDS):
            console.print("[bold green]Wake word detected![/bold green]")
            _state["wake_active"] = False
            touch_interaction()
            return "__WAKE__"

        if any(t in text for t in WIFI_TRIGGERS):
            _state["wake_active"] = False
            touch_interaction()
            return "__WIFI_SCAN__"

    _state["wake_active"] = False
    return None

# ==================== LLM ====================

def ask_llm(prompt, history):
    set_state("thinking")
    with console.status("[dim]Thinking...[/dim]", spinner="dots"):
        context = ""
        for msg in history[-6:]:
            role = "You" if msg["role"] == "assistant" else "Human"
            context += f"{role}: {msg['content']}\n"
        full_prompt = f"System: {SYSTEM_PROMPT}\n\n{context}Human: {prompt}\nYou:"
        payload = {
            "model": MODEL,
            "prompt": full_prompt,
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 150, "num_gpu": 20},
        }
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=120)
            if r.status_code != 200:
                return "Sorry, I encountered an error."
            response = r.json().get("response", "").strip()
            if "<think>" in response:
                end = response.find("</think>")
                if end != -1:
                    response = response[end + 8:].strip()
            if response.lower().startswith("dream:"):
                response = response[6:].strip()
            return response if response else "I didn't catch that."
        except requests.exceptions.Timeout:
            return "That took too long. Please try again."
        except Exception as e:
            return f"Error: {e}"

# ==================== SPEAK (Piper → MuseTalk → display) ====================

def _play_wav(wav_path: str):
    """Play a WAV through pygame.mixer.Sound and block until done.

    Uses Sound (not music) because mixer.music.get_busy() hangs indefinitely
    on some Windows audio stacks.  Falls back to a time.sleep duration guard
    so the caller is never blocked forever.
    """
    sound = None
    try:
        sound   = pygame.mixer.Sound(wav_path)
        channel = sound.play()
        if channel is None:
            console.print("[yellow]Audio: no free mixer channel[/yellow]")
            return
        deadline = time.time() + sound.get_length() + 2.0
        while channel.get_busy() and time.time() < deadline:
            time.sleep(0.05)
        channel.stop()
    except Exception as e:
        console.print(f"[red]Audio error: {e}[/red]")
    finally:
        del sound       # release pygame's file handle before caller deletes the file


def speak(text):
    """
    Fast path (cached): instant lipsync from pre-generated wav+video.
    Normal path: Piper TTS → MuseTalk runs in background while display stays
    in "thinking" state (hides inference delay) → switch to "talking" with
    audio + lipsync together once MuseTalk finishes.
    """
    if not text:
        return
    console.print(f"\n[bold cyan]DREAM:[/bold cyan] {text}\n")

    # ── Fast path: cached lipsync (Yes?, startup, etc.) ──────────────────────
    if text in _lipsync_cache:
        cached_wav, cached_vid = _lipsync_cache[text]
        if os.path.exists(cached_wav) and os.path.exists(cached_vid):
            set_state("talking")
            _state["force_video"] = cached_vid
            _play_wav(cached_wav)
            set_state("idle")
            return
        # Stale cache entry — files were deleted; fall through to normal path
        del _lipsync_cache[text]

    # ── Normal path ───────────────────────────────────────────────────────────
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=AUDIO_DIR)
    tmp.close()
    try:
        proc = subprocess.run(
            [PIPER_BIN, "-m", VOICE_MODEL, "-f", tmp.name],
            input=text.encode("utf-8"), capture_output=True, timeout=15,
        )
        if proc.returncode != 0:
            console.print(f"[red]Piper error: {proc.stderr.decode()}[/red]")
            return
        if not os.path.exists(tmp.name) or os.path.getsize(tmp.name) < 100:
            return

        # Measure audio duration to set a tight MuseTalk wait ceiling.
        try:
            import wave as _wave
            with _wave.open(tmp.name, 'rb') as _wf:
                audio_secs = _wf.getnframes() / _wf.getframerate()
        except Exception:
            audio_secs = 5.0

        mt_ready = threading.Event()

        if _musetalk_loaded and _musetalk_lock.acquire(blocking=False):
            import shutil as _shutil
            mt_wav = tmp.name + ".mt.wav"
            _shutil.copy(tmp.name, mt_wav)

            def _mt_thread():
                try:
                    vid = run_musetalk(mt_wav)
                    if vid and os.path.exists(vid):
                        # Always set force_video directly — this fires whether
                        # speak() is still waiting or has already played audio.
                        _state["force_video"] = vid
                        if text == "Yes?":
                            _save_lipsync_cache(text, mt_wav, vid,
                                                _CACHE_YES_WAV, _CACHE_YES_VID)
                        elif text == STARTUP_TEXT:
                            _save_lipsync_cache(text, mt_wav, vid,
                                                STARTUP_WAV, STARTUP_VID)
                except Exception as e:
                    console.print(f"[yellow]MuseTalk bg error: {e}[/yellow]")
                finally:
                    _musetalk_lock.release()
                    mt_ready.set()
                    if os.path.exists(mt_wav):
                        os.unlink(mt_wav)

            threading.Thread(target=_mt_thread, daemon=True).start()
        else:
            mt_ready.set()

        # Stay in "thinking" state while MuseTalk runs — the animation hides
        # the inference delay naturally.  Timeout is audio-length-derived so
        # we never wait longer than necessary on this hardware (RTX 2060:
        # ~3s overhead + ~2s per second of audio).  After timeout, audio plays
        # immediately and the background thread queues force_video when done.
        mt_timeout = max(20.0, audio_secs * 2.5 + 8.0)
        mt_ready.wait(timeout=mt_timeout)

        set_state("talking")
        _play_wav(tmp.name)

    except Exception as e:
        console.print(f"[red]TTS failed: {e}[/red]")
        import traceback; traceback.print_exc()
    finally:
        # Retry deletion — Windows briefly locks WAV files after pygame releases them
        for _ in range(10):
            try:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)
                break
            except PermissionError:
                time.sleep(0.05)
        set_state("idle")

# ==================== SYSTEM STATS ====================

_stats_cache = {"data": {}, "last": 0.0}

def get_system_stats():
    now = time.time()
    if now - _stats_cache["last"] < 2.0:
        return _stats_cache["data"]
    data = {}
    mem = psutil.virtual_memory()
    data["ram_used"]    = mem.used  / 1073741824
    data["ram_total"]   = mem.total / 1073741824
    data["ram_pct"]     = mem.percent
    data["cpu_pct"]     = psutil.cpu_percent(interval=None)
    data["cpu_cores"]   = psutil.cpu_count(logical=False) or 1
    data["cpu_threads"] = psutil.cpu_count(logical=True)  or 1

    data["cpu_temp"] = None
    if not IS_WINDOWS:
        try:
            temps = psutil.sensors_temperatures()
            all_t = []
            for key in ("coretemp", "k10temp", "cpu_thermal", "acpitz"):
                if key in temps:
                    all_t += [t.current for t in temps[key]]
            data["cpu_temp"] = max(all_t) if all_t else None
        except Exception:
            pass

    disk_path = "C:\\" if IS_WINDOWS else "/"
    disk = psutil.disk_usage(disk_path)
    data["disk_used"]  = disk.used  / 1073741824
    data["disk_total"] = disk.total / 1073741824
    data["disk_pct"]   = disk.percent

    _stats_cache["data"] = data
    _stats_cache["last"] = now
    return data

def build_stats_summary():
    s = get_system_stats()
    parts = [
        f"CPU is at {s['cpu_pct']:.0f} percent",
        f"RAM usage is {s['ram_used']:.1f} of {s['ram_total']:.1f} gigabytes, that's {s['ram_pct']:.0f} percent",
        f"disk is {s['disk_used']:.0f} of {s['disk_total']:.0f} gigabytes used",
    ]
    if s.get("cpu_temp"):
        parts.append(f"CPU temperature is {s['cpu_temp']:.0f} degrees Celsius")
    return ". ".join(parts) + "."

# ==================== WIFI SCANNER ====================

def _ping_host(ip):
    if IS_WINDOWS:
        cmd = ["ping", "-n", "1", "-w", "1000", str(ip)]
    else:
        cmd = ["ping", "-c", "1", "-W", "1", str(ip)]
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(ip) if r.returncode == 0 else None

def _get_hostname(ip):
    try:
        return socket.gethostbyaddr(ip)[0]
    except Exception:
        return ""

def _get_vendor(mac):
    try:
        r = requests.get(f"https://api.macvendors.com/{mac}", timeout=3)
        if r.status_code == 200:
            return r.text.strip()
    except Exception:
        pass
    return "Unknown"

def _is_randomized(mac):
    try:
        return bool(int(mac.split(":")[0], 16) & 0x02)
    except Exception:
        return False

def _guess_type(mac, hostname, vendor):
    if _is_randomized(mac):
        return "phone or tablet with randomized MAC"
    h = (hostname + vendor).lower()
    if any(x in h for x in ["iphone", "apple", "ipad"]):                          return "Apple device"
    if any(x in h for x in ["samsung", "android", "xiaomi", "huawei"]):           return "Android device"
    if any(x in h for x in ["router", "gateway", "dlink", "tp-link", "asus", "netgear"]): return "router"
    if any(x in h for x in ["windows", "intel", "realtek"]):                      return "Windows PC"
    if any(x in h for x in ["ubuntu", "linux", "debian", "raspi"]):               return "Linux device"
    if any(x in h for x in ["tv", "cast", "roku", "echo", "alexa"]):              return "smart TV or IoT device"
    if vendor and vendor != "Unknown":
        return vendor
    return "unknown device"

def _read_arp_cache():
    arp_cache = {}
    try:
        out = subprocess.check_output(["arp", "-a"], text=True, stderr=subprocess.DEVNULL)
        if IS_WINDOWS:
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    ip  = parts[0].strip()
                    mac = parts[1].strip().replace("-", ":").upper()
                    try:
                        ipaddress.ip_address(ip)
                        if mac not in ("FF:FF:FF:FF:FF:FF", ""):
                            arp_cache[ip] = mac
                    except ValueError:
                        pass
        else:
            for line in out.splitlines():
                p = line.split()
                if "lladdr" in p and "FAILED" not in line and "INCOMPLETE" not in line:
                    arp_cache[p[0]] = p[p.index("lladdr") + 1]
    except Exception as e:
        console.print(f"[yellow]ARP cache read warning: {e}[/yellow]")
    return arp_cache

def run_wifi_scan():
    set_state("thinking")
    speak("Scanning the network. Give me a moment.")

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        my_ip  = s.getsockname()[0]
        s.close()
        subnet = my_ip.rsplit(".", 1)[0] + ".0/24"

        console.print(f"[dim]Scanning {subnet}...[/dim]")

        network = ipaddress.ip_network(subnet, strict=False)
        with ThreadPoolExecutor(max_workers=80) as ex:
            list(ex.map(_ping_host, network.hosts()))

        arp_direct = {}
        if SCAPY_AVAILABLE:
            scapy_conf.verb = 0
            pkt = Ether(dst="ff:ff:ff:ff:ff:ff") / ARP(pdst=subnet)
            answered, _ = srp(pkt, timeout=3, verbose=False, retry=1)
            arp_direct = {r.psrc: r.hwsrc for _, r in answered}

        arp_cache = _read_arp_cache()

        all_ips = set(arp_direct) | set(arp_cache)
        devices = []
        for ip in sorted(all_ips, key=lambda x: list(map(int, x.split(".")))):
            raw_mac  = (arp_direct.get(ip) or arp_cache.get(ip) or "??:??:??:??:??:??")
            mac      = raw_mac.replace("-", ":").upper()
            hostname = _get_hostname(ip)
            vendor   = _get_vendor(mac) if not _is_randomized(mac) else "N/A"
            dtype    = _guess_type(mac, hostname, vendor)
            devices.append({
                "ip": ip, "mac": mac, "hostname": hostname or "",
                "vendor": vendor, "type": dtype, "me": ip == my_ip,
            })

        count  = len(devices)
        phones = sum(1 for d in devices if "phone" in d["type"] or "tablet" in d["type"])
        others = count - phones

        lines = [f"I found {count} device{'s' if count != 1 else ''} on the network."]
        for d in devices:
            if d["me"]:
                lines.append(f"{d['ip']} is this machine.")
            else:
                lines.append(f"{d['ip']}: {d['type']}{', hostname ' + d['hostname'] if d['hostname'] else ''}.")

        summary = " ".join(lines)
        if len(summary) > 600:
            summary = (f"I found {count} devices on your network. "
                       f"{phones} appear to be phones or tablets. "
                       f"The rest include {others} other devices.")

        return summary

    except Exception as e:
        console.print(f"[red]WiFi scan error: {e}[/red]")
        return "Sorry, the network scan failed."
    finally:
        set_state("idle")

# ==================== SLEEP WATCHER (background thread) ====================

def sleep_watcher():
    """
    Monitors idle time and puts DREAM to sleep after SLEEP_IDLE_TIMEOUT seconds.
    Separate from flirt_watcher — flirt fires at 10 min, sleep at 15 min.
    """
    while _state["running"]:
        time.sleep(5)
        if not _state["running"]:
            break
        if _state["sleeping"]:
            continue
        if _state["value"] != "idle":
            continue
        elapsed = time.time() - _state["last_wake_ts"]
        if elapsed >= SLEEP_IDLE_TIMEOUT:
            console.print(f"[blue]Idle for {elapsed:.0f}s — entering sleep mode[/blue]")
            enter_sleep()

# ==================== FLIRT WATCHER (background thread) ====================

def flirt_watcher():
    while _state["running"]:
        time.sleep(5)
        if not _state["running"]:
            break
        if _state["flirt_played"]:
            continue
        if _state["sleeping"]:
            continue
        if _state["value"] != "idle":
            touch_interaction()
            continue
        elapsed = time.time() - _state["last_wake_ts"]
        if elapsed >= FLIRT_IDLE_TIMEOUT:
            pool = VIDEO_POOLS.get("flirtytalk", [])
            if pool:
                chosen = random.choice(pool)
                console.print(f"[magenta]Flirt timeout — queueing {os.path.basename(chosen)}[/magenta]")
                _state["force_video"] = chosen
                _state["flirt_played"] = True
            else:
                console.print("[yellow]Flirt timeout but no flirtytalk videos found.[/yellow]")
                _state["flirt_played"] = True

# ==================== VIDEO PLAYER ====================

class VideoPlayer:
    def __init__(self, path, sw, sh, loop=True):
        self.sw, self.sh = sw, sh
        self.loop     = loop
        self.finished = False
        self.cap      = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        self.fps      = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.ms_frame = 1000.0 / self.fps
        self._surface = pygame.Surface((sw, sh))
        self._last_ms = 0.0
        self._read_next()

    def _read_next(self):
        ok, frame = self.cap.read()
        if not ok:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
            else:
                self.finished = True
                return
        if ok:
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ih, iw = rgb.shape[:2]
            scale  = max(self.sw / iw, self.sh / ih)
            nw, nh = int(iw * scale), int(ih * scale)
            rgb    = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
            xo, yo = (nw - self.sw) // 2, (nh - self.sh) // 2
            rgb    = rgb[yo:yo + self.sh, xo:xo + self.sw]
            self._surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))

    def get_frame(self, now_ms):
        if not self.finished and now_ms - self._last_ms >= self.ms_frame:
            self._read_next()
            self._last_ms = now_ms
        return self._surface

    def release(self):
        self.cap.release()

# ==================== VIDEO STATE MANAGER ====================

class VideoStateManager:
    def __init__(self, sw, sh):
        self.sw = sw
        self.sh = sh
        self._player: VideoPlayer | None       = None
        self._current_path   = None
        self._current_state  = None
        self._prev_idle_path = None
        self._force_player: VideoPlayer | None = None

    def _pick_random(self, pool, avoid=None):
        if not pool:
            return None
        if len(pool) == 1:
            return pool[0]
        choices = [p for p in pool if p != avoid]
        return random.choice(choices) if choices else random.choice(pool)

    def _load(self, path, loop=True):
        if self._player:
            self._player.release()
        try:
            self._player       = VideoPlayer(path, self.sw, self.sh, loop=loop)
            self._current_path = path
        except Exception as e:
            console.print(f"[red]VideoPlayer load error: {e}[/red]")
            self._player       = None
            self._current_path = None

    def get_frame(self, now_ms, state):
        # Force video (lipsync output or flirt clip) takes priority
        force_path = _state.get("force_video")
        if force_path and self._force_player is None:
            try:
                self._force_player = VideoPlayer(force_path, self.sw, self.sh, loop=False)
                console.print(f"[magenta]Playing force video: {os.path.basename(force_path)}[/magenta]")
            except Exception as e:
                console.print(f"[red]Force video error: {e}[/red]")
                _state["force_video"] = None

        if self._force_player is not None:
            frame = self._force_player.get_frame(now_ms)
            if self._force_player.finished:
                self._force_player.release()
                self._force_player = None
                _state["force_video"] = None
                console.print("[magenta]Force video done.[/magenta]")
            return frame

        # Sleeping state → loop sleeping.mp4
        if state == "sleeping":
            pool = VIDEO_POOLS.get("sleeping", [])
            if pool and self._current_state != "sleeping":
                self._current_state = "sleeping"
                self._load(pool[0], loop=True)
            if self._player:
                return self._player.get_frame(now_ms)

        pool = VIDEO_POOLS.get(state, [])
        if not pool:
            pool = VIDEO_POOLS.get("idle", [])

        if state != self._current_state:
            avoid = self._prev_idle_path if state == "idle" else None
            path  = self._pick_random(pool, avoid=avoid)
            if state == "idle":
                self._prev_idle_path = path
            self._current_state = state
            if path:
                self._load(path, loop=True)

        if state == "idle" and self._player and self._player.finished:
            path = self._pick_random(pool, avoid=self._current_path)
            self._prev_idle_path = path
            if path:
                self._load(path, loop=True)

        if self._player is None:
            blank = pygame.Surface((self.sw, self.sh))
            blank.fill((10, 12, 20))
            return blank

        return self._player.get_frame(now_ms)

    def release(self):
        if self._player:
            self._player.release()
        if self._force_player:
            self._force_player.release()

# ==================== MAIN VOICE LOOP ====================

def voice_loop():
    # Load any pre-cached lipsync videos (Yes?, startup, etc.)
    _init_lipsync_cache()
    # Kick off MuseTalk model loading immediately so models are warm
    # before the user's first real request.
    threading.Thread(target=_load_musetalk, daemon=True).start()

    try:
        r      = requests.get("http://localhost:11434/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        console.print(f"[green]OK[/green] Ollama: {', '.join(models) or 'no models'}")
        if not any(MODEL in m for m in models):
            console.print(f"[yellow]'{MODEL}' not pulled.[/yellow]")
    except Exception:
        console.print("[red]Ollama not running — voice loop disabled. Display will still open.[/red]")
        return

    get_whisper()

    # Startup announcement.
    # Cached path: instant lipsync from pre-generated files.
    # First-run path: wait for MuseTalk models to be ready, then speak() will
    # run inference and auto-save startup_intro.mp4 + startup.wav for next run.
    if not (os.path.exists(STARTUP_VID) and os.path.exists(STARTUP_WAV)):
        console.print("[dim]First run — waiting for MuseTalk to generate startup cache...[/dim]")
        deadline = time.time() + 120
        while not _musetalk_loaded and time.time() < deadline:
            time.sleep(0.5)

    if os.path.exists(STARTUP_VID) and os.path.exists(STARTUP_WAV):
        console.print("[green]Startup lipsync cache hit[/green]")
        _lipsync_cache[STARTUP_TEXT] = (STARTUP_WAV, STARTUP_VID)
        set_state("talking")
        _state["force_video"] = STARTUP_VID
        _play_wav(STARTUP_WAV)
        set_state("idle")
    else:
        speak(STARTUP_TEXT)  # Generates + caches startup_intro.mp4 for next run

    history = []

    while _state["running"]:
        try:
            trigger = listen_for_wake_word()

            if trigger is None or not _state["running"]:
                break

            if trigger == "__WIFI_SCAN__":
                summary = run_wifi_scan()
                speak(summary)
                continue

            speak("Yes?")
            set_state("listening")
            console.print("[bold yellow]Listening for command...[/bold yellow]")

            if not _record_clip(AUDIO_FILE, RECORD_SECONDS):
                speak("I didn't catch that.")
                continue

            if not check_audio_levels():
                speak("I couldn't hear you clearly.")
                continue

            user_text = transcribe_file(AUDIO_FILE)
            if not user_text:
                speak("I didn't catch that.")
                continue

            console.print(f"\n[bold green]You:[/bold green] {user_text}")
            set_state("idle")
            touch_interaction()

            lower = user_text.lower()

            if any(w in lower for w in ["goodbye", "exit", "quit", "bye", "shut down", "shutdown"]):
                speak("Goodbye.")
                _state["running"] = False
                break

            if any(t in lower for t in WIFI_TRIGGERS):
                summary = run_wifi_scan()
                speak(summary)
                continue

            if any(t in lower for t in STATS_TRIGGERS):
                speak(build_stats_summary())
                continue

            response = ask_llm(user_text, history)
            history.append({"role": "user",      "content": user_text})
            history.append({"role": "assistant",  "content": response})
            if len(history) > 12:
                history = history[-12:]
            speak(response)

        except Exception as e:
            console.print(f"[red]Error: {escape(str(e))}[/red]")
            set_state("idle")
            continue

# ==================== DISPLAY (main thread) ====================

def run_display():
    console.print("[dim]>> run_display: enter[/dim]")

    info   = pygame.display.Info()
    SW, SH = info.current_w, info.current_h
    console.print(f"[dim]>> Display info: {SW}x{SH}[/dim]")

    # Borderless windowed fullscreen — avoids SDL's FULLSCREEN mode which
    # crashes at the C level on some Windows/driver configurations.
    os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"
    screen = pygame.display.set_mode((SW, SH), pygame.NOFRAME)
    console.print("[dim]>> Borderless window created[/dim]")

    pygame.display.set_caption("DREAM")
    pygame.mouse.set_visible(False)

    if not CV2_AVAILABLE:
        console.print("[red]opencv-python required for video playback — exiting.[/red]")
        _state["running"] = False
        return

    vsm   = VideoStateManager(SW, SH)
    clock = pygame.time.Clock()

    console.print(f"[dim]>> Entering display loop (running={_state['running']})[/dim]")

    # Drain any stale events (e.g. SDL_QUIT that Windows fires when a new
    # fullscreen window is created and immediately loses focus)
    pygame.event.clear()

    while _state["running"]:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                console.print("[dim]>> pygame.QUIT event — exiting[/dim]")
                _state["running"] = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    console.print(f"[dim]>> Key exit: {event.key}[/dim]")
                    _state["running"] = False

        state  = _state["value"]
        now_ms = pygame.time.get_ticks()

        frame = vsm.get_frame(now_ms, state)
        screen.blit(frame, (0, 0))
        pygame.display.flip()
        clock.tick(30)

    console.print(f"[dim]>> Display loop exited (running={_state['running']})[/dim]")
    vsm.release()
    pygame.quit()

# ==================== MAIN ====================

def main():
    startup_banner()

    piper_sr = get_piper_sample_rate()
    console.print(f"[dim]Mixer init: {piper_sr} Hz, 16-bit mono, buffer=4096[/dim]")
    pygame.mixer.pre_init(frequency=piper_sr, size=-16, channels=1, buffer=4096)

    init_ok, init_fail = pygame.init()
    console.print(f"[dim]pygame.init(): {init_ok} OK, {init_fail} failed[/dim]")

    try:
        pygame.mixer.init()
    except Exception as e:
        console.print(f"[yellow]Mixer init warning: {e} — continuing without audio[/yellow]")

    mixer_info = pygame.mixer.get_init()
    if mixer_info:
        console.print(f"[green]OK[/green] Mixer: {mixer_info[0]} Hz / {mixer_info[1]}-bit / {mixer_info[2]} ch")
    else:
        console.print("[yellow]Mixer not available — audio will be silent[/yellow]")

    # Background watchers
    ft = threading.Thread(target=flirt_watcher, daemon=True)
    ft.start()

    st = threading.Thread(target=sleep_watcher, daemon=True)
    st.start()

    vt = threading.Thread(target=voice_loop, daemon=True)
    vt.start()

    try:
        run_display()
    except Exception:
        import traceback as _tb
        console.print("[red]Display crashed — see below[/red]")
        _tb.print_exc()

    vt.join(timeout=2)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback as _tb
        _tb.print_exc()
        input("Press Enter to exit...")
