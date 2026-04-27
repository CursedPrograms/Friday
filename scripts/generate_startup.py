#!/usr/bin/env python3
"""
One-time script: generates the permanent startup intro lipsync video.
Run once from the project root:
    venv311\Scripts\python scripts\generate_startup.py

Outputs:
    audio/startup.wav          — Piper TTS of the startup phrase
    videos/startup_intro.mp4   — MuseTalk lipsync video (loops cleanly)
"""

import os, sys, copy, glob, pickle, re, subprocess

os.environ["TORCH_COMPILE_DISABLE"]       = "1"
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))

STARTUP_TEXT   = "ComCentre online. DREAM is ready. Say Hey DREAM to wake me."
AUDIO_DIR      = os.path.join(BASE_DIR, "audio")
VIDEOS_DIR     = os.path.join(BASE_DIR, "videos")
MODELS_DIR     = os.path.join(BASE_DIR, "models")
MUSETALK_VIDEO = os.path.join(VIDEOS_DIR, "musetalk_talk.mp4")
OUTPUT_WAV     = os.path.join(AUDIO_DIR,  "startup.wav")
OUTPUT_VID     = os.path.join(VIDEOS_DIR, "startup_intro.mp4")
TEMP_DIR       = os.path.join(BASE_DIR,   "musetalk_out", "startup_gen")

VENV_DIR   = os.path.join(BASE_DIR, "venv311")
PIPER_BIN  = os.path.join(VENV_DIR, "Scripts", "piper.exe")
VOICES_DIR = os.path.join(BASE_DIR, "voices")

os.makedirs(AUDIO_DIR,  exist_ok=True)
os.makedirs(TEMP_DIR,   exist_ok=True)

# ── Find voice model ──────────────────────────────────────────────────────────
VOICE_MODEL = next(
    (os.path.join(VOICES_DIR, f) for f in sorted(os.listdir(VOICES_DIR)) if f.endswith(".onnx")),
    None,
)
if VOICE_MODEL is None:
    sys.exit("ERROR: No .onnx voice model found in voices/")

print(f"Voice  : {os.path.basename(VOICE_MODEL)}")
print(f"Text   : {STARTUP_TEXT}")
print(f"Ref vid: {MUSETALK_VIDEO}")

# ── Step 1: Piper TTS ─────────────────────────────────────────────────────────
print("\n[1/3] Piper TTS...")
proc = subprocess.run(
    [PIPER_BIN, "-m", VOICE_MODEL, "-f", OUTPUT_WAV],
    input=STARTUP_TEXT.encode("utf-8"),
    capture_output=True, timeout=30,
)
if proc.returncode != 0:
    sys.exit(f"Piper failed: {proc.stderr.decode()}")
print(f"       -> {OUTPUT_WAV}  ({os.path.getsize(OUTPUT_WAV):,} bytes)")

# ── Step 2: Load MuseTalk models ──────────────────────────────────────────────
print("\n[2/3] Loading MuseTalk models...")
import torch, imageio, numpy as np, cv2
from transformers import WhisperModel
from musetalk.utils.audio_processor  import AudioProcessor
from musetalk.utils.blending         import get_image
from musetalk.utils.face_parsing     import FaceParsing
from musetalk.utils.utils            import get_video_fps, datagen
from musetalk.utils.preprocessing    import get_landmark_and_bbox, read_imgs, coord_placeholder
from musetalk.models.vae             import VAE
from musetalk.models.unet            import UNet, PositionalEncoding
from moviepy.editor                  import VideoFileClip, AudioFileClip

device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
print(f"       Device: {device}")

vae  = VAE(model_path=os.path.join(MODELS_DIR, "sd-vae"))
unet = UNet(
    unet_config=os.path.join(MODELS_DIR, "musetalkV15", "musetalk.json"),
    model_path =os.path.join(MODELS_DIR, "musetalkV15", "unet.pth"),
    device=device,
)
pe = PositionalEncoding(d_model=384)
pe.to(device).to(weight_dtype)
vae.vae    = vae.vae.to(device).to(weight_dtype)
unet.model = unet.model.to(device).to(weight_dtype)

audio_processor = AudioProcessor(feature_extractor_path=os.path.join(MODELS_DIR, "whisper"))
whisper = WhisperModel.from_pretrained(os.path.join(MODELS_DIR, "whisper"))
whisper = whisper.to(device=device, dtype=weight_dtype).eval()
whisper.requires_grad_(False)
timesteps = torch.tensor([0], device=device)
print("       Models loaded.")

# ── Step 3: Inference ─────────────────────────────────────────────────────────
print("\n[3/3] Running MuseTalk inference...")

MUSETALK_OUT_DIR = os.path.join(BASE_DIR, "musetalk_out")
frames_cache_dir = os.path.join(MUSETALK_OUT_DIR, "musetalk_talk_frames")
coord_cache_path = os.path.join(MUSETALK_OUT_DIR, "musetalk_talk.pkl")
result_frames_dir = os.path.join(TEMP_DIR, "frames")
os.makedirs(result_frames_dir, exist_ok=True)

# Extract reference video frames (reuses dream.py cache if already present)
if not os.path.isdir(frames_cache_dir) or not os.listdir(frames_cache_dir):
    print("       Extracting reference frames...")
    os.makedirs(frames_cache_dir, exist_ok=True)
    reader = imageio.get_reader(MUSETALK_VIDEO)
    for i, im in enumerate(reader):
        imageio.imwrite(f"{frames_cache_dir}/{i:08d}.png", im)
input_img_list = sorted(glob.glob(os.path.join(frames_cache_dir, "*.[jpJP][pnPN]*[gG]")))
fps = get_video_fps(MUSETALK_VIDEO)
print(f"       {len(input_img_list)} frames @ {fps:.1f} fps")

# Audio features
whisper_feats, librosa_length = audio_processor.get_audio_feature(OUTPUT_WAV)
whisper_chunks = audio_processor.get_whisper_chunk(
    whisper_feats, device, weight_dtype, whisper, librosa_length,
    fps=fps, audio_padding_length_left=2, audio_padding_length_right=2,
)

# Face landmarks (reuses cache if dream.py already ran)
if os.path.exists(coord_cache_path):
    print("       Using cached landmarks.")
    with open(coord_cache_path, "rb") as f:
        coord_list = pickle.load(f)
    frame_list = read_imgs(input_img_list)
else:
    print("       Detecting face landmarks (one-time, takes ~30s)...")
    coord_list, frame_list = get_landmark_and_bbox(input_img_list, 0)
    with open(coord_cache_path, "wb") as f:
        pickle.dump(coord_list, f)

# Encode to latents
extra_margin = 10
fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)
input_latent_list = []
for bbox, frame in zip(coord_list, frame_list):
    if bbox == coord_placeholder:
        continue
    x1, y1, x2, y2 = bbox
    y2   = min(y2 + extra_margin, frame.shape[0])
    crop = cv2.resize(frame[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LANCZOS4)
    input_latent_list.append(vae.get_latents_for_unet(crop))

frame_list_cycle        = frame_list        + frame_list[::-1]
coord_list_cycle        = coord_list        + coord_list[::-1]
input_latent_list_cycle = input_latent_list + input_latent_list[::-1]

# UNet inference
print("       UNet inference...")
gen = datagen(whisper_chunks=whisper_chunks, vae_encode_latents=input_latent_list_cycle,
              batch_size=8, delay_frame=0, device=device)
res_frame_list = []
for wb, lb in gen:
    af = pe(wb)
    lb = lb.to(dtype=weight_dtype)
    pred = unet.model(lb, timesteps, encoder_hidden_states=af).sample
    for rf in vae.decode_latents(pred):
        res_frame_list.append(rf)

# Composite
print("       Compositing frames...")
for i, res_frame in enumerate(res_frame_list):
    bbox      = coord_list_cycle[i % len(coord_list_cycle)]
    ori_frame = copy.deepcopy(frame_list_cycle[i % len(frame_list_cycle)])
    if bbox == coord_placeholder:
        continue
    x1, y1, x2, y2 = bbox
    y2 = min(y2 + extra_margin, ori_frame.shape[0])
    try:
        res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
    except Exception:
        continue
    combine = get_image(ori_frame, res_frame, [x1, y1, x2, y2], mode="jaw", fp=fp)
    cv2.imwrite(f"{result_frames_dir}/{str(i).zfill(8)}.png", combine)

# Assemble video + bake audio
print("       Assembling video...")
temp_vid = os.path.join(TEMP_DIR, "temp.mp4")
valid    = re.compile(r"\d{8}\.png")
files    = sorted(
    [f for f in os.listdir(result_frames_dir) if valid.match(f)],
    key=lambda x: int(x.split(".")[0])
)
images = [imageio.imread(os.path.join(result_frames_dir, f)) for f in files]
imageio.mimwrite(temp_vid, images, "FFMPEG", fps=25, codec="libx264", pixelformat="yuv420p")

vc = VideoFileClip(temp_vid)
ac = AudioFileClip(OUTPUT_WAV)
vc.set_audio(ac).write_videofile(OUTPUT_VID, codec="libx264", audio_codec="aac", fps=25, logger=None)
vc.close(); ac.close()
os.remove(temp_vid)

print(f"\n{'='*50}")
print(f"  startup.wav          → {OUTPUT_WAV}")
print(f"  startup_intro.mp4    → {OUTPUT_VID}")
print(f"{'='*50}")
print("DREAM will use these files automatically on next startup.")
