import queue
import sounddevice as sd
import numpy as np
import sys
import os
import subprocess
import re
import torch
import whisper
import tempfile
import scipy.io.wavfile as wavfile
import scipy.signal
from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# ─────────── Configuration ───────────
mic_device_index = 8                     # This one works on your system
mic_sample_rate = 48000                 # Actual mic capture rate
whisper_sample_rate = 16000            # Whisper needs this
robot_script_path = os.path.expanduser(
    "~/tiago_public_ws/src/tiago_manual_goal/tiago_manual_goal/robot_controller.py"
)
llama_path = Path(
    "~/tiago_public_ws/src/tiago_manual_goal/tiago_manual_goal/models/merged-tinyllama"
).expanduser().resolve().as_posix()

print(f"🦙 Loading TinyLLaMA model from: {llama_path}")

# ─────────── Load Whisper (CPU) ───────────
print("🔊 Loading Whisper model: large-v3 (local, CPU)...")
model = whisper.load_model("large-v3", device="cpu")

# ─────────── Load TinyLLaMA (offline, local) ───────────
print("🦙 Loading TinyLLaMA model locally...")
tokenizer = AutoTokenizer.from_pretrained(llama_path, local_files_only=True)
llm_model = AutoModelForCausalLM.from_pretrained(
    llama_path,
    local_files_only=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True
)
llm_pipeline = pipeline("text-generation", model=llm_model, tokenizer=tokenizer)

# ─────────── Audio Stream Setup ───────────
q = queue.Queue()
def callback(indata, frames, time, status):
    if status:
        print("⚠️", status, file=sys.stderr)
    q.put(indata.copy())

print(f"🎙️ Listening from mic index {mic_device_index} at {mic_sample_rate}Hz — Speak now...")

# ─────────── Processing Function ───────────
def process_with_llm(text):
    print(f"\n🧠 Whisper Transcript: {text}")
    print("🤖 Thinking...")
    prompt = f"You heard: '{text}'. Extract the item name in format: item: <name>"
    response = llm_pipeline(prompt, max_length=60, do_sample=False)[0]["generated_text"]
    print("🧠 LLM Response:\n", response.strip())

    match = re.search(r'item\s*:\s*(\w+)', response, re.IGNORECASE)
    if match:
        item = match.group(1).lower()
        print(f"🎯 Extracted item: {item}")
        print(f"🚀 Triggering robot_controller.py with item: {item}")
        subprocess.Popen(["python3", robot_script_path, item])
    else:
        print("❌ No valid 'item:' found in LLM response.")

# ─────────── Main Loop ───────────
try:
    with sd.InputStream(samplerate=mic_sample_rate, channels=1, dtype='float32',
                        callback=callback, device=mic_device_index):
        buffer = []
        while True:
            data = q.get()
            buffer.extend(data[:, 0])

            if len(buffer) >= mic_sample_rate * 5:  # 5 seconds of audio
                raw_audio = np.array(buffer[:mic_sample_rate * 5])
                buffer = buffer[mic_sample_rate * 5:]

                # Normalize
                raw_audio = raw_audio / np.max(np.abs(raw_audio))

                # Resample from mic_sample_rate to 16000Hz
                audio_chunk = scipy.signal.resample(raw_audio, whisper_sample_rate * 5)

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    wavfile.write(tmp.name, whisper_sample_rate, audio_chunk.astype(np.float32))
                    audio_path = tmp.name

                result = model.transcribe(audio_path, fp16=False)
                text = result.get("text", "").strip()
                os.remove(audio_path)

                if text:
                    process_with_llm(text)

except KeyboardInterrupt:
    print("\n🛑 Stopped by user")
except Exception as e:
    print(f"\n❌ ERROR: {e}")

