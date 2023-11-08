import pyaudio
import wave
from transformers import pipeline
# import torch

audio = pyaudio.PyAudio()
input_device = 0

# Get All Audio Adapters / Recording Devices
for i in range(audio.get_device_count()):
    device_info = audio.get_device_info_by_index(i)
    device_name = device_info['name']
    device_index = device_info['index']
    # max_input_channels = device_info['maxInputChannels']
    # max_output_channels = device_info['maxOutputChannels']
    
    print(f"Device {i}: {device_name}")
    print(f"  Index: {device_index}")
    # print(f"  Max Input Channels: {max_input_channels}")
    # print(f"  Max Output Channels: {max_output_channels}")

# audio.terminate()

stream = audio.open(
    format=pyaudio.paInt16,
    channels=1, #Mono
    rate=44100, # Sample Rate
    input=True,
    input_device_index=input_device,
    frames_per_buffer=1024,
    
)

frames = []

print("recording...")

for _ in range(0, int(44100 / 1024 * 5)):
    data = stream.read(1024)
    frames.append(data)

print("recording finished")

stream.start_stream()
stream.close()
audio.terminate()

with wave.open("recorded1.wav", "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    wf.setframerate(44100)
    wf.writeframes(b''.join(frames))

print("Created Audio File")

print("Transfering Audio to Words")
channel = pipeline("automatic-speech-recognition", model="openai/whisper-medium")
result = channel("recorded1.wav")

print(result)