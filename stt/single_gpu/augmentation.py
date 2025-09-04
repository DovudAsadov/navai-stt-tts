import random
import numpy as np
import torch
import torchaudio
import librosa


class AudioAugmentation:
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
    
    def add_noise(self, audio, noise_factor=0.005):
        """Add random noise to audio"""
        noise = np.random.randn(len(audio))
        augmented_audio = audio + noise_factor * noise
        return augmented_audio.astype(np.float32)
    
    def time_shift(self, audio, shift_max=0.2):
        """Randomly shift audio in time"""
        shift = np.random.randint(int(-shift_max * len(audio)), int(shift_max * len(audio)))
        if shift > 0:
            augmented_audio = np.pad(audio, (shift, 0), mode='constant')[:-shift]
        elif shift < 0:
            augmented_audio = np.pad(audio, (0, -shift), mode='constant')[-shift:]
        else:
            augmented_audio = audio
        return augmented_audio.astype(np.float32)
    
    def change_speed(self, audio, speed_factor=None):
        """Change speed of audio"""
        if speed_factor is None:
            speed_factor = np.random.uniform(0.8, 1.2)
        
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
        
        augmented_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
            audio_tensor, self.sample_rate, [["speed", str(speed_factor)], ["rate", str(self.sample_rate)]]
        )
        
        return augmented_audio.squeeze(0).numpy().astype(np.float32)
    
    def change_pitch(self, audio, pitch_factor=None):
        """Change pitch of audio"""
        if pitch_factor is None:
            pitch_factor = np.random.uniform(-2, 2) 
        
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
        
        augmented_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
            audio_tensor, self.sample_rate, [["pitch", str(pitch_factor * 100)]]
        )
        
        return augmented_audio.squeeze(0).numpy().astype(np.float32)
