// Ambient Sound Therapy System
// Synchronized with posture and stress levels

class SoundTherapyManager {
    constructor() {
        this.isEnabled = false;
        this.volume = 0.3;
        this.currentTrack = null;
        this.audioContext = null;
        this.sounds = {};
        this.fadeTimer = null;
        this.currentPostureQuality = 'unknown';
        this.stressLevel = 'low';
        
        // Sound therapy tracks for different states
        this.soundConfigs = {
            'excellent': {
                name: 'Forest Rain',
                frequency: 432, // Hz - healing frequency
                type: 'nature',
                volume: 0.2,
                color: '#28a745'
            },
            'good': {
                name: 'Ocean Waves',
                frequency: 396, // Hz - stress relief
                type: 'water',
                volume: 0.25,
                color: '#17a2b8'
            },
            'fair': {
                name: 'Wind Chimes',
                frequency: 528, // Hz - focus and clarity
                type: 'ambient',
                volume: 0.3,
                color: '#ffc107'
            },
            'poor': {
                name: 'Deep Breathing',
                frequency: 174, // Hz - pain relief and relaxation
                type: 'meditation',
                volume: 0.4,
                color: '#dc3545'
            },
            'high_stress': {
                name: 'Tibetan Bowls',
                frequency: 639, // Hz - stress and anxiety relief
                type: 'healing',
                volume: 0.35,
                color: '#6f42c1'
            }
        };
        
        this.init();
    }
    
    async init() {
        try {
            // Initialize Web Audio API
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            await this.generateSounds();
            console.log('Sound therapy system initialized');
        } catch (error) {
            console.warn('Audio context not supported:', error);
        }
    }
    
    async generateSounds() {
        // Generate binaural beats and ambient sounds
        for (const [quality, config] of Object.entries(this.soundConfigs)) {
            this.sounds[quality] = await this.createAmbientSound(config);
        }
    }
    
    async createAmbientSound(config) {
        if (!this.audioContext) return null;
        
        // Create oscillators for binaural beats and ambient tones
        const baseFreq = config.frequency;
        const binauralBeat = 10; // 10 Hz alpha waves for focus
        
        // Left channel
        const leftOsc = this.audioContext.createOscillator();
        leftOsc.frequency.setValueAtTime(baseFreq, this.audioContext.currentTime);
        leftOsc.type = 'sine';
        
        // Right channel (slightly different for binaural effect)
        const rightOsc = this.audioContext.createOscillator();
        rightOsc.frequency.setValueAtTime(baseFreq + binauralBeat, this.audioContext.currentTime);
        rightOsc.type = 'sine';
        
        // Create gain nodes for volume control
        const leftGain = this.audioContext.createGain();
        const rightGain = this.audioContext.createGain();
        const masterGain = this.audioContext.createGain();
        
        // Create stereo panner
        const merger = this.audioContext.createChannelMerger(2);
        
        // Add subtle noise for natural ambient sound
        const noiseBuffer = this.createNoiseBuffer(config.type);
        const noiseSource = this.audioContext.createBufferSource();
        noiseSource.buffer = noiseBuffer;
        noiseSource.loop = true;
        
        const noiseGain = this.audioContext.createGain();
        noiseGain.gain.setValueAtTime(0.1, this.audioContext.currentTime);
        
        // Connect nodes
        leftOsc.connect(leftGain);
        rightOsc.connect(rightGain);
        leftGain.connect(merger, 0, 0);
        rightGain.connect(merger, 0, 1);
        
        noiseSource.connect(noiseGain);
        noiseGain.connect(merger);
        
        merger.connect(masterGain);
        masterGain.connect(this.audioContext.destination);
        
        // Set initial volumes
        leftGain.gain.setValueAtTime(config.volume * 0.3, this.audioContext.currentTime);
        rightGain.gain.setValueAtTime(config.volume * 0.3, this.audioContext.currentTime);
        masterGain.gain.setValueAtTime(0, this.audioContext.currentTime);
        
        return {
            leftOsc,
            rightOsc,
            noiseSource,
            masterGain,
            config,
            isPlaying: false
        };
    }
    
    createNoiseBuffer(type) {
        const bufferSize = this.audioContext.sampleRate * 2; // 2 seconds
        const buffer = this.audioContext.createBuffer(1, bufferSize, this.audioContext.sampleRate);
        const output = buffer.getChannelData(0);
        
        for (let i = 0; i < bufferSize; i++) {
            switch (type) {
                case 'nature':
                    // Brown noise with filtering for forest sounds
                    output[i] = (Math.random() * 2 - 1) * Math.pow(1 / (i % 100 + 1), 0.5);
                    break;
                case 'water':
                    // Pink noise for ocean-like sounds
                    output[i] = (Math.random() * 2 - 1) * Math.pow(1 / (i % 50 + 1), 0.3);
                    break;
                case 'ambient':
                    // White noise filtered for wind-like sounds
                    output[i] = (Math.random() * 2 - 1) * 0.1;
                    break;
                default:
                    // Soft pink noise
                    output[i] = (Math.random() * 2 - 1) * 0.05;
            }
        }
        
        return buffer;
    }
    
    updatePostureState(postureData) {
        const quality = postureData.posture_quality || 'unknown';
        const angle = postureData.angle || 0;
        
        // Calculate stress level based on posture
        let newStressLevel = 'low';
        if (angle > 20 || quality === 'poor') {
            newStressLevel = 'high';
        } else if (angle > 15 || quality === 'fair') {
            newStressLevel = 'medium';
        }
        
        // Update sound if state changed
        if (quality !== this.currentPostureQuality || newStressLevel !== this.stressLevel) {
            this.currentPostureQuality = quality;
            this.stressLevel = newStressLevel;
            
            if (this.isEnabled) {
                this.transitionToSound(this.selectSoundForState());
            }
            
            // Update UI
            this.updateSoundTherapyUI();
        }
    }
    
    selectSoundForState() {
        // High stress overrides posture quality
        if (this.stressLevel === 'high') {
            return 'high_stress';
        }
        
        // Use posture quality for sound selection
        return this.currentPostureQuality in this.soundConfigs ? 
               this.currentPostureQuality : 'fair';
    }
    
    async transitionToSound(soundKey) {
        const newSound = this.sounds[soundKey];
        if (!newSound || !this.audioContext) return;
        
        // Fade out current sound
        if (this.currentTrack && this.currentTrack.isPlaying) {
            await this.fadeOut(this.currentTrack);
            this.stopSound(this.currentTrack);
        }
        
        // Start new sound
        this.currentTrack = newSound;
        await this.playSound(newSound);
        await this.fadeIn(newSound);
        
        console.log(`Sound therapy: ${newSound.config.name} (${soundKey})`);
    }
    
    async playSound(sound) {
        if (!sound || sound.isPlaying) return;
        
        try {
            // Resume audio context if suspended
            if (this.audioContext.state === 'suspended') {
                await this.audioContext.resume();
            }
            
            sound.leftOsc.start();
            sound.rightOsc.start();
            sound.noiseSource.start();
            sound.isPlaying = true;
        } catch (error) {
            console.warn('Error starting sound:', error);
        }
    }
    
    stopSound(sound) {
        if (!sound || !sound.isPlaying) return;
        
        try {
            sound.leftOsc.stop();
            sound.rightOsc.stop();
            sound.noiseSource.stop();
            sound.isPlaying = false;
        } catch (error) {
            console.warn('Error stopping sound:', error);
        }
    }
    
    async fadeIn(sound, duration = 2000) {
        if (!sound || !sound.masterGain) return;
        
        const targetVolume = this.volume * sound.config.volume;
        sound.masterGain.gain.setValueAtTime(0, this.audioContext.currentTime);
        sound.masterGain.gain.linearRampToValueAtTime(
            targetVolume, 
            this.audioContext.currentTime + (duration / 1000)
        );
    }
    
    async fadeOut(sound, duration = 1000) {
        if (!sound || !sound.masterGain) return;
        
        sound.masterGain.gain.setValueAtTime(
            sound.masterGain.gain.value, 
            this.audioContext.currentTime
        );
        sound.masterGain.gain.linearRampToValueAtTime(
            0, 
            this.audioContext.currentTime + (duration / 1000)
        );
    }
    
    enable() {
        this.isEnabled = true;
        const soundKey = this.selectSoundForState();
        this.transitionToSound(soundKey);
        this.updateSoundTherapyUI();
    }
    
    disable() {
        this.isEnabled = false;
        if (this.currentTrack) {
            this.fadeOut(this.currentTrack, 1000);
            setTimeout(() => {
                if (this.currentTrack) {
                    this.stopSound(this.currentTrack);
                    this.currentTrack = null;
                }
            }, 1000);
        }
        this.updateSoundTherapyUI();
    }
    
    setVolume(volume) {
        this.volume = Math.max(0, Math.min(1, volume));
        if (this.currentTrack && this.currentTrack.masterGain) {
            const targetVolume = this.volume * this.currentTrack.config.volume;
            this.currentTrack.masterGain.gain.setValueAtTime(
                targetVolume, 
                this.audioContext.currentTime
            );
        }
    }
    
    updateSoundTherapyUI() {
        const statusElement = document.getElementById('sound-therapy-status');
        const currentSoundElement = document.getElementById('current-sound');
        const soundColorElement = document.getElementById('sound-color-indicator');
        
        if (statusElement) {
            statusElement.textContent = this.isEnabled ? 'Active' : 'Inactive';
            statusElement.className = `badge ${this.isEnabled ? 'bg-success' : 'bg-secondary'}`;
        }
        
        if (this.isEnabled && this.currentTrack) {
            const config = this.currentTrack.config;
            
            if (currentSoundElement) {
                currentSoundElement.textContent = config.name;
            }
            
            if (soundColorElement) {
                soundColorElement.style.backgroundColor = config.color;
                soundColorElement.style.display = 'inline-block';
            }
        } else {
            if (currentSoundElement) {
                currentSoundElement.textContent = 'None';
            }
            
            if (soundColorElement) {
                soundColorElement.style.display = 'none';
            }
        }
    }
}

// Global sound therapy manager
let soundTherapy = null;

// Initialize sound therapy when page loads
document.addEventListener('DOMContentLoaded', () => {
    soundTherapy = new SoundTherapyManager();
});

// Export for use in other scripts
window.soundTherapy = soundTherapy;