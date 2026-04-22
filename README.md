# LocalSotaTalk - TTS Backend Service

**LocalSotaTalk** is a comprehensive Text-to-Speech (TTS) backend service supporting multiple frameworks. It provides a unified REST API for voice synthesis with support for voice cloning and voice design capabilities. This project mainly focuses on low-cost SOTA models which supports voice cloning.

It has currently supported these frameworks:

- VoxCPM
- OmniVoice
- LongCat-AudioDiT

## ✨ Key Features

- **Multi-Framework Support**: Seamlessly works with OmniVoice (600+ languages) and LongCat-AudioDiT models
- **Voice ID System**: Simple speaker management using `voice_id` instead of file paths
- **Voice Design Support**: OmniVoice supports voice design via textual descriptions
- **RESTful API**: Simple API, which is compatible with [daswer123/xtts-api-server](https://github.com/daswer123/xtts-api-server)
- **Cross-Origin Support**: Built-in CORS middleware for web applications
- **Automatic Speaker Detection**: Scans `samples/` directory for audio and design files
- **Model Switching**: Hot-swap between different TTS models at runtime

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA support (recommended)
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/MoRanYue/LocalSotaTalk.git
cd LocalSotaTalk
```

2. **Initialize submodules**
```bash
git submodule update --init --recursive
```

3. **Create virtual environment and install dependencies**
```bash
# Create virtual environment
python -m venv .venv
# Or use UV
# uv venv -p 3.10

# Activate on Windows
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

4. **Prepare samples directory**
```bash
# Create samples directory
mkdir samples

# Add your speaker files (optional)
# Example: paimon.wav + paimon.txt for voice cloning
# Example: flba.design.txt for voice design
```

### Running the Service

```bash
# Start with OmniVoice (default)
python main.py --model k2-fsa/OmniVoice --host 0.0.0.0 --port 8000

# Start with LongCat-AudioDiT
python main.py --model meituan-longcat/LongCat-AudioDiT-1B --host 0.0.0.0 --port 8000

# With custom samples directory
python main.py --model k2-fsa/OmniVoice --samples-dir ./my_speakers --port 8000
```

Once running, access:
- **API Documentation**: http://localhost:8000/docs
- **OpenAPI Specification**: http://localhost:8000/openapi.json

## 📁 Speaker Management

LocalSotaTalk automatically scans the `samples/` directory for speaker files. Two types of speakers are supported:

### 1. Voice Cloning Speakers (Audio + Text)
For traditional voice cloning, add these files:
- `{speaker_id}.wav` - Reference audio file (supports .wav, .mp3, .flac)
- `{speaker_id}.txt` - Transcript of the reference audio (optional but recommended)

**Example**: `samples/paimon.wav` + `samples/paimon.txt` creates speaker with `voice_id="paimon"`

### 2. Voice Design Speakers (Text Description)
For voice design (OmniVoice only), add:
- `{speaker_id}.design.txt` - Text description of voice characteristics

**Example**: `samples/flba.design.txt` with content:
```
female, low pitch, british accent
```

The prompt above is an example for OmniVoice.

### Speaker Detection Rules
- Files are automatically associated by their base name
- `.wav` files take priority over `.design.txt` files
- System supports mixed configurations (some speakers with audio, others with design)

## 🔌 API Reference

### Core Endpoints

#### `GET /speakers`
Returns detailed information about all available speakers.

**Response**:
```json
[
  {
    "name": "paimon",
    "voice_id": "paimon",
    "type": "audio_with_text",
    "file_path": "samples/paimon.wav",
    "text_path": "samples/paimon.txt",
    "design_description": null
  },
  {
    "name": "flba",
    "voice_id": "flba",
    "type": "design_only",
    "file_path": "samples/flba.design.txt",
    "text_path": null,
    "design_description": "female, low pitch, british accent"
  }
]
```

#### `POST /tts_to_audio/`
Synthesize speech and return audio stream.

**Request**:
```json
{
  "text": "Hello, this is a test",
  "speaker_wav": "paimon",
  "language": "en"
}
```

**Parameters**:
- `text`: Text to synthesize (required)
- `speaker_wav`: Speaker voice_id or audio file path (required)
- `language`: Language code (optional, default: "en")

**Response**: Audio/WAV stream with headers:
- `Content-Type: audio/wav`
- `Duration`: Audio duration in seconds
- `Sample-Rate`: 24000 Hz

#### `POST /tts_to_file`
Synthesize speech and save to file.

**Request**:
```json
{
  "text": "Hello, this is a test",
  "speaker_wav": "paimon",
  "language": "en",
  "file_name_or_path": "output.wav"
}
```

**Response**:
```json
{
  "file_path": "output/output.wav",
  "duration": 2.32,
  "sample_rate": 24000
}
```

### Configuration Endpoints

- `GET /languages` - Get supported languages
- `GET /get_models_list` - Get available models
- `POST /switch_model` - Switch TTS model
- `POST /set_tts_settings` - Update TTS parameters
- `GET /get_folders` - Get current folder paths
- `POST /set_output` - Set output directory
- `POST /set_speaker_folder` - Set samples directory

## 🎯 Usage Examples

### Python Client Example
```python
import requests
import json

# TTS synthesis
def tts_synthesis(text, voice_id="paimon", language="en"):
    url = "http://localhost:8000/tts_to_audio/"
    data = {
        "text": text,
        "speaker_wav": voice_id,
        "language": language
    }
    
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        with open("output.wav", "wb") as f:
            f.write(response.content)
        print("Audio saved to output.wav")
    else:
        print(f"Error: {response.status_code}")
        print(response.json())

# Get available speakers
def get_speakers():
    response = requests.get("http://localhost:8000/speakers")
    speakers = response.json()
    for speaker in speakers:
        print(f"{speaker['voice_id']}: {speaker['type']}")
```

### curl Examples
```bash
# Get speakers list
curl -X GET http://localhost:8000/speakers

# Synthesize with voice_id
curl -X POST http://localhost:8000/tts_to_audio/ \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world","speaker_wav":"paimon","language":"en"}' \
  --output output.wav

# Synthesize with voice design
curl -X POST http://localhost:8000/tts_to_audio/ \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world","speaker_wav":"flba","language":"en"}' \
  --output design_output.wav
```

## 🏗️ Architecture

### Component Overview
```
LocalSotaTalk/
├── api/              # FastAPI endpoints and schemas
├── models/           # TTS adapter implementations
│   ├── base_adapter.py      # Abstract base class
│   ├── omnivoice_adapter.py # OmniVoice implementation
│   ├── longcat_adapter.py   # LongCat-AudioDiT implementation
│   ├── voxcpm_adapter.py    # VoxCPM implementation
│   └── manager.py           # Model manager
├── utils/            # Utility functions
├── systems/          # Framework submodules
│   ├── OmniVoice/    # OmniVoice framework
│   ├── VoxCPM/       # VoxCPM framework
│   └── LongCat-AudioDiT/ # LongCat framework
├── samples/          # Speaker samples directory
├── output/           # Generated audio files
└── config.py         # Configuration management
```

### Model Support Matrix

| Feature | VoxCPM2 | VoxCPM1.5 | OmniVoice | LongCat-AudioDiT |
|---------|-----------|------------------|-|-|
| Voice Cloning | ✅ | ✅ | ✅ | ✅ |
| Voice Design | ✅ | ❌ | ✅ | ❌ |
| Languages | 30 + 9 Chinese dialects | Chinese, English | 600+ | Chinese, English |
| Reference Audio | Optional | Required | Optional | Required |
| Reference Text | Recommended | Recommended | Optional | Recommended |
| Inference Speed | Medium | Fast | Fast | Medium |

## 🔧 Configuration

### Command Line Arguments
```bash
python main.py --help

--model MODEL          Model repository (default: k2-fsa/OmniVoice)
--samples-dir DIR      Samples directory (default: samples)
--output-dir DIR       Output directory (default: output)
--host HOST            Server host (default: 0.0.0.0)
--port PORT            Server port (default: 8000)
--log-level LEVEL      Log level (default: info)
```

### Supported Models
- `openbmb/VoxCPM2` - Multilingual voice cloning and design
- `openbmb/VoxCPM1.5` - Chinese and English voice cloning (stabler than `openbmb/VoxCPM-0.5B`)
- `openbmb/VoxCPM-0.5B` - Chinese and English voice cloning
- `k2-fsa/OmniVoice` - Multilingual voice cloning and design
- `meituan-longcat/LongCat-AudioDiT-1B` - Chinese and English voice cloning
- `meituan-longcat/LongCat-AudioDiT-3.5B` - Chinese and English voice cloning (larger)

## 🐛 Troubleshooting

### Common Issues

1. **"Speaker with voice_id 'xxx' not found"**
   - Check that files exist in `samples/` directory
   - Verify file naming: `{voice_id}.wav` or `{voice_id}.design.txt`
   - Ensure server has read permissions

2. **"Current model does not support voice design"**
   - LongCat-AudioDiT doesn't support voice design
   - Switch to OmniVoice: `POST /switch_model` with `{"model_name": "k2-fsa/OmniVoice"}`

3. **Model loading failures**
   - Ensure CUDA is properly installed
   - Check internet connection for model downloads
   - Verify sufficient GPU memory

4. **Audio quality issues**
   - For voice cloning, ensure reference audio is clean
   - For voice design, use clear, specific descriptions
   - Adjust TTS settings via `/set_tts_settings`

### Logs
- Check server console output for detailed error messages
- Enable debug logging: `--log-level debug`
- Logs are written to console and optionally to file

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -e .

# Run tests
python -m pytest tests/

# Check code style
python -m black .
python -m flake8 .
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [VoxCPM](https://github.com/OpenBMB/VoxCPM) by OpenBMB
- [OmniVoice](https://github.com/k2-fsa/omnivoice) by Xiaomi
- [LongCat-AudioDiT](https://github.com/meituan-longcat/LongCat-AudioDiT) by Meituan
- [xtts-api-server](https://github.com/daswer123/xtts-api-server) by daswer123
- All contributors and the open-source community

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/MoRanYue/LocalSotaTalk/issues)
<!-- - **Documentation**: [GitHub Wiki](https://github.com/MoRanYue/LocalSotaTalk/wiki) -->
- **Discussions**: [GitHub Discussions](https://github.com/MoRanYue/LocalSotaTalk/discussions)

---

**Note**: This project is under development. API changes may occur between minor versions.