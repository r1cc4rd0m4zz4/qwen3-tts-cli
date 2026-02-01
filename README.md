# Qwen3-TTS CLI

A command-line interface for generating high-quality speech using the Qwen3-TTS VoiceDesign model. This tool leverages natural language voice design instructions to create customizable text-to-speech output.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Features

- 🎯 **Natural Voice Design**: Describe the voice you want using natural language (e.g., "warm friendly voice", "deep authoritative tone")
- 🌍 **Multi-language Support**: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian, and Auto-detect
- 🚀 **GPU Acceleration**: Automatic CUDA/MPS detection for fast inference
- 📦 **Multiple Output Formats**: WAV, MP3, M4A with optimized encoding
- 🔒 **Security Focused**: Input validation, path sanitization, and secure subprocess handling
- ⚡ **Zero Configuration**: Uses `uv` for dependency management - no virtual environment setup needed

## Requirements

### System Requirements

- **Python**: 3.12 or higher
- **GPU** (recommended): NVIDIA GPU with CUDA support or Apple Silicon with MPS
  - CPU inference is supported but significantly slower
- **Storage**: ~7GB for model weights (downloaded automatically on first run)
- **RAM**: Minimum 16GB recommended

### External Dependencies

- **FFmpeg** (required for MP3/M4A output):
  ```bash
  # Ubuntu/Debian
  sudo apt install ffmpeg
  
  # macOS
  brew install ffmpeg
  
  # Fedora/RHEL
  sudo dnf install ffmpeg
  ```

- **uv** (Python package installer):
  ```bash
  # Install uv if you don't have it
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

## Installation

### One-liner with uvx (Simplest)

Run directly from the internet without downloading anything:

```bash
uvx --script https://raw.githubusercontent.com/r1cc4rd0m4zz4/qwen3-tts-cli/main/q3_tts_cli.py "Hello world" -i "friendly voice"
```

### Quick Start by Downloading

If you prefer to have the script locally:

```bash
# Download the script
curl -O https://raw.githubusercontent.com/r1cc4rd0m4zz4/qwen3-tts-cli/main/q3_tts_cli.py

# Make it executable
chmod +x q3_tts_cli.py

# Run it (dependencies install automatically)
./q3_tts_cli.py "Hello world" -i "friendly voice"
```

### Clone Repository

```bash
git clone https://github.com/r1cc4rd0m4zz4/qwen3-tts-cli.git
cd qwen3-tts-cli
chmod +x q3_tts_cli.py
```

## Usage

### Basic Usage

```bash
# Using uv run (recommended)
uv run q3_tts_cli.py "Hello world" -i "warm friendly voice"

# Or if made executable
./q3_tts_cli.py "Hello world" -i "warm friendly voice"
```

### Command Line Options

```
Usage: q3_tts_cli.py [OPTIONS] [TEXT]

Options:
  -o, --output PATH          Output filename (default: output.wav)
  -f, --format [wav|mp3|m4a] Output format (default: wav)
  -l, --language TEXT        Language for TTS (default: English)
  -i, --instruct TEXT        Voice design instruction
  -v, --verbose              Enable verbose output
  -d, --device TEXT          Device to use (cuda:0, cpu, mps)
  --help                     Show this message and exit
```

### Examples

#### Basic Text-to-Speech

```bash
# Simple generation with default voice
uv run q3_tts_cli.py "Hello, how are you?" -i "clear voice"

# Save to specific file
uv run q3_tts_cli.py "Welcome!" -i "warm voice" -o welcome.wav

# Generate MP3 instead of WAV
uv run q3_tts_cli.py "Test audio" -i "deep voice" -f mp3
```

#### Voice Customization

```bash
# Female voice
uv run q3_tts_cli.py "Hi there!" -i "warm female voice with gentle tone"

# Male voice with specific characteristics
uv run q3_tts_cli.py "Good morning" -i "deep male voice, authoritative and clear"

# Energetic young voice
uv run q3_tts_cli.py "Let's go!" -i "energetic young voice, excited and upbeat"

# Calm narrator
uv run q3_tts_cli.py "Once upon a time..." -i "calm narrator voice, slow paced"
```

#### Multi-language Support

```bash
# Chinese
uv run q3_tts_cli.py "你好世界" -l Chinese -i "温暖的声音"

# Spanish
uv run q3_tts_cli.py "Hola mundo" -l Spanish -i "voz cálida"

# Japanese
uv run q3_tts_cli.py "こんにちは" -l Japanese -i "優しい声"

# Auto-detect language
uv run q3_tts_cli.py "Bonjour!" -l Auto -i "friendly voice"
```

#### Piped Input

```bash
# From echo
echo "This is a test" | uv run q3_tts_cli.py -i "clear voice"

# From file
cat speech.txt | uv run q3_tts_cli.py -i "narrator voice" -o speech.mp3

# From command output
fortune | uv run q3_tts_cli.py -i "wise voice" -f mp3
```

#### Advanced Options

```bash
# Verbose output to see progress
uv run q3_tts_cli.py "Test" -i "clear voice" -v

# Force CPU usage (when GPU available)
uv run q3_tts_cli.py "Test" -i "clear voice" -d cpu

# Specific CUDA device
uv run q3_tts_cli.py "Test" -i "clear voice" -d cuda:0
```

## Voice Design Tips

The voice instruction parameter (`-i`) accepts natural language descriptions. Here are some effective patterns:

### Tone & Emotion
- "warm and friendly"
- "professional and authoritative"
- "calm and soothing"
- "energetic and enthusiastic"
- "serious and formal"

### Gender & Age
- "young female voice"
- "mature male voice"
- "elderly narrator"
- "middle-aged professional"

### Speaking Style
- "slow and deliberate"
- "fast-paced and excited"
- "conversational and casual"
- "clear and articulate"

### Combined Examples
- "warm female voice with gentle tone and slow pace"
- "deep male voice, authoritative and clear, professional broadcaster"
- "energetic young voice, excited and upbeat, radio DJ style"

## Dependencies

q3_tts_cli.py automatically manages the following Python dependencies via `uv`:

## Performance

### Inference Speed (approximate)

| Device | Speed | Quality |
|--------|-------|---------|
| NVIDIA RTX 4090 | ~2x real-time | Excellent |
| NVIDIA RTX 3080 | ~1.5x real-time | Excellent |
| Apple M1 Max (MPS) | ~1x real-time | Excellent |
| CPU (12-core) | ~0.3x real-time | Excellent |

### Model Size

- **Model**: Qwen3-TTS-12Hz-1.7B-VoiceDesign
- **Download Size**: ~3.5GB
- **Disk Space**: ~7GB (with cache)
- **VRAM Usage**: ~4GB (GPU inference)

## Troubleshooting

### Common Issues

#### 1. "FFmpeg not found"

**Problem**: MP3/M4A conversion fails
```
Error: FFmpeg not found. Please install ffmpeg to use MP3/M4A formats.
```

**Solution**:
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

#### 2. "CUDA out of memory"

**Problem**: GPU runs out of memory

**Solution**:
```bash
# Use CPU instead
uv run q3_tts_cli.py "text" -i "voice" -d cpu

# Or try shorter text segments
```

#### 3. Flash Attention Warning

**Problem**: Warning message about flash-attn

```
Warning: flash-attn is not installed. Will only run the manual PyTorch version.
```

**Solution**: This is informational only. The script works fine without flash-attn. It's an optional performance optimization that requires complex installation.

#### 4. Slow First Run

**Problem**: First execution takes a long time

**Solution**: This is normal - the model (~3.5GB) is being downloaded. Subsequent runs will be much faster as the model is cached.

#### 5. Permission Denied

**Problem**: Cannot write output file

**Solution**:
```bash
# Check current directory permissions
ls -la

# Or specify a different output location
uv run q3_tts_cli.py "text" -i "voice" -o ~/Desktop/output.wav
```

### Debug Mode

Enable verbose output to see detailed information:

```bash
uv run q3_tts_cli.py "test" -i "voice" -v
```

This shows:
- Device selection
- Model loading progress
- Audio generation details
- File conversion steps

## Security

This script implements several security best practices:

- ✅ **Input Validation**: Token limits using tiktoken (5000 tokens for text, 500 for instructions)
- ✅ **Path Sanitization**: Prevents directory traversal attacks
- ✅ **Subprocess Safety**: Uses list arguments to prevent command injection
- ✅ **Resource Cleanup**: Proper cleanup of temporary files
- ✅ **Dependency Pinning**: Version constraints prevent vulnerable packages

### Security Limitations

- Output files are written in the current directory or subdirectories only
- Text input limited to 5000 tokens (calculated using tiktoken)
- Voice instructions limited to 500 tokens (calculated using tiktoken)
- Temporary files are cleaned up even on errors

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

```bash
# Clone repository
git clone https://github.com/r1cc4rd0m4zz4/qwen3-tts-cli.git
cd qwen3-tts-cli

# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run tests
uv run q3_tts_cli.py "test" -i "clear voice" -v
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright © 2026 Riccardo Mazza

**Disclaimer:** This tool was created with AI agents (Claude) based on user specifications written by Riccardo Mazza.

The Qwen3-TTS model is developed by Alibaba and subject to its own license terms.

## Acknowledgments

- **Qwen Team** at Alibaba for the excellent TTS model
- **Astral** for the amazing `uv` package manager
- All contributors and users of this project

## Citation

If you use this tool in your research or project, please cite:

```bibtex
@software{qwen3_tts_cli,
  title = {Qwen3-TTS CLI: Command-line interface for Qwen3-TTS},
  author = {Riccardo Mazza},
  year = {2026},
  url = {https://github.com/r1cc4rd0m4zz4/qwen3-tts-cli}
}
```

## Links

- [Qwen3-TTS Model](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign)
- [Official Qwen3-TTS Repository](https://github.com/QwenLM/Qwen3-TTS)
- [Qwen3-TTS Blog Post](https://qwen.ai/blog?id=qwen3tts-0115)
- [uv Package Manager](https://github.com/astral-sh/uv)
- [Issue Tracker](https://github.com/r1cc4rd0m4zz4/qwen3-tts-cli/issues)

---

**Note**: This is an unofficial community tool. For official Qwen TTS support, please refer to the [official Qwen repository](https://github.com/QwenLM/Qwen).
