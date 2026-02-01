# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-02-01

### Added
- Accurate token counting using tiktoken library
- Support for multilingual token estimation with cl100k_base encoding
- Fallback estimation mechanism if tiktoken fails

### Changed
- Token limits now use actual token counts instead of character estimates:
  - Text input: 5000 tokens (previously ~5000 chars)
  - Voice instructions: 500 tokens (previously ~500 chars)
- More accurate validation for multilingual text
- Better error messages showing token count, character count, and word count
- **Script renamed**: `q3_tts_cuda.py` → `q3_tts_cli.py` (more accurate name reflecting CLI nature)

### Technical
- Added tiktoken>=0.5.0 dependency
- Updated `estimate_tokens()` function to use tiktoken encoding
- Uses GPT-4's cl100k_base encoding for universal compatibility

## [1.0.0] - 2026-02-01

### Added
- Initial release of Qwen3-TTS CLI
- Multi-language support (11 languages)
- Natural language voice design instructions
- Multiple output formats (WAV, MP3, M4A)
- GPU acceleration (CUDA, MPS, CPU fallback)
- Security features:
  - Input validation (text/instruction length limits)
  - Path sanitization (directory traversal prevention)
  - Secure subprocess handling
  - Automatic cleanup of temporary files
- Comprehensive documentation
- uv package manager integration
- Verbose mode for debugging
- Piped input support
- Auto-detection of optimal device
- Custom help with examples

### Security
- Text input limited to 5000 characters
- Voice instructions limited to 500 characters
- Output paths validated to prevent directory traversal
- Subprocess calls use list arguments (prevents injection)
- Temporary files cleaned up on success and error

### Performance
- Model caching (loads once, reuses for subsequent calls)
- Optimized audio encoding (96k for MP3, 64k for M4A)
- bfloat16 precision on CUDA for better stability
- Eager attention implementation (no flash-attn dependency)

### Documentation
- Comprehensive README with examples
- Troubleshooting guide
- Security best practices
- Performance benchmarks
- Voice design tips

## [Unreleased]

### Desired Features
- Batch processing mode
- Voice cloning from audio samples
- Real-time streaming output
- Web UI interface
- Docker container support
