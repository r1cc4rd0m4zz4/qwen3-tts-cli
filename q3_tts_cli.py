#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "qwen-tts>=0.0.5,<1.0.0",
#     "torch>=2.4.0,<2.6.0",
#     "click>=8.0.0,<9.0.0",
#     "numpy>=1.24.0,<3.0.0",
#     "soundfile>=0.12.0,<1.0.0",
#     "librosa>=0.10.2,<1.0.0",
#     "numba>=0.59.0,<1.0.0",
#     "tiktoken>=0.5.0,<1.0.0",
# ]
# ///
"""
Qwen3-TTS CLI - Generate high-quality speech using Qwen3-TTS VoiceDesign model.

Note: This tool was created with AI agents (Claude) based on user specifications written by Riccardo Mazza.

This script uses the official qwen-tts package for proper model inference.
Supports voice design instructions for customizing voice characteristics.

Usage with uv (recommended):
    uv run q3_tts_cli.py "Hello world" -i "warm friendly voice"
    uv run q3_tts_cli.py "Ciao!" -l Italian -v
    
Or make it executable:
    chmod +x q3_tts_cli.py
    ./q3_tts_cli.py "Hello world" -i "clear voice"

Note: FFmpeg is required for MP3/M4A output (install via brew/apt/yum)

Security:
    - Output paths are validated to prevent directory traversal
    - Subprocess calls use list arguments to prevent injection
    - Temporary files are cleaned up properly
"""
import subprocess
import sys
from pathlib import Path
from typing import Optional

import click
import soundfile as sf
import tiktoken
import torch

# Lazy import to speed up --help
_model = None


def get_unique_filename(base_path: Path) -> Path:
    """Return a unique filename, adding -2, -3, etc. if the file already exists.
    
    Args:
        base_path: The desired output path
        
    Returns:
        A unique Path that doesn't exist yet
    """
    if not base_path.exists():
        return base_path
    
    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent
    
    counter = 2
    while True:
        new_path = parent / f"{stem}-{counter}{suffix}"
        if not new_path.exists():
            return new_path
        counter += 1


def estimate_tokens(text: str) -> int:
    """Calculate the number of tokens in a text string using tiktoken.
    
    Uses the cl100k_base encoding (used by GPT-4 and similar models) which
    provides accurate token counting for multilingual text.
    
    Args:
        text: Input text to count tokens for
        
    Returns:
        Actual number of tokens
    """
    try:
        # Use cl100k_base encoding (GPT-4, GPT-3.5-turbo)
        # This is a good general-purpose tokenizer for multilingual text
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception:
        # Fallback to simple estimation if tiktoken fails
        # ~4 chars per token is a reasonable approximation
        return len(text) // 4


def validate_output_path(output: str) -> Path:
    """Validate and sanitize output path to prevent directory traversal.
    
    Args:
        output: User-provided output filename
        
    Returns:
        Validated Path object
        
    Raises:
        click.UsageError: If path contains invalid characters or traversal attempts
    """
    # Resolve to absolute path and check for directory traversal
    try:
        output_path = Path(output).resolve()
        cwd = Path.cwd().resolve()
        
        # Ensure output is in current directory or subdirectory
        # This prevents writing to arbitrary locations like /etc/passwd
        try:
            output_path.relative_to(cwd)
        except ValueError:
            raise click.UsageError(
                f"Output path must be within current directory. "
                f"Got: {output_path}, Expected under: {cwd}"
            )
        
        # Check for suspicious characters
        if ".." in str(output):
            raise click.UsageError("Output path cannot contain '..' (parent directory references)")
            
        return output_path
    except Exception as e:
        if isinstance(e, click.UsageError):
            raise
        raise click.UsageError(f"Invalid output path: {e}")


class CustomHelpCommand(click.Command):
    """Custom Click command that adds usage examples to help text."""
    
    def format_help(self, ctx, formatter):
        super().format_help(ctx, formatter)
        prog = ctx.info_name
        with formatter.section("Examples"):
            formatter.write_paragraph()
            formatter.write_text(f'{prog} "say this text out loud" -i "friendly voice"')
            formatter.write_text(f'{prog} -o saved.wav "hello world" -i "deep voice"')
            formatter.write_text(f'{prog} -l Chinese "你好世界" -i "warm voice"')
            formatter.write_text(f'{prog} -i "deep low voice" "hello" -f mp3')
            formatter.write_text(f'echo "piped text" | {prog} -i "clear voice"')


def get_device() -> str:
    """Detect and return the best available device for inference.
    
    Priority: CUDA GPU > Apple Silicon (MPS) > CPU
    
    Returns:
        Device string: 'cuda:0', 'mps', or 'cpu'
    """
    if torch.cuda.is_available():
        return "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_torch_dtype(device: str) -> torch.dtype:
    """Get the optimal dtype for the specified device.
    
    Args:
        device: Device string ('cuda:0', 'mps', 'cpu')
        
    Returns:
        torch.bfloat16 for CUDA (more stable than float16), torch.float32 otherwise
    """
    if "cuda" in device:
        return torch.bfloat16  # bfloat16 is more stable than float16
    return torch.float32


def load_model(device: str, verbose: bool = False):
    """Load the Qwen3-TTS VoiceDesign model with caching.
    
    Args:
        device: Target device ('cuda:0', 'mps', or 'cpu')
        verbose: Whether to print loading messages
        
    Returns:
        Loaded Qwen3TTSModel instance
        
    Note:
        Model is cached globally to avoid reloading on subsequent calls.
        Uses eager attention implementation (flash_attention requires
        complex installation and may show compatibility warnings).
    """
    global _model
    if _model is not None:
        return _model
    
    from qwen_tts import Qwen3TTSModel
    
    if verbose:
        click.echo("Loading Qwen3-TTS VoiceDesign model...")
    
    dtype = get_torch_dtype(device)
    
    # Use eager attention implementation
    # Note: flash_attention_2 provides better performance but requires
    # additional dependencies and may show compatibility warnings
    _model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        device_map=device,
        dtype=dtype,
        attn_implementation="eager",
    )
    
    return _model


@click.command(cls=CustomHelpCommand)
@click.argument("text", required=False)
@click.option("-o", "--output", default="output.wav", help="Output filename (default: output.wav)")
@click.option("-f", "--format", "output_format", type=click.Choice(["wav", "mp3", "m4a"], case_sensitive=False), default="wav", help="Output format (default: wav)")
@click.option("-l", "--language", default="English", help="Language for TTS (default: English). Supported: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian, Auto")
@click.option("-i", "--instruct", default=None, help="Voice design instruction (e.g., 'Warm female voice with gentle tone')")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
@click.option("-d", "--device", default=None, help="Device to use (cuda:0, cpu, mps). Auto-detected if not specified.")
def main(text: Optional[str], output: str, output_format: str, language: str, instruct: Optional[str], verbose: bool, device: Optional[str]):
    """Generate audio using Qwen3-TTS VoiceDesign model.
    
    This tool uses the VoiceDesign model which allows you to describe the voice
    characteristics you want using natural language instructions.
    
    TEXT: The text to convert to speech. Can also be piped via stdin.
    
    Examples:
        uv run q3_tts_cli.py "Hello world" -i "friendly voice"
        echo "Test" | uv run q3_tts_cli.py -i "deep voice" -f mp3
    """
    # Handle piped input
    if text is None:
        if not sys.stdin.isatty():
            text = sys.stdin.read().strip()
        else:
            raise click.UsageError("No text provided. Pass text as an argument or pipe it via stdin.")
    
    if not text:
        raise click.UsageError("Text cannot be empty.")
    
    # Validate text length to prevent excessive resource usage
    # Using token estimation instead of raw character count
    estimated_tokens = estimate_tokens(text)
    max_tokens = 5000
    if estimated_tokens > max_tokens:
        raise click.UsageError(
            f"Text too long (~{estimated_tokens} tokens estimated). "
            f"Maximum {max_tokens} tokens allowed. "
            f"Current text: {len(text)} chars, {len(text.split())} words"
        )
    
    # Set default voice instruction if not provided
    if not instruct:
        instruct = "Natural, clear voice with moderate pace and friendly tone"
        if verbose:
            click.echo(f"No voice instruction provided, using default: '{instruct}'")
    
    # Validate instruction length using token estimation
    instruction_tokens = estimate_tokens(instruct)
    max_instruction_tokens = 500
    if instruction_tokens > max_instruction_tokens:
        raise click.UsageError(
            f"Voice instruction too long (~{instruction_tokens} tokens estimated). "
            f"Maximum {max_instruction_tokens} tokens allowed."
        )
    
    # Determine device
    if device is None:
        device = get_device()
    
    if verbose:
        click.echo(f"Using device: {device}")
    
    # Validate and determine output path with correct extension
    if output == "output.wav":
        # Default output - use format-specific extension
        output_path = Path(f"output.{output_format.lower()}")
        output_path = get_unique_filename(output_path)
    else:
        # User-specified output - validate for security
        output_path = validate_output_path(output)
        if not output_path.suffix:
            output_path = Path(f"{output}.{output_format.lower()}")
    
    # Load model (cached after first call)
    model = load_model(device, verbose)
    
    if verbose:
        click.echo(f"Generating audio for: {text[:50]}{'...' if len(text) > 50 else ''}")
        click.echo(f"Voice instruction: {instruct[:50]}{'...' if len(instruct) > 50 else ''}")
        click.echo(f"Language: {language}")
    
    # Generate audio using the VoiceDesign API
    # Parameters optimized for quality and diversity
    wavs, sample_rate = model.generate_voice_design(
        text=text,
        language=language,
        instruct=instruct,
        do_sample=True,       # Enable sampling for more natural output
        temperature=0.9,      # Slightly creative while maintaining quality
        top_k=50,            # Limit vocabulary for stability
        top_p=1.0,           # Nucleus sampling disabled
        max_new_tokens=4096, # Maximum audio length
    )
    
    # Extract generated audio (first item from batch)
    audio = wavs[0]
    
    # Save to file
    if output_format.lower() == "wav":
        # Direct WAV output - no conversion needed
        sf.write(str(output_path), audio, sample_rate)
        if verbose:
            click.echo(f"Audio saved to: {output_path}")
    else:
        # Convert to MP3/M4A using FFmpeg
        temp_wav = output_path.with_suffix(".tmp.wav")  # Use .tmp.wav extension for soundfile
        try:
            # First save as temporary WAV (explicitly specify format)
            sf.write(str(temp_wav), audio, sample_rate, format='WAV')
            
            if verbose:
                click.echo(f"Converting to {output_format.upper()}...")
            
            # Configure FFmpeg codec and bitrate for voice optimization
            if output_format.lower() == "mp3":
                codec = "libmp3lame"
                bitrate = "96k"  # Optimal quality/size ratio for voice
            else:  # m4a
                codec = "aac"
                bitrate = "64k"  # AAC is more efficient, 64k sufficient for voice
            
            # Run FFmpeg conversion
            # Using list arguments prevents command injection
            subprocess.run(
                ["ffmpeg", "-i", str(temp_wav), "-c:a", codec, "-b:a", bitrate, "-y", str(output_path)],
                check=True,
                capture_output=not verbose,
                text=True
            )
            
            if verbose:
                click.echo(f"Audio saved to: {output_path}")
                
        except subprocess.CalledProcessError as e:
            raise click.ClickException(f"FFmpeg conversion failed: {e}")
        except FileNotFoundError:
            raise click.ClickException(
                "FFmpeg not found. Please install FFmpeg to use MP3/M4A formats.\n"
                "Install: apt install ffmpeg (Ubuntu/Debian) or brew install ffmpeg (macOS)"
            )
        finally:
            # Always clean up temporary file
            if temp_wav.exists():
                temp_wav.unlink()
    
    if not verbose:
        click.echo(str(output_path))


if __name__ == "__main__":
    main()
