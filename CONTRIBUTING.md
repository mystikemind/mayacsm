# Contributing

Thank you for your interest in contributing to Maya.

## Getting Started

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Set up your development environment (see README.md)
4. Make your changes
5. Run the test suite: `python -m pytest tests/`
6. Submit a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/maya-csm1b-whole-pipeline.git
cd maya-csm1b-whole-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install pytest pytest-asyncio black isort mypy
```

## Code Style

- **Formatting**: Use `black` with default settings
- **Imports**: Use `isort` for import ordering
- **Type hints**: Preferred for all public functions
- **Docstrings**: Google style

## Architecture Guidelines

- All configuration goes in `maya/config.py`
- Engine implementations go in `maya/engine/`
- Pipeline orchestration goes in `maya/pipeline/`
- Keep latency in mind — every millisecond counts

## Testing

```bash
# Run unit tests
python -m pytest tests/

# Test latency
python scripts/test_full_latency.py

# Test audio quality
python scripts/test_audio_quality.py
```

## Reporting Issues

Please include:
- Hardware (GPU model, VRAM)
- Python/PyTorch/CUDA versions
- Steps to reproduce
- Logs (from `python run.py --debug`)
