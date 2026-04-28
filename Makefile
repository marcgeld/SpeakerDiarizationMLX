.PHONY: help sync run test clean

# Pass CLI options to diarization like:
# make run ARGS="--audio-path sample_audio/conversation_english.wav --no-plot"
ARGS ?=

help:
	@echo "Available targets:"
	@echo "  make sync   - Install/update dependencies with uv"
	@echo "  make run    - Run diarization (use ARGS=\"...\" for CLI flags)"
	@echo "  make test   - Run test suite"
	@echo "  make clean  - Remove Python cache artifacts"

sync:
	uv sync --extra dev

run:
	uv run diarize $(ARGS)

test:
	uv run --extra dev pytest -q

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

