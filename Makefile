.PHONY: train eval generate_data

train:
	uv run seaseer/train.py

eval:
	uv run seaseer/eval.py

generate_data:
	uv run data/generate_data.py