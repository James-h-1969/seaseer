.PHONY: train eval generate_data

train:
	uv run seaseer/train.py

eval:
	uv run seaseer/eval.py

generate_data:
	uv run helpers/scripts/generate_data.py