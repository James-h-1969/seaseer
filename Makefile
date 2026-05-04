.PHONY: train eval generate_data

train:
	uv run seaseer/train.py

eval:
	uv run seaseer/eval.py

generate_data:
	tmux new-session -d -s generate_data 'uv run helpers/scripts/generate_data.py; echo "Done. Press enter to close."; read'
	@echo "Running in tmux session 'generate_data'. Attach with: tmux attach -t generate_data"