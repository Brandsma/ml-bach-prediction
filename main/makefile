.PHONY: data

data:
	python ./data/input/midi2img/midi2img.py ./data/input/midi/ ./data/input/img/A/

train:
	python src/main.py --epochs 100 --batch_size 16 --checkpoints ./src/checkpoints --log_level INFO --training --output_number 8 ./data/input/img/ ./data/output/img/

test:
	python src/main.py --checkpoints ./src/checkpoints --log_level INFO --output_number 8 ./data/input/img/ ./data/output/img/
