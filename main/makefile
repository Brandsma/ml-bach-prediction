.PHONY: data

data:
	python ./data/input/midi2img/midi2img.py ./data/input/midi/ ./data/input/img/A/

train:
	python src/main.py --epochs 200 --batch_size 32 --checkpoints ./src/checkpoints --log_level INFO --training --output_number 4 ./data/input/img/ ./data/output/img/

test:
	python src/main.py --checkpoints ./src/checkpoints --log_level INFO --output_number 4 ./data/input/img/ ./data/output/img/
