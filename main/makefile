.PHONY: data

data:
	python ./data/input/midi2img/midi2img.py ./data/input/midi/ ./data/input/img/A/

train:
	python src/main.py --name "first_train" --epochs 400 --batch_size 16 --patience 5 --learning_rate 0.01 --dropout_rate 0.5 --optimizer adam --checkpoints ./src/checkpoints/ --training --output_number 1 ./data/input/img/ ./data/output/img/

test:
	python src/main.py --name "first_train" --checkpoints ./src/checkpoints/ --log_level INFO --output_number 2 ./data/input/img/ ./data/output/img/
