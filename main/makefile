.PHONY: data

data:
	python ./data/input/midi2img/midi2img.py ./data/input/midi/ ./data/input/img/A/

train-test:
	python src/main.py --name "train-test" --epochs 2 --batch_size 2 --patience 10 --learning_rate 0.001 --dropout_rate 0.5 --optimizer adam --checkpoints ./src/checkpoints/ --image_size_file ./data/input/img/A/latestImageSettings.txt --log_level INFO --training --output_number 2 ./data/input/img/ ./data/output/img/

train-test-continue:
	python src/main.py --name "train-test-continue" --continue_from_checkpoint --epochs 2 --batch_size 2 --patience 10 --learning_rate 0.001 --dropout_rate 0.5 --optimizer adam --checkpoints ./src/checkpoints/ --image_size_file ./data/input/img/A/latestImageSettings.txt --log_level INFO --training --output_number 2 ./data/input/img/ ./data/output/img/

train:
	python src/main.py --epochs 200 --batch_size 16 --patience 10 --checkpoints ./src/checkpoints/ --log_level INFO --training --output_number 4 ./data/input/img/ ./data/output/img/

test:
	python src/main.py --checkpoints ./src/checkpoints/ --log_level INFO --output_number 4 ./data/input/img/ ./data/output/img/
