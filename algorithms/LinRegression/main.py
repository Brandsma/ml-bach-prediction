import music
from music21 import *

predicted = music.load_txt_to_stream(fp="predicted_score.txt")
predicted.show('midi')
