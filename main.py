import os
import numpy as np
from markov_chain_model import generate_markov_chain
import music

def getTextFromMidi(file):
    mid = mido.MidiFile(file)
    print(mid)

    #for msg in mid.play():
        

class Composition():
    # TODO functionality to add:
        # Read out the notes per voice with their duration
        # Be able to add midi files to this composition
        # Add a function which identifies which chord is being played
            # Group by letter, dim, min, maj, aug
            # (This is purely to delimit the "playing field" for the randomness of the neural net for example)
            # Must also identify and return moments when chords change

    def __init__(self, music_file):
        self.rawData = music_file

    def writeMIDI(self, filename):
        #TODO deprecated; use music21 for this now, and ditch the old music format
        # Will write midi file according to raw data:
        track = 0   # only 1 track anyway
        MyMIDI = MIDIFile(1, eventtime_is_ticks=False)    # 4 voices/channels, but just 1 track
        MyMIDI.addTempo(0, time = 0, tempo = 360)
        duration = 1.0   # in beats
        volume = 100     # fixed for lack of data
        
        for voice_idx, voice in enumerate(self.rawData.T):  
            for sixteenth_idx, pitch in enumerate(voice):
                if voice[sixteenth_idx] == 0:
                    continue
                MyMIDI.addNote(track, voice_idx, int(voice[sixteenth_idx]), sixteenth_idx, duration, volume)

        with open(filename, "wb") as output_file:
            MyMIDI.writeFile(output_file)

    




def main():
    music.to_vector_ts(music.read_midi())

if __name__=="__main__":
    main()