import os
import numpy as np
import midi
import mido
from midiutil import MIDIFile
from markov_chain_model import generate_markov_chain
from midiplayer import *

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

    


def load_music_txt(filename = "sample/F.txt"):
    # Reads music from a txt file
    F = np.loadtxt(filename)
    return F

def load_music_midi(file):
    # Reads midi file, outputs a rawdata format of music (like F.txt)
    #TODO WIP
    mid = mido.MidiFile(file)
    # Summary of midi file
    print(mid)

    # Find out how many channels there are
    channels = [False for i in range(16)]
    print("Loading file: {}...".format(file))
    for msg in mid.play():
        print(msg)
        if not channels[msg.channel]:
            channels[msg.channel] = True
        
    
    print(channels)
    print(np.sum(channels))


def main():
    # First off: some configuration:
    m_F = "sample/F.txt"
    m_output = "output/BachFromTheDead.mid"

    # Load the training data
    music_file = load_music_txt()

    # TODO: Transform the data into something useful for the ML algorithm

    # Generate the markov chains, one for each
    mc = generate_markov_chain(music_file)

    # Generate the new music
    generated_music = np.array([mc[idx].generate_states(music_file[0, idx], no=200) for idx in range(4)]).T
    
    # Transform the composition to Midi and write it to a file
    C = Composition(generated_music)
    C.writeMIDI(m_output)

    #TODO: WIP function; not yet functional
    #load_music_midi("BachFromTheDead.mid")


    # Finally; play the generated music file
    midi_filename = m_output
    try:
        # use the midi file you just saved
        play_music(midi_filename)
    except KeyboardInterrupt:
        # if user hits Ctrl/C then exit
        # (works only in console mode)
        pygame.mixer.music.fadeout(1000)
        pygame.mixer.music.stop()
    raise SystemExit

if __name__=="__main__":
    main()