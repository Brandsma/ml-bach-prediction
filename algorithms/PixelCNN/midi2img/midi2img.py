# Code copied from: https://github.com/mathigatti/midi2img/blob/master/midi2img.py
from music21 import converter, instrument, note, chord
import json
import sys
import numpy as np
from imageio import imwrite

def extractNote(element):
    return int(element.pitch.ps)

def extractDuration(element):
    return element.duration.quarterLength

def get_notes(notes_to_parse):

    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    durations = []
    notes = []
    start = []

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            if element.isRest:
                continue

            start.append(element.offset)
            notes.append(extractNote(element))
            durations.append(extractDuration(element))
                
        elif isinstance(element, chord.Chord):
            if element.isRest:
                continue
            for chord_note in element.notes:
                start.append(element.offset)
                durations.append(extractDuration(element))
                notes.append(extractNote(chord_note))

    return {"start":start, "pitch":notes, "dur":durations}

def midi2image(midi_path):
    mid = converter.parse(midi_path)

    instruments = instrument.partitionByInstrument(mid)

    data = {}

    try:
        i=0
        for instrument_i in instruments.parts:
            notes_to_parse = instrument_i.recurse()

            if instrument_i.partName is None:
                data["instrument_{}".format(i)] = get_notes(notes_to_parse)
                i+=1
            else:
                data[instrument_i.partName] = get_notes(notes_to_parse)

    except:
        notes_to_parse = mid.flat.notes
        data["instrument_0".format(i)] = get_notes(notes_to_parse)

    resolution = 0.25

    for instrument_name, values in data.items():
        # https://en.wikipedia.org/wiki/Scientific_pitch_notation#Similar_systems
        upperBoundNote = 127 # height of image
        lowerBoundNote = 21
        maxSongLength = 128 # length of image

        # padding TODO
        padding = False

        if maxSongLength%2 != 0:
            print("Warning: padding selected but length not even, change maxSongLength = {} to be an even number".format(maxSongLength))
        
        padding_size = int((maxSongLength - (upperBoundNote - lowerBoundNote)) / 2)
        print(padding_size)

        # calculate the amount of images required to display the full song
        images = int((len(values["pitch"]) // maxSongLength)) + 5
        print("{} {} {}".format(len(values["pitch"]), maxSongLength, images))

        index = 0
        prev_index = 0
        repetitions = 0

        # TODO: number of images generated is too large, seems to have something to do with that the len(values[pitch]) is larger than supposed (?).
        # so the basecase is never reached, and the prev_index is not increased anymore after a while, meaning empty loops are performed creating empty images

        while repetitions < images:
            if prev_index >= len(values["pitch"]):
                break

            # Image matrix
            if padding:
                matrix = np.zeros((maxSongLength,maxSongLength))
            else :
                matrix = np.zeros((upperBoundNote-lowerBoundNote,maxSongLength))

            pitchs = values["pitch"]
            durs = values["dur"]
            starts = values["start"]

            # From where we left off to the end
            for i in range(prev_index,len(pitchs)):
                pitch = pitchs[i]

                dur = int(durs[i]/resolution)
                start = int(starts[i]/resolution)

                # if we're not at the end of the image
                if dur+start - index*maxSongLength < maxSongLength:
                    # loop over something*
                    for j in range(start,start+dur):
                        if j - index*maxSongLength >= 0:
                            # with padding: add some padding pixels to the bottom (already added to the top by the matrix size) to create a square image
                            if padding:
                                matrix[pitch-lowerBoundNote + padding_size,j - index*maxSongLength] = 255
                            else:
                                matrix[pitch-lowerBoundNote,j - index*maxSongLength] = 255
                # when we are, break
                else:
                    prev_index = i
                    break

            imwrite(midi_path.split("/")[-1].replace(".mid",f"_{instrument_name}_{index}.png"),matrix)
            index += 1
            repetitions+=1

import sys
midi_path = sys.argv[1]
midi2image(midi_path)