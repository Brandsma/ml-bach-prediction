# Code copied from: https://github.com/mathigatti/midi2img/blob/master/midi2img.py
import os
import importlib.util
from music21 import converter, instrument, note, chord
import json
import sys
import numpy as np
from imageio import imwrite
import math


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

    return {"start": start, "pitch": notes, "dur": durations}


def midi2binary(midi_path):
    # returns a dictionionary with keys = part numbers, value = list with np binary matrices of the image.
    return midi2image(midi_path, True)


def midi2image(midi_path, lowest, highest, binary=False):
    # The lowest and highest parameters indicate what the midi code is of 
    # the lowest and highest notes respectively, and image resolution is adjusted to this 
    # Warning: function returns nothing (None), but catch the return anyway
    mid = converter.parse(midi_path)
    binary_matrix_list = {}  # parts
    parts = mid.recurse().getElementsByClass('Part')

    for partidx, midpart in enumerate(parts):

        data = {}

        notes_to_parse = midpart.recurse()
        data["part_{}".format(partidx)] = get_notes(notes_to_parse)

        resolution = 0.25

        for part_name, values in data.items():
            # https://en.wikipedia.org/wiki/Scientific_pitch_notation#Similar_systems
            upperBoundNote = highest + 1  # height of image
            lowerBoundNote = lowest
            maxSongLength = 128  # length of image

            # list of entries in part (each of which will contrain matrices)
            binary_matrix_list[part_name] = []

            # padding
            padding = False

            if maxSongLength % 2 != 0:
                print("Warning: padding selected but length not even, change maxSongLength = {} to be an even number".format(
                    maxSongLength))

            padding_size = int(
                (maxSongLength - (upperBoundNote - lowerBoundNote)) / 2)

            # calculate the amount of images required to display the full song
            images = math.ceil(
                ((max(values["start"]) + max(values["dur"])) * 4) / maxSongLength)

            index = 0
            prev_index = 0
            repetitions = 0

            while repetitions < images:
                if prev_index >= len(values["pitch"]):
                    break

                # Image matrix
                if padding:
                    matrix = np.zeros((maxSongLength, maxSongLength))
                else:
                    matrix = np.zeros(
                        (upperBoundNote-lowerBoundNote, maxSongLength))

                pitchs = values["pitch"]
                durs = values["dur"]
                starts = values["start"]

                # From where we left off to the end
                for i in range(prev_index, len(pitchs)):
                    pitch = pitchs[i]

                    dur = int(durs[i]/resolution)
                    start = int(starts[i]/resolution)

                    # if we're not at the end of the image
                    if dur+start - index*maxSongLength < maxSongLength:
                        # loop over something*
                        for j in range(start, start+dur):
                            if j - index*maxSongLength >= 0:
                                # with padding: add some padding pixels to the bottom (already added to the top by the matrix size) to create a square image
                                if padding:
                                    matrix[pitch-lowerBoundNote + padding_size,
                                           j - index*maxSongLength] = 255
                                else:
                                    matrix[pitch-lowerBoundNote, j -
                                           index*maxSongLength] = 255
                    # when we are, break
                    else:
                        prev_index = i
                        break
                if not (matrix == np.zeros(matrix.shape)).all() and not binary:
                    imwrite(midi_path.split(
                        "/")[-1].replace(".mid", f"_{part_name}_{index}.png"), matrix)
                elif binary:
                    binary_matrix_list[part_name].append(matrix > 0)
                else:
                    print("Empty generated image discarded!")
                index += 1
                repetitions += 1
    if binary:
        return binary_matrix_list
    else:
        # Abe may refractor if he wants, we dont want
        return None


midi_path = sys.argv[1]
directory = os.listdir(midi_path)

# We're gonna do two passes over all midi files: the first will index thehighest and lowest midi notes found,
# thus specifying the range of pixels we need in the vertical direction. This is then passed as an argument to midi2img
# and stored in a textfile named "latestImageSettings.txt".

mid = converter.parse(midi_path + directory[0])
lowestnote = int(mid.recurse().notes[0].pitch.ps)
highestnote = lowestnote

for filename in directory:
    mid = converter.parse(midi_path + filename)
    # first just set max and min to some value we know occurs:
    
    # Now let's find the actual highest and lowest
    for item in mid.recurse().notes:
        highestnote = max(highestnote, int(item.pitch.ps))
        lowestnote = min(lowestnote, int(item.pitch.ps))

# Write this specification to the file:
specfile = open("latestImageSettings.txt", "w")
specfile.write("{}\n{}".format(lowestnote, highestnote))

for filename in directory:
    matrix = midi2image(midi_path + filename, lowestnote, highestnote)
