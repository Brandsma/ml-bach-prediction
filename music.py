# This file will be a wrapper around music21,
# providing the functionality of easy file loading (midi or txt)
# and writing, as well as provides output formatted for NN in various ways

# By convention, different "voices" of a piece go into different stream.Part objects

import numpy as np
from music21 import *

environment.set("midiPath", "/usr/bin/musescore")


# The circle of fifths will be a central theme, and its encoding will be predefined:
# 0 means no note is played (rest)
# 0.5 to 1.0 are evenly spaced the notes:
# C, G, D, A, E, B, F#/Gb, Db/C#, Ab, Eb, Bb, F
cof = {
    "C": 1,
    "G": 2,
    "D": 3,
    "A": 4,
    "E": 5,
    "B": 6,
    "F#": 7,
    "G-": 7,
    "D-": 8,
    "C#": 8,
    "A-": 9,
    "G#": 9,
    "E-": 10,
    "D#": 10,
    "B-": 11,
    "A#": 11,
    "F": 12,
}
# This corresponds to the circle of fifths, and should make noise less painful


def load_txt_to_stream(fp="sample/F.txt"):
    # Read from text file into a music21 stream/part
    # TODO make it so that different voices go into different parts
    music = stream.Part()
    # Create a duration object for all notes:
    sixteenth_duration = duration.Duration("16th")
    F = np.loadtxt(fp)
    for sixteenth_idx, notes in enumerate(F):
        for channel_idx, thisnote in enumerate(notes):
            if thisnote != 0:
                newnote = note.Note(thisnote)
                newnote.duration = sixteenth_duration
            else:
                newnote = note.Rest()
                newnote.duration = sixteenth_duration

            music.insert(sixteenth_duration.quarterLength * sixteenth_idx, newnote)

    print(
        "Converting {} poorly formulated notes worth of data to midi; may take a while...".format(
            len(music)
        )
    )
    music.show("midi")

    return music


def write_midi(musicstream: stream.Stream, fp: str):
    mf = midi.translate.streamToMidiFile(musicstream)
    mf.open(fp, "wb")
    mf.write()
    mf.close()


def load_midi(fp: str = "sample/unfin.mid"):
    # By default, it is assumed that the
    # midi file contains a stream.Score object,
    # which in turn contains a number of stream.Part objects
    streamscore = midi.translate.midiFilePathToStream(fp)
    return streamscore


def pitch_encode(pitch: pitch.Pitch):
    # TODO: make this correct w.r.t. two different Circle of Fifths...
    # This function takes a pitch object and encodes it as follows:
    # 0 means no note is played (rest)
    # 0.5 to 1.0 are evenly spaced the notes:
    # C, G, D, A, E, B, F#/Gb, Db/C#, Ab, Eb, Bb, F
    # Not all notes are represented in the cof, as there is a major and minor cof...
    # Correcting for this effect, at least for the initial dataset:

    return 0.5 + (0.5 / 12) * cof[pitch.name]


def pitch_round(target_value: float, rest_cutoff: float = 0.25):
    if target_value < rest_cutoff:
        # This is in fact not a pitch but a rest
        target_value = 0.0
        return target_value
    elif target_value < 0.5:
        target_value += 0.5
    elif target_value > 1.0:
        target_value -= 0.5

    return int(round((target_value - 0.5) / 0.5 * 12, 0))


def pitchval_decode(target_value: float, rest_cutoff: float = 0.25):
    # Does the opposite of pitch_encode: should output a pitch
    if target_value < rest_cutoff:
        print(
            "Non-fatal error: pitch decoded was actually a rest, not a note: {}".format(
                target_value
            )
        )
    rounded_value = pitch_round(target_value, rest_cutoff)

    # if target_value < rest_cutoff:
    #     # This is in fact not a pitch but a rest
    #     return None
    # elif target_value < 0.5 - (0.5/11):
    #     target_value += 0.5
    # elif target_value > 1.0 + (0.5/11):
    #     target_value -= 0.5

    # The above should implement the cyclic behavior of the cof
    # Round to the nearest proper note:

    for name, val in cof.items():
        if val == rounded_value:
            return pitch.Pitch(name)

    print("No note or rest could be linked to the value...")


def pitchidx_decode(target_idx: float, rest_cutoff: float = 0.25):
    # Does the opposite of pitch_encode: should output a pitch
    # This version assumes the note was already rounded
    for name, val in cof.items():
        if val == target_idx:
            return pitch.Pitch(name)

    print("No note or rest could be linked to the value...")


def to_vector_ts(score: stream.Score, hasParts=True):
    # It is assumed that the actual music is in some number of voices/parts
    # Per part, we open all notes, sort them, and output an array of vectors encoding the music
    # where each timestep corresponds to a 1/16th note
    note_endings = [
        note.offset + note.duration.quarterLength
        for note in score.recurse().getElementsByClass("Note")
    ]
    score_duration = max(note_endings)
    # Multiplied by 4, because we want it in 16th notes rather than quarter notes:
    output_ts = np.zeros(
        [int(score_duration * 4 + 1), len(score.getElementsByClass("Part"))]
    )
    for voice_idx, p in enumerate(score.getElementsByClass("Part")):
        notes = list(p.recurse().getElementsByClass("Note"))
        notes.sort(key=lambda n: n.offset)
        for note in notes:
            sixteenth_idx = int(note.offset * 4)
            while sixteenth_idx <= 4 * (note.offset + note.duration.quarterLength):
                output_ts[sixteenth_idx, voice_idx] = pitch_encode(note.pitch)
                sixteenth_idx += 1

    return output_ts


def from_vector_ts(data: np.ndarray):
    # This does the opposite of to_vector_ts:
    # It received an array representing the voices' notes,
    # and generates notes which it puts into parts and a score
    # then returns the score.

    # Note this can only be an approximation, because we cannot tell
    # The difference between repeated 16th notes and continuous notes.
    # We assume continuous notes.

    # For each part:
    score = stream.Score()

    for part_idx, p in enumerate(data.T):
        score.append(stream.Part())
        # Go over all the 16th indices, and extract notes:
        # When did a note start?
        offset_idx = 0
        tentative_note = 0  # assumed rest/no sound

        for sixteenth_idx, val in enumerate(p):

            if pitch_round(val) == tentative_note:
                continue
            # If they are not the same; register a note and
            # reset the search for the next note:
            notelength = sixteenth_idx - offset_idx
            if tentative_note == 0:
                score[part_idx].append(
                    note.Rest(quarterLength=notelength * 0.25, offset=offset_idx)
                )
            else:
                print()
                score[part_idx].append(
                    note.Note(
                        pitchidx_decode(tentative_note),
                        quarterLength=notelength * 0.25,
                        offset=offset_idx,
                    )
                )

            # reset:
            offset_idx = sixteenth_idx
            tentative_note = pitch_round(val)

    # After the for-loop, anything tentative is added anyway:
    notelength = len(p) - offset_idx
    if tentative_note == 0:
        score[part_idx].append(
            note.Rest(quarterLength=notelength * 0.25, offset=offset_idx)
        )
    else:
        score[part_idx].append(
            note.Note(
                pitchidx_decode(tentative_note),
                quarterLength=notelength * 0.25,
                offset=offset_idx,
            )
        )

    return score