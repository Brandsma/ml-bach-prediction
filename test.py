from music21 import *
import music

streamscore = music.load_midi('sample/unfin.mid')

streamscore.show('midi')
codedsinglepart = music.to_vector_ts(streamscore)

music.from_vector_ts(music.to_vector_ts(streamscore)).show('midi')


# singlepart = streamscore[3]
# singlepartscore = stream.Score()
# singlepartscore.append(singlepart)

# print(type(music.to_vector_ts(streamscore)))
# print(music.pitch_decode(music.pitch_encode(pitch.Pitch('F5'))))


# # Converting midi to stream
# b = corpus.parse('bwv66.6')
# bChords = b.chordify()
# #bChords.show()

# print(len(b.getElementsByClass(stream.Part)[0].getElementsByClass(stream.Measure)))

# streamscore = midi.translate.midiFilePathToStream('sample/unfin.mid')
# print([type(streamscore[i]) for i in range(len(streamscore))])
# streamscore.show('midi')
# for i in streamscore.recurse().getElementsByClass('Voice'):
#     #print(i.recurse()[0].offset)
#     i.show('midi')
# for n in streamscore.recurse().getElementsByClass('Note'):
#     print(n.offset)

# print(len(streamscore.getElementsByClass(stream.Part)[0].getElementsByClass(stream.Measure)))

# for thisChord in streamscore.recurse().getElementsByClass('Chord'):
#     print(thisChord.measureNumber)
# streamscore.show()



# Converting back to a midi file
#mf = midi.translate.streamToMidiFile(streamscore)
##mf.open("new-major-scale.mid", "wb")
#mf.write()
#mf.close()