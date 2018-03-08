class Scales:
    'This class involves major and minor diatonic scales'
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "H"]

    def __init__(self, first_note, scale_type):
        self.first_note = first_note
        self.scale_type = scale_type

    def constructScale(self, scale_first_note, intervals):
        first_note_pos = Scales.notes.index(scale_first_note)
        out_scale=[Scales.notes[first_note_pos]]
        s=0
        for k in intervals:
            s=s+k
            out_scale.append(Scales.notes[(first_note_pos+s) % 12])
        return out_scale

    def displayScale(self):
        first_note_pos = Scales.notes.index(self.first_note)
        if self.scale_type=='dur': print('scale',self.first_note,"dur:",Scales.constructScale(self, self.first_note, [2,2,1,2,2,2,1]))
        elif self.scale_type=='moll': print('scale',self.first_note,"moll:", Scales.constructScale(self, self.first_note, [2,1,2,2,1,2,2]))
        else: print('Improper scale scale_type!')

    def returnScale(self):
        first_note_pos = Scales.notes.index(self.first_note)
        if self.scale_type=='dur': return Scales.constructScale(self, self.first_note, [2,2,1,2,2,2,1])
        elif self.scale_type=='moll': return Scales.constructScale(self, self.first_note, [2,1,2,2,1,2,2])
        else: return None

    def displayTriad(self):
        first_note_pos = Scales.notes.index(self.first_note)
        if self.scale_type == 'dur':
            print(self.first_note, self.scale_type, "chord:", Scales.constructScale(self, self.first_note, [4,3]))
        elif self.scale_type == 'moll':
            print(self.first_note, self.scale_type, "chord:", Scales.constructScale(self, self.first_note, [3,4]))
        else:
            print('Improper scale type!')
