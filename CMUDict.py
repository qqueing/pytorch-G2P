

import torchtext.data as data

class CMUDict(data.Dataset):
    def __init__(self, data_lines, g_field, p_field):
        fields = [('grapheme', g_field), ('phoneme', p_field)]
        examples = []  # maybe ignore '...-1' grapheme
        for line in data_lines:
            grapheme, phoneme = line.split(maxsplit=1)
            examples.append(data.Example.fromlist([grapheme, phoneme],
                                                  fields))
        self.sort_key = lambda x: len(x.grapheme)
        super(CMUDict, self).__init__(examples, fields)

    @classmethod
    def splits(cls, path, g_field, p_field, seed=None):
        import random

        if seed is not None:
            random.seed(seed)
        with open(path) as f:
            lines = f.readlines()
        random.shuffle(lines)
        train_lines, val_lines, test_lines = [], [], []
        for i, line in enumerate(lines):
            if i % 20 == 0:
                val_lines.append(line)
            elif i % 20 < 3:
                test_lines.append(line)
            else:
                train_lines.append(line)
        train_data = cls(train_lines, g_field, p_field)
        val_data = cls(val_lines, g_field, p_field)
        test_data = cls(test_lines, g_field, p_field)
        return (train_data, val_data, test_data)

