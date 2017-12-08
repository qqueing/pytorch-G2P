

import torchtext.data as data
import os
import random

class KORDict(data.Dataset):
    def __init__(self, data_lines, g_field, p_field):
        fields = [('grapheme', g_field), ('phoneme', p_field)]
        examples = []  # maybe ignore '...-1' grapheme
        for line in data_lines:
            grapheme, phoneme = line.strip().split(maxsplit=1)
            examples.append(data.Example.fromlist([grapheme, phoneme],
                                                  fields))
        self.sort_key = lambda x: len(x.grapheme)
        super(KORDict, self).__init__(examples, fields)

    @classmethod
    def splits(cls, path,train,valid,test, g_field, p_field, seed=None):

        train_path = os.path.join(path, train)

        test_path = os.path.join(path, test)


        if seed is not None:
            random.seed(seed)
        with open(train_path,encoding='utf-8') as f:
            lines = f.readlines()
        random.shuffle(lines)
        train_lines, val_lines, test_lines = [], [], []

        if valid:
            valid_path = os.path.join(path, valid)
            if seed is not None:
                random.seed(seed)
            with open(valid_path,encoding='utf-8') as f:
                v_lines = f.readlines()
            random.shuffle(v_lines)
            for line in v_lines:
                val_lines.append(line)

        if test_path:
            test_path = os.path.join(path, test)
            if seed is not None:
                random.seed(seed)
            with open(test_path,encoding='utf-8') as f:
                v_lines = f.readlines()
            random.shuffle(v_lines)
            for line in v_lines:
                test_lines.append(line)


        for i, line in enumerate(lines):
            if i % 20 == 0 and not valid:
                val_lines.append(line)
            elif i % 20 < 3 and not test:
                test_lines.append(line)
            else:
                train_lines.append(line)
        train_data = cls(train_lines, g_field, p_field)
        val_data = cls(val_lines, g_field, p_field)
        test_data = cls(test_lines, g_field, p_field)
        return (train_data, val_data, test_data)

