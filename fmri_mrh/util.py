import sys
import os


DATA_DIR = os.path.join('..', 'data', 'fMRI_MRH')
PREPROCESSED_DIR = os.path.join(DATA_DIR, 'preprocessed')
VALIDATION_TEXTS = {
    'haveyoumethimyet',
}
TEST_TEXTS = {
    'wheretheressmoke'
}
HELD_OUT_TEXTS = VALIDATION_TEXTS | TEST_TEXTS
SUBJECTS = ['UTS%02d' % i for i in range(1, 9)]


def stderr(s):
    sys.stderr.write(s)
    sys.stderr.flush()

