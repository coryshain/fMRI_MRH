import os
import pandas as pd
import argparse
from fmri_mrh.util import *

GPT_DIR = os.path.join(PREPROCESSED_DIR, 'pca_gpt')
BOLD_DIR = os.path.join(PREPROCESSED_DIR, 'pca_bold')
OUT_DIR = DATA_DIR

if __name__ == '__main__':

    argparser = argparse.ArgumentParser('''Combine GPT and BOLD tables into single tables for regression.''')
    argparser.add_argument('-n', '--n_components_gpt', default=500, help='Number of GPT2 principal components to use.')
    argparser.add_argument('-N', '--n_components_bold', default=500, help='Number of BOLD principal components to use.')
    argparser.add_argument('-a', '--all_subjects', action='store_true', help='Place all subjects into a single table. Otherwise, produces separate table sets by subject.')
    args = argparser.parse_args()

    n_components_gpt = args.n_components_gpt
    n_components_bold = args.n_components_bold
    all_subjects = args.all_subjects

    stderr('Processing stimulus tables...\n')
    stimuli = {}
    for path in [x for x in os.listdir(GPT_DIR) if x.endswith('_PC%03d.csv' % n_components_gpt)]:
        name = path.split('_')[0]
        stderr('  Text: %s\n' % name)
        df = pd.read_csv(os.path.join(GPT_DIR, path), sep=' ')
        stimuli[name] = df
    stderr('\n')

    stderr('Processing response tables...\n')
    all_X_train = []
    all_X_dev = []
    all_X_test = []
    all_Y_train = []
    all_Y_dev = []
    all_Y_test = []
    for subject in SUBJECTS:
        stderr('  Subject: %s\n' % subject)
        X_train = []
        X_dev = []
        X_test = []
        Y_train = []
        Y_dev = []
        Y_test = []
        for path in [x for x in os.listdir(BOLD_DIR) if (x.startswith(subject) and x.endswith('_PC%d.csv' % n_components_bold))]:
            name = path.split('_')[1]
            stderr('    Text: %s\n' % name)
            df = pd.read_csv(os.path.join(BOLD_DIR, path), sep=' ')
            if name in VALIDATION_TEXTS:
                X_dev.append(stimuli[name])
                Y_dev.append(df)
            elif name in TEST_TEXTS:
                X_test.append(stimuli[name])
                Y_test.append(df)
            else:
                X_train.append(stimuli[name])
                Y_train.append(df)
        X_train = pd.concat(X_train, axis=0)
        X_dev = pd.concat(X_dev, axis=0)
        X_test = pd.concat(X_test, axis=0)
        Y_train = pd.concat(Y_train, axis=0)
        Y_dev = pd.concat(Y_dev, axis=0)
        Y_test = pd.concat(Y_test, axis=0)

        Y_train['subject'] = subject
        Y_dev['subject'] = subject
        Y_test['subject'] = subject

        if all_subjects:
            all_X_train.append(X_train)
            all_X_dev.append(X_dev)
            all_X_test.append(X_test)
            all_Y_train.append(Y_train)
            all_Y_dev.append(Y_dev)
            all_Y_test.append(Y_test)
        else:
            X_train.to_csv(os.path.join(OUT_DIR, '%s_PC%03d_GPT_train.csv' % (subject, n_components_gpt)), sep=' ', index=False)
            X_dev.to_csv(os.path.join(OUT_DIR, '%s_PC%03d_GPT_dev.csv' % (subject, n_components_gpt)), sep=' ', index=False)
            X_test.to_csv(os.path.join(OUT_DIR, '%s_PC%03d_GPT_test.csv' % (subject, n_components_gpt)), sep=' ', index=False)
            Y_train.to_csv(os.path.join(OUT_DIR, '%s_PC%03d_BOLD_train.csv' % (subject, n_components_bold)), sep=' ', index=False)
            Y_dev.to_csv(os.path.join(OUT_DIR, '%s_PC%03d_BOLD_dev.csv' % (subject, n_components_bold)), sep=' ', index=False)
            Y_test.to_csv(os.path.join(OUT_DIR, '%s_PC%03d_BOLD_test.csv' % (subject, n_components_bold)), sep=' ', index=False)

    if all_subjects:
        all_X_train = pd.concat(all_X_train)
        all_X_dev = pd.concat(all_X_dev)
        all_X_test = pd.concat(all_X_test)
        all_Y_train = pd.concat(all_Y_train)
        all_Y_dev = pd.concat(all_Y_dev)
        all_Y_test = pd.concat(all_Y_test)

        all_X_train.to_csv(os.path.join(OUT_DIR, 'ALLSUBJ_PC%03d_GPT_train.csv' % (n_components_gpt)), sep=' ', index=False)
        all_X_dev.to_csv(os.path.join(OUT_DIR, 'ALLSUBJ_PC%03d_GPT_dev.csv' % (n_components_gpt)), sep=' ', index=False)
        all_X_test.to_csv(os.path.join(OUT_DIR, 'ALLSUBJ_PC%03d_GPT_test.csv' % (n_components_gpt)), sep=' ', index=False)
        all_Y_train.to_csv(os.path.join(OUT_DIR, 'ALLSUBJ_PC%03d_BOLD_train.csv' % (n_components_bold)), sep=' ', index=False)
        all_Y_dev.to_csv(os.path.join(OUT_DIR, 'ALLSUBJ_PC%03d_BOLD_dev.csv' % (n_components_bold)), sep=' ', index=False)
        all_Y_test.to_csv(os.path.join(OUT_DIR, 'ALLSUBJ_PC%03d_BOLD_test.csv' % (n_components_bold)), sep=' ', index=False)

