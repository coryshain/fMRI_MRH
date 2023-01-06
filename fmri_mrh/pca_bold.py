import os
import pickle
import numpy as np
import pandas as pd
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from fmri_mrh.util import *

BOLD_DIR = os.path.join(DATA_DIR, 'src', 'derivative', 'preprocessed_data')
OUT_DIR = os.path.join(PREPROCESSED_DIR, 'pca_bold')
N_COMPONENTS = 500

for subject in SUBJECTS:
    stderr('Subject: %s\n' % subject)
    path = os.path.join(DATA_DIR, subject)
    D = {}
    for _path in os.listdir(path):
        name = _path[:-4]
        stderr('  Text: %s\n' % name)
        _path = os.path.join(path, _path)
        f = h5py.File(_path)
        _D = np.nan_to_num(np.array(f['data']), 0.)
        _D = pd.DataFrame(_D, columns=['v%05d' % i for i in range(_D.shape[1])])
        _D['subject'] = subject
        _D['docid'] = name
        _D['time'] = np.arange(len(_D)) * 2 + 10  # 2s TR, first 10s of story is trimmed
        D[name] = _D

    D_train = []
    for name in D:
        if name not in HELD_OUT_TEXTS:
            D_train.append(D[name])
    D_train = pd.concat(D_train, axis=0)
    X = D_train[sorted([col for col in D_train if col.startswith('v')])]

    m = Pipeline([('scaler', StandardScaler()), ('pca', PCA(N_COMPONENTS))])
    m.fit(X)
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    with open(os.path.join(OUT_DIR, '%s_pca_model.obj' % subject), 'wb') as f:
        pickle.dump(m, f)
    stderr('  %s %% variance explained: %0.2f\n' % (subject, m['pca'].explained_variance_ratio_.sum() * 100))

    for name in D:
        _D = D[name]
        X = _D[sorted([col for col in _D if col.startswith('v')])]
        X = m.transform(X)
        X = pd.DataFrame(X, columns=['BOLD_PC%03d' % i for i in range(N_COMPONENTS)])
        out = pd.concat([_D[[col for col in _D if not col.startswith('v')]], X], axis=1)
        subject = str(out.subject.unique().squeeze())
        out.to_csv(os.path.join(OUT_DIR, '%s_%s_PC%d.csv' % (subject, name, N_COMPONENTS)), sep=' ', index=False)
    
