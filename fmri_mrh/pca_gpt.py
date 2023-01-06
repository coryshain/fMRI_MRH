import re
import os
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from fmri_mrh.util import *

TEXT_DIR = os.path.join(PREPROCESSED_DIR, 'texts')
OUT_DIR = os.path.join(PREPROCESSED_DIR, 'pca_gpt')
N_COMPONENTS = 500
GPT_COL = re.compile('D(\d\d\d)(_L(\d\d))?')


def rename(x, level=0):
    if GPT_COL.match(x):
        return x + '_L%02d' % level
    return x


df = {}
for path in os.listdir(DATA_DIR):
    name, level = path.split('_')
    level = int(level[1:-4])
    path = os.path.join(DATA_DIR, path)
    _df = pd.read_csv(path, sep=' ').rename(lambda x, level=level: rename(x, level=level), axis=1)
    gpt_cols = [col for col in _df.columns if GPT_COL.match(col)]
    if name not in df:
        df[name] = _df
    else:
        df[name] = pd.concat([df[name], _df[gpt_cols]], axis=1)

df_train = pd.concat([df[x] for x in df if x not in HELD_OUT_TEXTS], axis=0)

gpt_cols = [col for col in df_train.columns if GPT_COL.match(col)]
m = Pipeline([('scaler', StandardScaler()), ('pca', PCA(N_COMPONENTS))])
X = df_train[gpt_cols]
m.fit(X)
stderr('%% variance explained: %0.2f\n' % m['pca'].explained_variance_ratio_.sum())

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

outpath = os.path.join(OUT_DIR, 'pca_model.obj')
with open(outpath, 'wb') as f:
    pickle.dump(m, f)

for name in df:
    _df = df[name]
    X = _df[gpt_cols]
    X = m.transform(X)
    _df = _df[[x for x in _df.columns if x not in gpt_cols]]
    X = pd.DataFrame(X, columns=['PC%03d' % i for i in range(N_COMPONENTS)])
    X = pd.concat([_df, X], axis=1)
    X.to_csv(os.path.join(TEXT_DIR, '%s_PC%d.csv' % (name, N_COMPONENTS)), sep=' ', index=False)

