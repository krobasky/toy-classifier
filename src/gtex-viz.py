import gzip
import numpy as np
import pandas as pd
import os
import pathlib
from sklearn.decomposition import PCA

#TYPE="" # implicitly mean
#TYPE="frame"
#TYPE="median"
TYPE="max"

ext=""
if TYPE != "":
    ext=f".{TYPE}"

# xxx can only be 2
CLASS_NAMES=["Colon","Heart"]
#CLASS_NAMES=["Prostate","Pituitary"]
#CLASS_NAMES=["Stomach","Spleen"]
#CLASS_NAMES=["Brain","Skin"]
USE_PYPLOT=False
NUM_PCS=3
MAX_SAMPS=200000

# read data
def read_from_npys():
    df_orig=pd.DataFrame()
    class_name_set= set()
    for cls in list(os.walk(f"data/gtex{ext}"))[0][1]:
        if cls not in CLASS_NAMES:
            continue
        class_name_set.add(cls)
        samps=next(os.walk(f"data/gtex{ext}/{cls}"))[2]
        for i,samp in enumerate(samps):
            with gzip.GzipFile(f'data/gtex{ext}/{cls}/{samp}') as f:
                counts=np.load(f)
            df_orig[samp]=[cls] + list(counts)
            if i >= MAX_SAMPS:
                break

    df=df_orig.copy()
    df=df.transpose()
    return df

def read_from_raw():
    attr_path = 'data/dist/gtex/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt'
    attr_df = pd.read_table(attr_path)

    expr_df = pd.read_feather("data/gtex/expr.ftr")
    expr_df.reset_index(drop=True)
    expr_df=expr_df.set_index('sample_id')
    expr_df.index.name="SAMPID"
    expr_df.reset_index(inplace=True)

    expr_df=pd.merge(expr_df,attr_df[["SAMPID","SMTS"]])

    expr_df=expr_df.set_index('SAMPID')
    cols = expr_df.columns
    cols =cols[-1:].append(cols[:-1])
    expr_df=expr_df[cols]
    expr_df =expr_df[(expr_df['SMTS']==CLASS_NAMES[0]) | (expr_df['SMTS']==CLASS_NAMES[1])]

    return expr_df

if TYPE == "frame":
    df=read_from_raw()
else:
    df=read_from_npys()


# compute PCs
pca = PCA()
components = pca.fit_transform(df.iloc[:,1:15])
labels = {
    str(i): f"PC {i+1} ({var:.1f}%)"
    for i, var in enumerate(pca.explained_variance_ratio_ * 100)
}

# plot & save
if USE_PYPLOT:
    for idx, cls in enumerate(class_name_set):
        # xxx df.iloc[df[0]==cls, 0] = idx
        df[df['SMTS']=="Adipose Tissue"]

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(NUM_PCS, NUM_PCS, figsize=(150,100))
    for x in range(NUM_PCS):
        for y in range(NUM_PCS):
            if x == y:
                continue
            # xxxscatter = ax[x][y].scatter(components[:,x], components[:,y], c=df[0], cmap='RdYlBu', marker="2", alpha=0.70)
            scatter = ax[x][y].scatter(components[:,x], components[:,y], c=df['SMTS'], cmap='RdYlBu', marker="2", alpha=0.70)
            legend  = ax[x][y].legend(*scatter.legend_elements(),
                                      loc="upper right", title="Classes")
            ax[x][y].add_artist(legend)

            ax[x][y].plot([1,2])
            ax[x][y].set_title(f"PC{x+1} vs PC{y+1}")
            ax[x][y].set_xlabel(f"PC{x+1}")
            ax[x][y].set_ylabel(f"PC{y+1}")
            ax[x][y].legend()

    # set the spacing between subplots
    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.4, 
                        hspace=0.4)

    plt.show()

else:
    import plotly.express as px
    if TYPE=="frame":
        fig = px.scatter_matrix(
            components,
            labels=labels,
            dimensions=range(NUM_PCS),
            color=df['SMTS'],
            opacity=0.50
        )
    else:
        fig = px.scatter_matrix(
            components,
            labels=labels,
            dimensions=range(NUM_PCS),
            color=df[0],
            opacity=0.50
        )

    fig.update_traces(diagonal_visible=False, marker={'size':4})
    #fig.show()

    if not os.path.exists("images"):
        os.mkdir("images")

    file_name = "_".join(CLASS_NAMES)
    fig.write_image(f"images/{file_name}{ext}.png")

