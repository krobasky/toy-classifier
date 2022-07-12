import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split

print("+ This will take about 10 minutes with a power laptop, but requires a lot of memory for doing a groupby median on the gene expression")
print("+ Results in annotated geneset that is compatible with other datasets, like TCGA and TARGET")
print("""+ First run: 
  wget https://storage.googleapis.com/gtex_analysis_v8/rna_seq_data/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz -O download//GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz 
  wget https://storage.googleapis.com/gtex_analysis_v8/annotations/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt -O download/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt
""")

# maybe use this instead of the groupby to save memory and also use median instead?:
# !wget https://storage.googleapis.com/gtex_analysis_v8/rna_seq_data/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz -O data/dist/gtex/https://storage.googleapis.com/gtex_analysis_v8/rna_seq_data/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz

# Commit from https://github.com/cognoma/genes
# use this to make a compatible geneset annotation (thanks, Biobombe!)
genes_commit = 'ad9631bb4e77e2cdc5413b0d77cb8f7e93fc5bee'

def get_gene_df():
    url = 'https://raw.githubusercontent.com/cognoma/genes/{}/data/genes.tsv'.format(genes_commit)
    gene_df = pd.read_table(url)

    # Only consider protein-coding genes
    gene_df = (
        gene_df.query("gene_type == 'protein-coding'")
    )
    return gene_df

def get_old_to_new():
    # Load gene updater - old to new Entrez gene identifiers
    url = 'https://raw.githubusercontent.com/cognoma/genes/{}/data/updater.tsv'.format(genes_commit)
    updater_df = pd.read_table(url)
    old_to_new_entrez = dict(zip(updater_df.old_entrez_gene_id,
                                 updater_df.new_entrez_gene_id))
    return old_to_new_entrez


random.seed(1234)
os.makedirs("data/dist/gtex",exist_ok=True)

attr_path = 'data/dist/gtex/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt'
attr_df = pd.read_table(attr_path)

###### Process the gene expression data

# This involves updating Entrez gene ids, sorting and subsetting

print("+ Read gene expression - this takes a little while")
os.makedirs("data/gtex",exist_ok=True)
expr_path = 'data/dist/gtex/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz'
expr_df = pd.read_table(expr_path, sep='\t', skiprows=2, index_col=1)
print("+ Get GTEx gene mapping")
expr_gene_ids = (
    expr_df
    .loc[:, ['Name']]
    .reset_index()
    .drop_duplicates(subset='Description')
)
print("+ Perform inner merge gene df to get ensembl to entrez mapping")
gene_df=get_gene_df()
map_df = expr_gene_ids.merge(gene_df, how='inner', left_on='Description', right_on='symbol')
print("+ Save map, expression dataframes")
map_df.reset_index().to_feather("data/gtex/map.ftr") # if you run out of memory, this will load fast
## execute if needed -
# map_df = pd.read_feather("map.ftr")
# map_df = map_df.set_index('index')

# save expr so far
expr_df.reset_index().to_feather("data/gtex/expr.ftr")
## execute if needed -
#expr_df = pd.read_feather("expr.ftr")
#expr_df = expr_df.set_index('Description')

print("+ *Drop 'Name' column...")
expr_df=expr_df.drop(['Name'], axis='columns')
print("+ *Drop any rows with 'na's...")
expr_df=expr_df.dropna(axis='rows')
print("+ *groupby mean...") # xxx is this correct? Should we use median, or preprocessed file?
expr_df=expr_df.groupby(level=0).mean()
print("+ *reindex map...")
expr_df=expr_df.reindex(map_df.symbol)
symbol_to_entrez = dict(zip(map_df.symbol, map_df.entrez_gene_id))
print("+ *rename...")
expr_df=expr_df.rename(index=symbol_to_entrez)
##### get gene annotations
print("+ *rename again...")
expr_df=expr_df.rename(index=get_old_to_new())
print("+ *transpose...")
expr_df = expr_df.transpose()
print("+ *sort by row...")
expr_df = expr_df.sort_index(axis='rows')
print("+ *sort bye columns...")
expr_df = expr_df.sort_index(axis='columns')
print("+ write expr again")
expr_df.columns = expr_df.columns.astype(str)
expr_df.reset_index().to_feather("data/gtex/expr.ftr")

'''
expr_df = (expr_df
 .drop(['Name'], axis='columns')
 .dropna(axis='rows')
 .groupby(level=0).mean()
 .reindex(map_df.symbol)
 .rename(index=symbol_to_entrez)
 .rename(index=old_to_new_entrez)
 .transpose()
 .sort_index(axis='rows')
 .sort_index(axis='columns') 
)
'''
print("+ rename index")
expr_df.index.rename('sample_id', inplace=True)
print("+ write expr one more time")
expr_df.reset_index().to_feather("data/gtex/expr.ftr")

file="data/gtex/gene_ids.txt"
print(f"+ Write out gene ids in order ({file})")
with open(file,"a") as f:
    for col in expr_df.columns:
        f.write(f"{col}\n")

print("+ Write out superclass names")

print("++ Change attr tissue type names to something directory-friendly")
attr_df["SMTS"] = attr_df["SMTS"].str.strip()
attr_df["SMTS"] = attr_df["SMTS"].str.replace(' - ','-')
attr_df["SMTS"] = attr_df["SMTS"].str.replace(' \(','__').replace('\)','__')
attr_df["SMTS"] = attr_df["SMTS"].str.replace(' ','_')

class_names=set(attr_df["SMTS"])
print(f"++ Class names set: {class_names}")

attr_df[["SAMPID","SMTS"]].to_csv("data/gtex/sample_id-superclass_name.tsv", sep="\t", index=False, header=False)

print("+ Create dir structure for classes")
os.makedirs('data/gtex', exist_ok=True)
for cls in class_names:
  os.makedirs(f"data/gtex/{cls}",exist_ok=True)

print("+ Create a numpy for each row, write to data/gtex, separate out later")
import gzip
import numpy as np
for idx, nparray in enumerate(np.array(expr_df.iloc[:])):
    nparray=nparray.astype(np.float16)
    sample_id=expr_df.index[idx]
    cls=attr_df.loc[attr_df["SAMPID"]==sample_id, "SMTS"].iloc[0]
    with gzip.GzipFile(f"data/gtex/{cls}/{sample_id}.npy.gz", "w") as f:
        np.save(file=f, arr=nparray)

strat = attr_df.set_index('SAMPID').reindex(expr_df.index).SMTSD
tissuetype_count_df = (
    pd.DataFrame(strat.value_counts())
    .reset_index()
    .rename({'index': 'tissuetype', 'SMTSD': 'n ='}, axis='columns')
)

file = 'data/gtex/superclass-count.tsv'
print(f"+Write tissue type counts {file}")
tissuetype_count_df.to_csv(file, sep='\t', index=False)

print(f"+ tissue type counts: {tissuetype_count_df}")

print("""+ Reload each observation like such:
import gzip
import numpy as np
with gzip.GzipFile('data/gtex/<cls>/<sample_id>.npy.gz') as f:
    obs=np.load(f)
""")


###### Stratify Balanced Training and Testing Sets in GTEx Gene Expression
'''
# untested

train_df, test_df = train_test_split(expr_df,
                                     test_size=0.1,
                                     random_state=123,
                                     stratify=strat)
print(train_df.shape)
print(test_df.shape)

train_file = os.path.join('data', 'train_gtex_expression_matrix_processed.tsv.gz')
train_df.to_csv(train_file, sep='\t', compression='gzip', float_format='%.3g')

test_file = os.path.join('data', 'test_gtex_expression_matrix_processed.tsv.gz')
test_df.to_csv(test_file, sep='\t', compression='gzip', float_format='%.3g')

####### Sort genes based on median absolute deviation and output to file

# Determine most variably expressed genes and subset
mad_genes_df = pd.DataFrame(train_df.mad(axis=0).sort_values(ascending=False)).reset_index()
mad_genes_df.columns = ['gene_id', 'median_absolute_deviation']

file = os.path.join('data', 'gtex_mad_genes.tsv')
mad_genes_df.to_csv(file, sep='\t', index=False)
'''
