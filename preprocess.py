import dask.dataframe as dd
import pandas as pd
from tqdm import tqdm
import os

def pre_xenium(folder):
    # transcripts = dd.read_parquet(folder + "transcripts.parquet")
    transcripts = dd.read_parquet(os.path.join(folder, "transcripts.parquet"))
    # gene_list = pd.read_csv(folder + "features.tsv", sep='\t', header=None)
    gene_list = pd.read_csv(os.path.join(folder, "features.tsv"), sep='\t', header=None)
    gene_list = gene_list[gene_list[2] == "Gene Expression"][1] 
    # cells = pd.read_csv(folder + "cells.csv")
    cells = pd.read_csv(os.path.join(folder, "cells.csv"))

    transcripts = transcripts.query('qv > 20')  # remove low quality transcripts
    transcripts = transcripts.query('feature_name.isin(@gene_list)', local_dict={"gene_list": gene_list})  # Remove the controlled code words
    transcripts = transcripts.query('cell_id != -1 and cell_id != "UNASSIGNED"')    # Remove the unassigned cells
    transcripts = transcripts.merge(cells[['cell_id', 'x_centroid', 'y_centroid']], on='cell_id', how='left')
    transcripts['x_local'] = transcripts['x_location'] - transcripts['x_centroid']
    transcripts['y_local'] = transcripts['y_location'] - transcripts['y_centroid']
    transcripts['distance'] = (transcripts['x_local']**2 + transcripts['y_local']**2)**0.5
    # transcripts.compute().to_parquet(folder + "transcripts_processed.parquet")
    transcripts.compute().to_parquet(os.path.join(folder, "transcripts_processed.parquet"))

    # split_path = folder + "MoleculesPerGene"
    split_path = os.path.join(folder, "MoleculesPerGene")
    if not os.path.exists(split_path):
        os.makedirs(split_path)

    for gene in tqdm(gene_list):
        gene_rows = transcripts[transcripts['feature_name'] == gene]
        gene_rows.compute().to_csv(os.path.join(split_path, f"{gene}.csv"), index=False)
