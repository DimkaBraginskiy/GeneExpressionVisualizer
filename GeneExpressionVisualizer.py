import gzip
import shutil
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt




def unpack(file: str):
    unpacked = file.replace('.gz', '')
    if not os.path.exists(unpacked):
        with gzip.open(file, 'rb') as f_in:
            with open(unpacked, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    return unpacked


def load_soft_data(filename: str):
    start_line = None
    end_line = None

    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if "!dataset_table_begin" in line:
                start_line = i + 1
            elif "!dataset_table_end" in line:
                end_line = i
                break

    df = pd.read_csv(
        filename,
        sep='\t',
        skiprows=start_line,
        nrows=(end_line - start_line)
    )

    df = df[~df.iloc[:, 0].astype(str).str.startswith(('#', '^'))]
    df = df.dropna(subset=['IDENTIFIER'])
    return df


def transform_to_long(df: pd.DataFrame):
    sample_cols = [col for col in df.columns if col.startswith('GSM')]

    df_long = pd.melt(
        df,
        id_vars=['IDENTIFIER'],
        value_vars=sample_cols,
        var_name='sample',
        value_name='expression'
    )

    df_long = df_long.rename(columns={'IDENTIFIER': 'gene'})

    def assign_group(sample_id):
        num = int(sample_id.replace('GSM', ''))
        if 114084 <= num <= 114088:
            return "non-smoker"
        elif 114078 <= num <= 114083:
            return "smoker"
        return "unknown"

    df_long['group'] = df_long['sample'].apply(assign_group)
    df_long['expression'] = pd.to_numeric(df_long['expression'], errors='coerce')
    return df_long.dropna(subset=['expression'])


def identify_top_genes(df_long: pd.DataFrame):
    means = df_long.groupby(['gene', 'group'])['expression'].mean().unstack()
    means['abs_diff'] = (means['smoker'] - means['non-smoker']).abs()
    top_10_stats = means.sort_values(by='abs_diff', ascending=False).head(10)
    top_10_stats['difference'] = top_10_stats['smoker'] - top_10_stats['non-smoker']
    return top_10_stats.reset_index()


def visualize_data(df_long, top_genes_list):
    df_plot = df_long[df_long['gene'].isin(top_genes_list)].copy()
    df_plot['gene'] = pd.Categorical(df_plot['gene'], categories=top_genes_list, ordered=True)

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_plot, x='gene', y='expression', hue='group')
    plt.xticks(rotation=45)
    plt.savefig('boxplot_top10_genes.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df_plot, x='gene', y='expression', hue='group', split=True, inner="quart")
    plt.xticks(rotation=45)
    plt.savefig('violinplot_top10_genes.png')
    plt.close()

    heatmap_data = df_plot.pivot_table(index='gene', columns='sample', values='expression')
    cols = [f"GSM{i}" for i in range(114084, 114089)] + [f"GSM{i}" for i in range(114078, 114084)]
    heatmap_data = heatmap_data[[c for c in cols if c in heatmap_data.columns]]
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, cmap='RdYlBu_r', fmt=".1f")
    plt.savefig('heatmap_top10_genes.png')
    plt.close()


def main():
    filename = unpack('GDS2490.soft.gz')
    df = load_soft_data(filename)
    df_long = transform_to_long(df)
    top_10_stats = identify_top_genes(df_long)

    print("=== GENE EXPRESSION ANALYSIS ===")
    print(f"Loaded data: {len(df_long['gene'].unique())} genes, 11 samples")
    print("\nTop 10 genes with greatest expression difference:\n")
    print(f"{'Gene':<15} {'Non-smoker':<12} {'Smoker':<12} {'Difference'}")
    for _, row in top_10_stats.iterrows():
        diff = f"{'+' if row['difference'] > 0 else ''}{row['difference']:.1f}"
        print(f"{row['gene']:<15} {row['non-smoker']:<12.1f} {row['smoker']:<12.1f} {diff}")

    visualize_data(df_long, top_10_stats['gene'].tolist())
    print("\nSaved plots:\n- boxplot_top10_genes.png\n- violinplot_top10_genes.png\n- heatmap_top10_genes.png")


if __name__ == '__main__':
    main()