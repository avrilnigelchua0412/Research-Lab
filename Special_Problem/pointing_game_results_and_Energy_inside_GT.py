import pandas as pd
from scipy.stats import spearmanr

EVAL_REGIMES = {
    'Levels 1–3': ['LEVEL_I', 'LEVEL_II', 'LEVEL_III'],
    'Levels 1–4': ['LEVEL_I', 'LEVEL_II', 'LEVEL_III', 'LEVEL_IV'],
    'Level 4 only': ['LEVEL_IV']
}

def summarize_energy(df):
    return {
        'mean': df['Energy inside GT'].mean(),
        'std': df['Energy inside GT'].std(),
        'median': df['Energy inside GT'].median(),
        'iqr': df['Energy inside GT'].quantile(0.75) - df['Energy inside GT'].quantile(0.25),
        'n': len(df)
    }

def summarize_pointing(df):
    return {
        'accuracy': df['Pointing game'].mean(),
        'hits': int(df['Pointing game'].sum()),
        'misses': int(len(df) - df['Pointing game'].sum()),
        'n': len(df)
    }

def compute_correlation(df):
    if df['Pointing game'].nunique() < 2:
        return {'rho': None, 'p': None}
    rho, p = spearmanr(df['Pointing game'], df['Energy inside GT'])
    return {'rho': rho, 'p': p}


def pipeline(data, eval_regimes, heads):
    for regime_name, levels in eval_regimes.items():
        print(f'\n===== {regime_name} =====')

        for head in heads:
            subset = data[
                (data['Level'].isin(levels)) &
                (data['Head'] == head)
            ]

            if subset.empty:
                print(f'{head}: no data')
                continue

            energy_stats = summarize_energy(subset)
            pointing_stats = summarize_pointing(subset)
            corr_stats = compute_correlation(subset)

            print(f'\nHead: {head}')
            print('Energy inside GT:', energy_stats)
            print('Pointing Game:', pointing_stats)

            if corr_stats['rho'] is not None:
                print(f"Spearman ρ = {corr_stats['rho']:.3f}, p = {corr_stats['p']:.4f}")
            else:
                print("Correlation not defined (constant Pointing Game outcome)")

        
if __name__ == "__main__":
    csvs = {
        "CL_yolov7tiny":'Special_Problem/CL_yolov7tiny_pointing_game_results and Energy inside GT.csv',
        "CL_yolov5nano":'Special_Problem/CL_yolov5nano_pointing_game_results and Energy inside GT.csv',
        "baseline_yolov7tiny":'Special_Problem/baseline_yolov7tiny_pointing_game_results and Energy inside GT.csv',
        "baseline_yolov5nano":'Special_Problem/baseline_yolov5nano_pointing_game_results and Energy inside GT.csv',
    }
    for csv in csvs:
        print(f'\n########## Processing {csv} ##########')
        data = pd.read_csv(
            csvs[csv],
            index_col=0
        )

        data['Pointing game'] = data['Pointing game'].map({'Hit': 1, 'Miss': 0})
        data = data[data['Energy inside GT'] > 0].copy()

        pipeline(
            data=data,
            eval_regimes=EVAL_REGIMES,
            heads=['P3', 'P4', 'P5']
        )
