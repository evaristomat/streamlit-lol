import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# Dark Theme for Matplotlib
#plt.style.use('dark_background')

background_color = '#0E1117'
params = {
    "axes.labelcolor": "white",
    "axes.edgecolor": "white",
    "axes.facecolor": background_color,
    "xtick.color": "white",
    "ytick.color": "white",
    "text.color": "white",
    "figure.facecolor": background_color,
    "grid.color": "gray",
    "grid.linestyle": "--",
}
plt.rcParams.update(params)

# Function to load data
@st.cache_data
def load_data():
    df = pd.read_csv("bets/bets.csv")
    return df

def process_data(df):
    df = df.copy()  # Create a copy of the dataframe
    df['ROI'] = df['ROI'].str.rstrip('%').astype('float')
    df = df[df['status'].isin(['win', 'loss'])]
    df = df[df['ROI'] >= 10]
    df['profit'] = df.apply(lambda row: row['odds'] - 1 if row['status'] == 'win' else -1, axis=1)
    df['cumulative_profit'] = df['profit'].cumsum()
    df['bet_group'] = df['bet_line'].str.split().str[0]
    return df

def bankroll_plot(df):
    window_size = 10
    df['moving_average'] = df['cumulative_profit'].rolling(window=window_size).mean()
    plt.figure(figsize=(12, 7))
    ax = sns.lineplot(data=df, x=df.index, y='cumulative_profit', marker="o", label='Cumulative Profit')
    sns.lineplot(data=df, x=df.index, y='moving_average', color='red', label=f'{window_size}-bet Moving Average')
    plt.title('Evolution of Bankroll Over Bets with Moving Average')
    plt.ylabel('Cumulative Profit')
    plt.xlabel('Bet Sequence')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(ax.get_figure())


def odds_plot(df):
    bins = [1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3]
    df['odds_bin'] = pd.cut(df['odds'], bins)
    grouped = df.groupby(['odds_bin', 'status']).size().unstack().fillna(0)
    mid_points = [(b + bins[i+1]) / 2 for i, b in enumerate(bins[:-1])]
    theoretical_probs = [1 / point for point in mid_points]
    grouped['total'] = grouped['win'] + grouped['loss']
    grouped['win_ratio'] = grouped['win'] / grouped['total']
    grouped['edge'] = grouped['win_ratio'] - theoretical_probs
    plt.figure(figsize=(10, 7))
    ax = sns.barplot(x=grouped.index, y=grouped['edge'], palette="coolwarm", errorbar=None)
    ax.axhline(0, color='black', linestyle='--')
    for i, value in enumerate(grouped['edge']):
        ax.text(i, value if value > 0 else 0, f'{value:.2%}', ha='center', va='bottom' if value > 0 else 'top', fontsize=10)
    plt.title('Edge Over Theoretical Probabilities by Odds Range')
    plt.ylabel('Edge (Win Rate - Theoretical Probability)')
    plt.xlabel('Odds Range')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(ax.get_figure())

def bet_groups_plot(df):
    grouped = df.groupby(['bet_group', 'status']).size().unstack().fillna(0)
    grouped['total'] = grouped['win'] + grouped['loss']
    grouped['win_ratio'] = grouped['win'] / grouped['total']
    grouped['loss_ratio'] = grouped['loss'] / grouped['total']
    grouped['bet_group'] = grouped.index

    plt.figure(figsize=(10, 7))
    ax = sns.barplot(data=grouped.melt(id_vars=['bet_group', 'win_ratio', 'loss_ratio'], value_vars=['win', 'loss'], 
                                       var_name='status', value_name='count'),
                    x='bet_group', y='count', hue='status', hue_order=['loss', 'win'], palette={"win": "green", "loss": "red"})
    for i, bar in enumerate(ax.patches):
        if i >= len(grouped):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2.0,
                        height / 2,
                        f'{grouped["win_ratio"].iloc[i - len(grouped)]:.2%}',
                        ha='center', va='center',
                        color='white', fontsize=10)
        else:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2.0,
                        height / 2,
                        f'{grouped["loss_ratio"].iloc[i]:.2%}',
                        ha='center', va='center',
                        color='white', fontsize=10)

    plt.title('Distribution of Bet Groups with Wins and Losses')
    plt.ylabel('Number of Bets')
    plt.xlabel('Bet Group')
    plt.legend(title='Status', loc='upper right')
    plt.tight_layout()
    st.pyplot(ax.get_figure())

def profit_plot(df):
    profit_by_group = df.groupby('bet_group')['profit'].sum()
    plt.figure(figsize=(10, 7))
    profit_plot = sns.barplot(x=profit_by_group.index, y=profit_by_group.values, palette="viridis")
    for i, value in enumerate(profit_by_group.values):
        profit_plot.text(i, value if value > 0 else 0, f'{value:.2f}',
                         ha='center', va='bottom' if value > 0 else 'top', fontsize=10)
    plt.title('Profit by Bet Group')
    plt.ylabel('Total Profit')
    plt.xlabel('Bet Group')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(profit_plot.get_figure())

def scatter_plot(df):
    # Color-coding based on status
    colors = df['status'].map({'win': 'green', 'loss': 'red'})

    ax = plt.figure(figsize=(12, 6))
    plt.scatter(df['fair_odds'], df['odds'], alpha=0.5, c=colors)
    plt.title('Fair Odds vs Actual Odds with Win/Loss Overlay')
    plt.xlabel('Fair Odds')
    plt.ylabel('Actual Odds')
    plt.plot([1, 3], [1, 3], color='blue')  # y=x line for reference
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Win', markersize=10, markerfacecolor='green'),
                    Line2D([0], [0], marker='o', color='w', label='Loss', markersize=10, markerfacecolor='red')]
    plt.legend(handles=legend_elements)
    st.pyplot(ax.get_figure())

# Main execution
def main():
    df = load_data()
    processed_df = process_data(df)
    
    st.title('Betting Statistics Dashboard - ROI 10%')
    
    if st.button('Show Analysis'):
        bankroll_plot(processed_df)
        odds_plot(processed_df)
        bet_groups_plot(processed_df)
        profit_plot(processed_df)
        scatter_plot(processed_df)

# Call the main execution
if __name__ == "__main__":
    main()