import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import numpy as np
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
    try:
        df = pd.read_csv("bets/bets.csv")
        df = df[df['status'].isin(['win', 'loss'])]
        data = map_league_names(df, 'league')
        return data
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return pd.DataFrame()

def process_data(df, min_roi, max_roi):
    df = df.copy()  # Create a copy of the dataframe
    df['ROI'] = df['ROI'].str.rstrip('%').astype('float')
    df = df[(df['ROI'] >= min_roi) & (df['ROI'] <= max_roi)]  # Filter ROI based on user input
    df['profit'] = df.apply(lambda row: row['odds'] - 1
    if row['status'] == 'win' else -1, axis=1)
    df['cumulative_profit'] = df['profit'].expanding().sum()
    df['bet_group'] = df['bet_line'].str.split().str[0]
    return df

def map_league_names(df: pd.DataFrame, league_column: str) -> pd.DataFrame:
    league_name_mapping = {
        "LOL - World Champs Play-In": "Worlds",
        "LOL - Worlds Qualifying Series": "Worlds",
        "2023 World Championship Play-In": "Worlds",
        "2023 World Championship": "Worlds",
        "LOL - World Champs": "Worlds",
        "World Championship": "Worlds"  # This one remains unchanged
    }

    df['league'] = df[league_column].map(lambda x: league_name_mapping.get(x, x))
    return df

def bankroll_plot(df):
    window_size = 10
    
    # Check if there's enough data to compute the moving average
    if len(df) < window_size:
        st.warning(f"Not enough data to compute a {window_size}-bet moving average. Plotting original data without moving average.")
    else:
        df['moving_average'] = df['cumulative_profit'].rolling(window=window_size).mean()

    plt.figure(figsize=(12, 7))
    ax = sns.lineplot(data=df, x=df.index, y='cumulative_profit', marker="o", label='Cumulative Profit')
    
    # Plot the moving average only if it's available
    if 'moving_average' in df.columns:
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
    grouped = df.groupby(['odds_bin', 'status'], observed=False).size().unstack().fillna(0)
    
    # Handle potentially missing columns
    if 'win' not in grouped:
        grouped['win'] = 0
    if 'loss' not in grouped:
        grouped['loss'] = 0
    
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
    
    # Handle potentially missing columns
    if 'win' not in grouped:
        grouped['win'] = 0
    if 'loss' not in grouped:
        grouped['loss'] = 0
    
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
    plt.figure(figsize=(10, 7))
    
    # Filter out the 'FD' bet_type when the bet_group is not 'first_dragon'
    df_filtered = df[~((df['bet_group'] != 'first_dragon') & (df['bet_type'] == 'FD'))]

    # Create a barplot with hue for bet_type
    profit_plot = sns.barplot(x='bet_group', y='profit', hue='bet_type', data=df_filtered, estimator=sum, palette="viridis", errorbar=None)

    # Annotate each bar with the profit value
    for p in profit_plot.patches:
        profit_plot.annotate(f'{p.get_height():.2f}', 
                             (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='center', 
                             xytext=(0, 10),
                             textcoords='offset points',
                             fontsize=10)

    plt.title('Profit by Bet Group')
    plt.ylabel('Total Profit')
    plt.xlabel('Bet Group')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend(title='Bet Type', loc='upper left')
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


def display_summary(df, roi_value):
    """
    Display a summary of the dataframe for the selected ROI value.
    
    Parameters:
    - df: The dataframe to extract summary from.
    - roi_value: The selected ROI value.
    """

    # Filter the dataframe
    filtered_df = df[df['ROI'] >= roi_value]

    # Calculate profit or loss for each bet and sum it
    filtered_df['profit'] = np.where(filtered_df['status'] == 'win', 
                                     (filtered_df['odds'] - 1), 
                                     -1)
    total_profit = filtered_df['profit'].sum()

    # Calculate the other summary statistics
    total_bets = len(filtered_df)
    wins = filtered_df[filtered_df['status'] == 'win'].shape[0]
    losses = total_bets - wins
    win_rate = wins / total_bets if total_bets != 0 else 0  # avoid division by zero
    avg_odd = filtered_df['odds'].mean()

    # Display the summary in columns
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    col1.metric(label="ROI Chosen", value=f"{roi_value}%")
    col2.metric(label="Total Bets", value=f"{total_bets}")
    col3.metric(label="Wins", value=f"{wins}")
    col4.metric(label="Losses", value=f"{losses}")
    col5.metric(label="Win Rate", value=f"{win_rate*100:.0f}%")  # Display full win rate
    col6.metric(label="Average Odd", value=f"{avg_odd:.2f}")
    col7.metric(label="Total Profit", value=f"{total_profit:.2f}U")


def main():
    try:
        df = load_data()

        st.title('Betting Statistics Dashboard')

        # Extract unique leagues and sort them
        available_leagues = sorted(df['league'].unique())

        # Streamlit dropdown for selecting a league
        selected_league = st.selectbox('Select a league:', ['All Leagues'] + available_leagues)

        # Filter the dataframe based on the selected league
        if selected_league != 'All Leagues':
            df = df[df['league'] == selected_league]

        # Extract the minimum and maximum ROI from the dataframe
        min_available_roi = df['ROI'].str.rstrip('%').astype('float').min()
        max_available_roi = df['ROI'].str.rstrip('%').astype('float').max()

        # Check if min and max ROI are the same
        if min_available_roi == max_available_roi:
            st.write(f"There's only one game in the {selected_league} league with an ROI of {min_available_roi}%. Adjust your filters or select another league.")
            return  # Exit the function

        # Extract the minimum and maximum ROI from the dataframe
        min_available_roi = df['ROI'].str.rstrip('%').astype('float').min()
        max_available_roi = df['ROI'].str.rstrip('%').astype('float').max()

        # Set a slider for selecting a minimum ROI
        chosen_roi = st.slider('Choose Minimum ROI (%)', int(min_available_roi), int(max_available_roi), int(min_available_roi))

        # Filter data based on the chosen ROI
        processed_df = process_data(df, chosen_roi, max_available_roi)

        # Display the summary below the ROI slider
        display_summary(processed_df, chosen_roi)

        # Check if the dataframe is empty
        if processed_df.empty:
            st.write(f"No data available for the chosen ROI of {chosen_roi}% or higher.")
            return  # Exit the function

        # Display plots
        bankroll_plot(processed_df)
        odds_plot(processed_df)
        bet_groups_plot(processed_df)
        profit_plot(processed_df)
        scatter_plot(processed_df)

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# Call the main execution
if __name__ == "__main__":
    main()
