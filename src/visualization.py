import matplotlib.pyplot as plt

def plot_ph_trajectories(df):
    """
    Plot pH evolution over time for each flask
    """

    plt.figure(figsize=(8, 5))

    for flask_id in df['Flask_ID'].unique():
        flask_df = df[df['Flask_ID'] == flask_id]
        plt.plot(
            flask_df['Time_days'],
            flask_df['pH'],
            marker='o',
            label=f'Flask {flask_id}'
        )

    plt.xlabel('Time (days)')
    plt.ylabel('pH')
    plt.title('Bio-leaching Digital Twin: pH Evolution')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
