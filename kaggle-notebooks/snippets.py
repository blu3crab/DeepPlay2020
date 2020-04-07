__author__ = "blu3crab"
__license__ = "Apache License 2.0"
__version__ = "0.0.1"

def plotCumulativeByDate(df, label, key):
    group = df.groupby('Date')[key].sum().reset_index()

    fig = px.line(group, x="Date", y=key,
                  title=label + " " + key + " Cases Over Time")

    fig.show()

    fig = px.line(df, x=df.index, y='Confirmed',
                  title='Confirmed' + " Cases Over Time")

    fig.show()

    # Set the width and height of the figure
    plt.figure(figsize=(14, 6))

    # Add title
    plt.title("Daily Confirmed for US")

    # Line chart showing daily global streams of 'Shape of You'
    # sns.lineplot(data=df['Confirmed'], label="Confirmed")

    sns.lineplot(data=df['Confirmed'], label="Confirmed")
    plt.show()

# Set the width and height of the figure
plt.figure(figsize=(12, 6))

# Add title
plt.title("US Confirmed")

# Bar chart showing average
# sns.barplot(x=ign_data.index, y=ign_data['Racing'])
sns.barplot(x=df.index, y=df['Confirmed'])

# Add label for vertical axis
plt.xlabel("Date")
plt.ylabel("Count")
plt.show()

def plotDailyByDate(df, label, key):
    group = df.groupby('Date')[key].nunique().reset_index()
    #group.head(5)
    fig = px.line(group, x="Date", y=key,
                  title=label + " " + key + " Cases Over Time")

    fig.show()