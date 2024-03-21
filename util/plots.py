import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def plot_adr_hist(df, color):
     x = df['adr'].clip(upper=400).round(0)
     mu = x.mean()
     sigma = x.std()
     num_bins = 80
     
     n, bins, patches = plt.hist(x,
                              num_bins, 
                              density = 1, 
                              color =color,
                              alpha = 0.7)

     y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
          np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
     
     plt.plot(bins, y, '--', color ='black')
     plt.xlabel('Average Daily Rate')
     plt.ylabel('Frequency')


def plot_room_rates(df,color):
    y = df.groupby('reserved_room_type')['adr'].mean()
    yerr = df.groupby('reserved_room_type')['adr'].std()
    x = y.index

    y.drop('P',inplace=True)
    yerr.drop('P',inplace=True)
    x = x.drop('P')

    fig, ax = plt.subplots()

    ax.errorbar(x, y, yerr, fmt='o', color=color, linewidth=2, capsize=6)

    ax.set_ylim(bottom=0)
    ax.set_xlabel('Room Type')
    ax.set_ylabel('Average Daily Rate')
    return fig, ax


def plot_reservations_roomtype(df_h1,df_h2):    
    # only plotting room types with at least 100 reservations
    y1 = df_h1.groupby('reserved_room_type').size()
    y1 = y1.loc[y1 > 100]
    y2 = df_h2.groupby('reserved_room_type').size()
    y2 = y2.loc[y2 > 100]
    x1 = y1.index
    x2 = y2.index

    # setting X-axis to be the same roomtypes for both hotels (even if the particular hotel has 0 reservations of a particular type)
    x12 = list(set(x1) | set(x2))
    x12.sort()
    
    # ordering the reservation counts accordingly to the previous step
    for ind in x12:
        if ind not in y1.index:
            y1[ind] = 0
        if ind not in y2.index:
            y2[ind] = 0
    y1 = y1.sort_index()
    y2 = y2.sort_index()

    opacity = 0.4
    bar_width = 0.35

    plt.xticks(range(len(x12)), x12)
    bar1 = plt.bar(np.arange(len(y1)) + bar_width, y1, bar_width, align='center', alpha=opacity, color='limegreen', label='City Hotel')
    bar2 = plt.bar(range(len(y2)), y2, bar_width, align='center', alpha=opacity, color='royalblue', label='Resort Hotel')

    # Add counts above the two bar graphs
    for rect in bar1 + bar2:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.0f}', ha='center', va='bottom',rotation=0)


    plt.legend()
    plt.xlabel('Room Type')
    plt.ylabel('Number of Reservations')
    plt.title('Reservations for each Room Type')
    plt.show()


def plot_reservations_country(df):
    countries = df['country'].value_counts()
    countries = countries.loc[countries > 200]

    y = countries.sort_values(ascending=False)
    x = countries.index


    fig, ax = plt.subplots(2,figsize=(10,10))
    ax[0].bar(x,y)
    ax[1].bar(x,y)
    #ax[0].xticks(x,rotation=90)
    ax[1].set_ylim(0,13000)
    ax[0].set_xlabel('Country')
    ax[0].set_xticklabels(x,rotation=90)
    ax[1].set_xticklabels(x,rotation=90)
    ax[1].set_xlabel('Country')
    ax[0].set_ylabel('Guests')
    ax[1].set_ylabel('Guests')
    ax[0].grid(linestyle='--',linewidth=0.5)
    ax[1].grid(linestyle='--',linewidth=0.5)

    ax[0].set_title('Reservations per Country')
    return fig, ax

def plot_cancellations_roomtype(df_h1,df_h2):
    # only plotting cancellation rates for a particular room type if at least 100 cancellations were made for the type
    df_h1_canceled = df_h1.groupby('reserved_room_type')['is_canceled'].sum()
    df_h1_canceled = df_h1_canceled.loc[df_h1_canceled > 100]
    df_h1_total = df_h1.groupby('reserved_room_type').size()
    df_h1_total = df_h1_total.loc[df_h1_total > 50]
    y1 = df_h1_canceled / df_h1_total
    y1 = y1.round(3)

    df_h2_canceled = df_h2.groupby('reserved_room_type')['is_canceled'].sum()
    df_h2_canceled = df_h2_canceled.loc[df_h2_canceled > 100]
    df_h2_total = df_h2.groupby('reserved_room_type').size()
    df_h2_total = df_h2_total.loc[df_h2_total > 50]
    y2 = df_h2_canceled / df_h2_total
    y2 = y2.round(3)
    x1 = y1.index
    x2 = y2.index

    # setting X-axis to be the same roomtypes for both hotels (even if the particular hotel has 0 reservations of a particular type)
    x12 = list(set(x1) | set(x2))
    x12.sort()
    
    # ordering the reservation counts accordingly to the previous step
    for ind in x12:
        if ind not in y1.index:
            y1[ind] = 0
        if ind not in y2.index:
            y2[ind] = 0
    y1 = y1.sort_index()
    y2 = y2.sort_index()

    opacity = 0.4
    bar_width = 0.35

    plt.xticks(range(len(x12)), x12)
    bar1 = plt.bar(np.arange(len(y1)) + bar_width, y1, bar_width, align='center', alpha=opacity, color='limegreen', label='City Hotel')
    bar2 = plt.bar(range(len(y2)), y2, bar_width, align='center', alpha=opacity, color='royalblue', label='Resort Hotel')


    plt.legend()
    plt.xlabel('Room Type')
    plt.ylabel('Cancellations')
    plt.title('Fraction of room cancellations per room type.')
    plt.grid(linestyle='--',linewidth=0.5)
    plt.show()