from sklearn.metrics import mean_absolute_percentage_error as mape
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot(results:dict, category_names:list, total_values:int):
    """
    Plotting the distribution of the train-test split for each split of the cross validation.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.colormaps['PRGn'](
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.invert_yaxis()
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)
        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        ax.bar_label(rects, label_type='center', color=text_color)
    ax.legend(ncols=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')
    ax.set_title("Cross Validation Splits")
    ax.set_xticks(np.arange(0,total_values,12))
    ax.grid(which='major', axis='x')

    return fig, ax


def create_splits(df:pd.DataFrame,n_splits:int=5,overlap:int=0,test_size:int=18)->list:
    """
    Creating n splits for cross validation. Returns a list with 'n_splits' dictionaries, each holding
    a training and a testing set in form of a dataframe.
    :df: The dataframe holding the complete data.
    :n_splits: The amount of train-test-splits.
    :overlap: The amount of overlap in months between each test set.
    :test_size: The size of the test set.
    """
    # remove all future rows with empty target variable to make sure that the last column in the dataframe holds the last known value
    df = df.dropna(subset=['sales_actuals_monthly__vehiclegroup01__orderintake'], axis=0)

    splits = []
    split_sizes = {}

    for split_no in range(1 , n_splits + 1):
        current = {}
        test_begin = (split_no * test_size - ((split_no-1) * overlap))
        current['train'] = df.iloc[:-test_begin]
        test_end  = len(current['train']) + test_size
        current['test'] = df.iloc[-test_begin:test_end]
        splits.append(current)
        split_sizes[f'split_{split_no}'] = [len(current['train']), len(current['test'])]

    # visualisation of splits
    split_names = ['train', 'test']
    plot(split_sizes, split_names, len(df))
    plt.show()

    return splits


def evaluate(results:list):
    """
    Expecting a list of dataframes, each containing a testset as well as a column in this set called 'prediction'.
    Visualizes and calculates metrics for each split as well as the overall performance.
    """
    mape_values = []
    for idx,result in enumerate(results):

        predictions = result['prediction']
        actual = result.rename(columns={'sales_actuals_monthly__vehiclegroup01__orderintake':'actual_orderintake'})['actual_orderintake']
        result['absolute percentage error'] = np.abs((actual - predictions) / actual)

        mape_score = mape(actual, predictions)
        mape_values.append(mape_score)

        predictions.plot(kind='line', legend=True, figsize=(8,6), title=f'Prediction for split_{idx+1}')
        actual.plot(kind='line', legend=True)
        plt.show()
        
        result['absolute percentage error'].plot(legend=True, kind='bar', figsize=(8,3), width=0.1)
        plt.axhline(y=mape_score, color='b')
        plt.show()

        print(f'Prediction mean: {predictions.mean()}')
        print(f'MAPE: {mape_score}')
        print('_________________________________________________________________________________________')

    print(f'Overall Result: {np.mean(mape_values)}')