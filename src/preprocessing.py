import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf


def group_and_join(df: pd.DataFrame, join_on: str, group_by: str) -> pd.DataFrame:
    """Performs a group operation on a dataframe and joins the resulting groups on a given column.

    Args:
        df (pd.DataFrame): Dataframe on which the operation should be performed
        join_on (str): Column to join on
        group_by (str): Column to group by

    Returns:
        pd.DataFrame: Transformed dataframe
    """
    columns = df.columns.tolist()
    columns.remove(join_on)
    columns.remove(group_by)

    dfs = []

    for country in df[group_by].unique():
        df_country = df[df[group_by] == country].copy()
        df_country.drop(columns=[group_by], inplace=True)
        df_country.set_index(join_on, inplace=True)
        df_country.rename({col: country.replace(" ", "_") + "_" + col for col in columns}, axis="columns", inplace=True)
        dfs.append(df_country)
    return pd.concat(dfs, axis=1)


def month_to_datestring(month: int) -> str:
    # Startpunkt Januar 2000
    m = (month - 1) % 12 + 1
    m_string = str(m) if m > 9 else '0' + str(m)
    y = (month-1) // 12
    y_string = str(y) if y > 9 else '0' + str(y)
    
    return f'{m_string}/{y_string}'


def convert_month_to_date(df: pd.DataFrame) -> pd.DataFrame:
    """Takes a dataframe with a month column and transforms it to a date column of type datetime.

    Args:
        df (pd.DataFrame): Dataframe to transform

    Returns:
        pd.DataFrame: Transformed dataframe
    """
    df['date'] = df['month'].apply(month_to_datestring)
    df['date'] = pd.to_datetime(df['date'], format='%m/%y')
    result = df.set_index('date').drop(['month'], axis=1)
    result.index = pd.DatetimeIndex(result.index).to_period('M')

    return result.sort_index()


def create_PowerBI_dataset():
    df = pd.read_csv("./data/kitCapstoneJan24_data.csv", sep=";")
    df = convert_month_to_date(df)

    # copy the index into a date col
    df['Date'] = df.index
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')  # Change the format accordingly
    ## Filter cols
    columns_to_keep = ['sales_actuals_monthly__vehiclegroup01__orderintake', 'countryname', "Date"]
    df = df[columns_to_keep]

    # Copy the specified column into an array
    sales_array = df['sales_actuals_monthly__vehiclegroup01__orderintake'].values

    # Add 12 values of 0 to the front
    sales_array = [0] * 12 + list(sales_array)

    # Remove the last 12 values
    sales_array = sales_array[:-12]

    # Create a new column with the name "last_year_order_intake"
    df['last_year_order_intake'] = sales_array
    df.to_csv("data_with_date_v5.csv")

def ad_test(dataset):
    dftest = adfuller(dataset, autolag='AIC')
    print("1. ADF : ", dftest[0])
    print("2. P-Value : ", dftest[1])
    print("3. Num Of Lags : ", dftest[2])
    print("4. Num Of Observations Used For ADF Regression:", dftest[3])
    print("5. Critical Values :")
    for key, val in dftest[4].items():
        print("\t", key, ": ", val)



def plot_line_graph(dataframe:pd.DataFrame, index_name:str, title:str, xlabel:str, ylabel:str, location_legend:str='', lower_xlim:int='', upper_xlim:int=''):
    fontweight = 'bold'
    fontsize_title = 16
    fontsize_label = 12
    
    dataframe[index_name].plot(figsize=(15, 4))
    plt.tick_params(axis='x', rotation=90)
            
    plt.title(title, fontsize=fontsize_title, fontweight=fontweight)
    plt.xlabel(xlabel, fontsize=fontsize_label, fontweight=fontweight)
    plt.ylabel(ylabel, fontsize=fontsize_label, fontweight=fontweight)
    #plt.legend(loc=location_legend)
    
    if lower_xlim != '' and upper_xlim != '':
        plt.xlim(lower_xlim, upper_xlim)
        
    plt.grid(True, alpha=0.6)
    plt.show()


def plot_decomposition_time_series(dataframe:pd.DataFrame, index_name:str):
    components_of_time_series = sm.tsa.seasonal_decompose(dataframe[index_name], model='additive')
    trend = components_of_time_series.trend
    seasonal = components_of_time_series.seasonal
    residuals = components_of_time_series.resid
    
    fig = components_of_time_series.plot()
    plt.show()
    
   # return trend, seasonal, residuals


def plot_partial_autocorrelation(dataframe:pd.DataFrame, index_name:str, title:str, lower_xlim:int=0, upper_xlim:int=100, xlabel:str='Lag', ylabel:str='Autokorrelation', savefig:str = 'Autokorrelation.png'):
    fontweight = 'bold'
    fontsize_title = 14
    fontsize_label = 11
    
    plot_pacf(dataframe[index_name], lags=100)
    plt.title(title, fontsize=fontsize_title, fontweight=fontweight)
    plt.xlabel(xlabel, fontsize=fontsize_label, fontweight=fontweight)
    plt.ylabel(ylabel, fontsize=fontsize_label, fontweight=fontweight)
    plt.grid(True, alpha=0.6)
    plt.xlim(lower_xlim, upper_xlim)
    plt.savefig(savefig)
    plt.show()


def outlier_replacement(feature):
    # Center the data so the mean is 0
    feature_outlier_centered = feature - feature.mean()
    # Calculate standard deviation
    std = feature.std()
    # Use the absolute value of each datapoint
    # to make it easier to find outliers
    outliers = np.abs(feature_outlier_centered) > (std * 3)
    # Replace outliers with the median value
    feature_outlier_fixed = feature_outlier_centered.copy() 
    feature_outlier_fixed[outliers] = np.nanmedian(feature_outlier_fixed)

    return feature_outlier_fixed


#https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/
def adfuller_test(time_series, signif:float=0.05, name:str='', verbose:bool=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    test = adfuller(time_series, autolag='AIC')
    
    #test[0] = test_statistic
    #test[1] = p_value
    #test[2] = number of chosen lags
    #test[3] = number of used observations
    #test[4] = critical values    
    
    output = {'test_statistic': round(test[0], 4), 
              'p_value': round(test[1], 4), 
              'num_lags': round(test[2], 4), 
              'num_observations': test[3]}
    
    p_value = output['p_value'] 

    # Print Summary
    print(f' Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["num_lags"]}')

    for key, value in test[4].items():
        print(f' Critical value {key} = {round(value, 3)}')

    # if probability is lower than level of significance (0.05)
    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")  


def plot_cross_correlation(dataframe1, dataframe2, start, ende, maxlags):
    fontweight = 'bold'
    fontsize_labels = 13
    fontsize_title = 13

    array = plt.xcorr(dataframe1, dataframe2, normed=True, maxlags=maxlags)
    plt.title(f'{dataframe1.name} über {dataframe2.name}', fontsize=fontsize_title, fontweight=fontweight)
    plt.xlabel('Lags (Woche)', fontsize=fontsize_labels, fontweight=fontweight)
    plt.ylabel('Korrelationskoeffizient (r)', fontsize=fontsize_labels, fontweight=fontweight)
    plt.savefig(f'Kreuzkorrelation {dataframe1.name} über {dataframe2.name}.png')
    plt.show()


def get_statistical_values_of_feature(dataframe, index_name, window_size=4):
    mean = dataframe[index_name].rolling(window=window_size, center=True).mean()
    max_value = dataframe[index_name].rolling(window=window_size, center=True).max()
    min_value = dataframe[index_name].rolling(window=window_size, center=True).min()
    std_value = dataframe[index_name].rolling(window=window_size, center=True).std()
    median = dataframe[index_name].rolling(window=window_size, center=True).median()
    x25 = dataframe[index_name].rolling(window=window_size, center=True).quantile(0.25)
    x75 = dataframe[index_name].rolling(window=window_size, center=True).quantile(0.75)
    
    max_value_df = pd.DataFrame(data=max_value)
    max_value_df = max_value_df.rename(columns={index_name: f'{index_name}_max_rolling_{window_size}'})
    
    min_value_df = pd.DataFrame(data=min_value)
    min_value_df = min_value_df.rename(columns={index_name: f'{index_name}_min_rolling_{window_size}'})
    
    std_value_df = pd.DataFrame(data=std_value)
    std_value_df = std_value_df.rename(columns={index_name: f'{index_name}_std_rolling_{window_size}'})
    
    median_df = pd.DataFrame(data=median)
    median_df = median_df.rename(columns={index_name: f'{index_name}_median_rolling_{window_size}'})
    
    x25_df = pd.DataFrame(data=x25)
    x25_df = x25_df.rename(columns={index_name: f'{index_name}_first_quantile_rolling_{window_size}'})
    
    x75_df = pd.DataFrame(data=x75)
    x75_df = x75_df.rename(columns={index_name: f'{index_name}_third_quantile_rolling_{window_size}'})
    
    df_statistical_values = pd.concat([max_value_df, min_value_df, std_value_df, median_df], axis=1)
    
    return df_statistical_values