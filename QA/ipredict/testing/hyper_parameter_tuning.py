import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from scipy.stats import uniform
import numpy as np
import requests
from pathlib import Path
import sys
cwd = os.getcwd()
sys.path.append(cwd)
from ipredict.algorithm_impl.utils import save_model, read_data_df, init_logger
from ipredict.config.config import init_config
# Define lists for each hyperparameter
growth_val_list = ['linear']
seasonality_mode_val_list = ['multiplicative' ]
changepoint_prior_scale_list = uniform(0.0, 0.1).rvs(size=5)
holidays_prior_scale_list = [0.01]
interval_width_list = [0.6]
seasonality_prior_scale_list = [0.01]
yearly_seasonality = [True]
weekly_seasonality = [True]
daily_seasonality_list = [True, False]
mcmc_samples_list = [0]
class HyperParameterTuning:
    def __init__(self) -> None:
        self.logger=logging.getLogger(__name__)

    def train_model_test(self, table, analysis_prophet, growth_val='linear', 
                    seasonality_mode_val='additive',
                    changepoint_prior_scale=0.05, 
                    holidays_prior_scale_val=10.0,
                    interval_width=0.80, 
                    seasonality_prior_scale=10.0,
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality="auto", 
                    mcmc_samples=0):
        self.logger.debug("Inside train model test")
        prophet_parameters = {
            'growth': growth_val,
            'seasonality_mode': seasonality_mode_val,
            'changepoint_prior_scale': changepoint_prior_scale,
            'holidays_prior_scale': holidays_prior_scale_val,
            'interval_width': interval_width,
            'seasonality_prior_scale': seasonality_prior_scale,
            'yearly_seasonality': yearly_seasonality,
            'weekly_seasonality': weekly_seasonality,
            'daily_seasonality': daily_seasonality,
            'mcmc_samples': mcmc_samples
        }
        
        test_name = f"test_G-{growth_val}_SM-{seasonality_mode_val}_CPS-{changepoint_prior_scale}_HPS-{holidays_prior_scale_val}_IW-{interval_width}_SPS_{seasonality_prior_scale}-YS_{yearly_seasonality}-WS_{weekly_seasonality}-DS_{daily_seasonality}-MS_{mcmc_samples}"
        
        if analysis_prophet.shape[0] != 0:
            model = Prophet(**prophet_parameters)
            model.fit(analysis_prophet)
            save_model(model, table)
            return model, test_name
        else:
            return None, test_name

    def find_accuracy(self, file_path):
        self.logger.debug("Inside find accuracy")
        database = "cost_data"
        resampled_df = read_data_df(database, file_path)
        self.logger.debug(f"input data size---{resampled_df.shape[0]}")
        
        train_df, test_df = train_test_split(resampled_df, test_size=0.2, shuffle=False)
        # split_point = int(len(resampled_df) * 0.8)
        # train_df = resampled_df[:split_point]
        # test_df = resampled_df[split_point:]
        self.logger.info(f"train_df-->{train_df.shape[0]}")
        self.logger.info(f"test_df-->{test_df.shape[0]}")
        
        table = 'test'
        validation_df = pd.DataFrame()
        mse_list = []
        mae_list = []
        accuracy_list = []
        accuracy_upper_list = []
        accuracy_lower_list = []
        test_name_list = []
        
        for growth_value in growth_val_list:
            for seasonality_mode_value in seasonality_mode_val_list:
                for changepoint_prior_scale_value in changepoint_prior_scale_list:
                    for holidays_prior_scale_value in holidays_prior_scale_list:
                        for interval_width_value in interval_width_list:
                            for seasonality_prior_scale_value in seasonality_prior_scale_list:
                                for daily_seasonality_value in daily_seasonality_list:
                                    for mcmc_samples_value in mcmc_samples_list:
                                        model, test_name = self.train_model_test( 
                                            table, train_df,
                                            growth_val=growth_value,
                                            seasonality_mode_val=seasonality_mode_value,
                                            changepoint_prior_scale=changepoint_prior_scale_value,
                                            holidays_prior_scale_val=holidays_prior_scale_value,
                                            interval_width=interval_width_value,
                                            seasonality_prior_scale=seasonality_prior_scale_value,
                                            daily_seasonality=daily_seasonality_value,
                                            mcmc_samples=mcmc_samples_value
                                        )
                                        for df, name1 in zip([train_df, test_df], ["train", "test"]):
                                            forecast = model.predict(df)
                                            self.logger.info(forecast.shape[0])
                                            
                                            if 'y' in df.columns and 'yhat' in forecast.columns:
                                                mse = mean_squared_error(df['y'], forecast['yhat'])
                                                mae = mean_absolute_error(df['y'], forecast['yhat'])
                                                accuracy = (1 - (mae / df['y'].mean())) * 100
                                                
                                                if 'yhat_upper' in forecast.columns and 'yhat_lower' in forecast.columns:
                                                    mae_upper = mean_absolute_error(df['y'], forecast['yhat_upper'])
                                                    mae_lower = mean_absolute_error(df['y'], forecast['yhat_lower'])
                                                    accuracy_upper = (1 - (mae_upper / df['y'].mean())) * 100
                                                    accuracy_lower = (1 - (mae_lower / df['y'].mean())) * 100
                                                else:
                                                    self.logger.error("Columns 'yhat_upper' or 'yhat_lower' are missing in the data.")
                                                    accuracy_upper = None
                                                    accuracy_lower = None

                                                self.logger.info(f"Mean Squared Error (MSE): {mse}")
                                                self.logger.info(f"Mean Absolute Error (MAE): {mae}")
                                                self.logger.info(f"Accuracy: {accuracy}%")
                                                if accuracy_upper is not None and accuracy_lower is not None:
                                                    self.logger.info(f"Accuracy (Upper): {accuracy_upper}%")
                                                    self.logger.info(f"Accuracy (Lower): {accuracy_lower}%")
                                                
                                            else:
                                                self.logger.error("Columns 'y' or 'yhat' are missing in the data.")

                                            mse_list.append(mse)
                                            mae_list.append(mae)
                                            accuracy_list.append(accuracy)
                                            accuracy_upper_list.append(accuracy_upper)
                                            accuracy_lower_list.append(accuracy_lower)
                                            test_name1 = f"{test_name}_{name1}"
                                            self.logger.info(test_name1)
                                            test_name_list.append(test_name1)
                                            sys.stdout.flush()
                                            
        validation_df['test_name'] = test_name_list
        validation_df['mse'] = mse_list
        validation_df['mae'] = mae_list
        validation_df['accuracy'] = accuracy_list
        validation_df['accuracy_upper'] = accuracy_upper_list
        validation_df['accuracy_lower'] = accuracy_lower_list
        folder_path = "./test_report"
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        path = os.path.join(folder_path, f"{file_path}_accuracy_val.csv")
        validation_df.to_csv(path, index=False)
        
        return True


    def readingtables1(self, address, port_number, table_name):
        questdb_url = f"http://{address}:{port_number}/exec"
        self.logger.debug(f"questdb_url----->{questdb_url}")
        sys.stdout.flush()
        show_tables_query = "SHOW TABLES"
        response = requests.get(questdb_url, params={"query": show_tables_query})

        try:
            data = response.json()
            dataset = data['dataset']
            table_names = [row[0] for row in dataset if row[0].startswith(f"{table_name}")]
            self.logger.info(table_names)
            self.logger.info(table_names)
            sys.stdout.flush()
            return table_names

        except requests.exceptions.JSONDecodeError as e:
            self.logger.exception(f"Error decoding JSON: {e}")
            self.logger.debug("Response text:", response.text)
            sys.stdout.flush()

    def accuracy_model(self, table, growth='linear', 
                    seasonality_mode='additive',
                    changepoint_prior_scale=0.05, 
                    holidays_prior_scale=10.0,
                    interval_width=0.80, 
                    seasonality_prior_scale=10.0,
                    daily_seasonality="auto", 
                    mcmc_samples=0):
        self.logger.debug("Inside accuracy model")
        # Initialize the Prophet model with given parameters
        model = Prophet(
            growth = growth,
            seasonality_mode = seasonality_mode,
            changepoint_prior_scale = changepoint_prior_scale,
            holidays_prior_scale = holidays_prior_scale,
            interval_width = interval_width,
            seasonality_prior_scale = seasonality_prior_scale,
            daily_seasonality = daily_seasonality,
            mcmc_samples = mcmc_samples
        )

        test_name = f"test_G-{growth}_SM-{seasonality_mode}_CPS-{changepoint_prior_scale}_HPS-{holidays_prior_scale}_IW-{interval_width}_SPS_{seasonality_prior_scale}-DS_{daily_seasonality}-MS_{mcmc_samples}"
        
        database = "cost_data"
        full_df = read_data_df(database, table)
        train_df, test_df = train_test_split(full_df, test_size=0.2, shuffle=False)
        self.logger.info(f"train_df-->{train_df.shape[0]}")
        self.logger.info(f"test_df-->{test_df.shape[0]}")
        # Fit the model to the data
        model.fit(train_df) 
        response={}
        for df,name1 in zip([train_df,test_df],["train","test"]):
            test_name1=f"{test_name}_{name1}"
            forecast = model.predict(df)
            mse = mean_squared_error(df['y'], forecast['yhat'])
            mae = mean_absolute_error(df['y'], forecast['yhat'])
            accuracy = (1 - (mae / df['y'].mean())) * 100

            response.update({
            f"{name1}_mse": mse,
            f"{name1}_mae": mae,
            f"{name1}_accuracy": accuracy
            })

        self.logger.info(response)
        sys.stdout.flush()
        return response

    def DataAnalysis(self, address, port_number, table_name):
        questdb_url = f"http://{address}:{port_number}/exec"
        self.logger.debug(f"questdb_url----->{questdb_url}")
        sys.stdout.flush()
        show_tables_query = "SHOW TABLES"
        response = requests.get(questdb_url, params={"query": show_tables_query})

        try:
            data = response.json()
            dataset = data['dataset']
            table_names = [row[0] for row in dataset if row[0].startswith(f"{table_name}")]
            self.logger.info(table_names)
            self.logger.info(len(table_names))

            # List to collect data for Excel
            excel_data = []

            for table in table_names:
                # Remove the prefix 'cost_data_' from the table name
                original_table_name = table  # Keep the original table name for logging
                table = table.replace("cost_data_", "")
                df = read_data_df(database='cost_data', table=table, start_date=None, end_date=None)
                print(f"df.columns-->{df.columns}")
                # Log the shape of the DataFrame
                self.logger.info(f"df.shape-->{df.shape}")

                # Ensure that 'ds' column exists
                if 'ds' in df.columns:
                    # Convert 'ds' to datetime
                    df['ds'] = pd.to_datetime(df['ds'])

                    # Calculate total number of instances (rows)
                    total_instances = len(df['ds'])

                    # Calculate the time differences between consecutive timestamps
                    time_diffs = df['ds'].diff().dt.total_seconds().dropna()

                    # Calculate the frequency of time differences
                    if not time_diffs.empty:
                        time_diff_counts = time_diffs.value_counts()

                        # Get the first mode (most common time gap)
                        first_mode_gap = time_diff_counts.index[0] if len(time_diff_counts) > 0 else np.nan
                        second_mode_gap = time_diff_counts.index[1] if len(time_diff_counts) > 1 else np.nan
                        third_mode_gap = time_diff_counts.index[2] if len(time_diff_counts) > 2 else np.nan
                        fourth_mode_gap = time_diff_counts.index[3] if len(time_diff_counts) > 3 else np.nan

                        # Find the maximum and minimum gaps
                        max_gap = time_diffs.max()
                        min_gap = time_diffs.min()

                        # Calculate the expected number of instances
                        start_time = df['ds'].min()  # First timestamp
                        end_time = df['ds'].max()    # Last timestamp
                        total_time_span = (end_time - start_time).total_seconds()  # Time span in seconds

                        if first_mode_gap > 0:  # Ensure the mode is valid and greater than 0
                            expected_instances = total_time_span / first_mode_gap
                        else:
                            expected_instances = np.nan  # Set to NaN if the first mode is invalid

                        # Generate the expected timestamp range
                        expected_timestamps = pd.date_range(start=start_time, end=end_time, freq=pd.to_timedelta(first_mode_gap, unit='s'))

                        # Identify missing timestamps
                        missing_timestamps = expected_timestamps.difference(df['ds'])
                        num_missing_timestamps = len(missing_timestamps)

                        # Identify instances with gaps different than the first mode
                        instances_with_different_gap = (time_diffs != first_mode_gap).sum()

                        # Append data to excel_data list
                        excel_data.append({
                            'Table Name': original_table_name,
                            'Total Instances': total_instances,
                            'First Mode Gap (s)': first_mode_gap,
                            'Second Mode Gap (s)': second_mode_gap if not np.isnan(second_mode_gap) else 'NaN',
                            'third_mode_gap (s)': third_mode_gap if not np.isnan(third_mode_gap) else 'NaN',
                            'fourth_mode_gap (s)': fourth_mode_gap if not np.isnan(fourth_mode_gap) else 'NaN',
                            'Max Gap (s)': max_gap,
                            'Min Gap (s)': min_gap,
                            # 'Expected Instances': expected_instances,
                            # 'Missing Timestamps': num_missing_timestamps,
                            'Instances with Different Gap': instances_with_different_gap,
                            # 'List of Missing Dates': missing_timestamps.tolist()  # Store as list
                        })

                        # Log the calculated values for this table
                        self.logger.info(f"Table: {original_table_name}, First Mode: {first_mode_gap}, Second Mode: {second_mode_gap}, "
                                         f"third_mode_gap: {third_mode_gap}, fourth_mode_gap: {fourth_mode_gap}"
                                        f"Max Gap: {max_gap}, Min Gap: {min_gap}, Expected Instances: {expected_instances}, "
                                        f"Missing Timestamps: {num_missing_timestamps}, Instances with Different Gap: {instances_with_different_gap}")
                    else:
                        self.logger.info(f"No time differences found for table '{original_table_name}'.")
                
                else:
                    self.logger.warning(f"'ds' column not found in DataFrame for table '{original_table_name}'.")

            # Create a DataFrame from the collected data
            analysis_df = pd.DataFrame(excel_data)

            folder_path = "./test_report"
        
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            path = os.path.join(folder_path, f"data_analysis2.csv")
            analysis_df.to_csv(path, index=False)

            sys.stdout.flush()
            return table_names

        except requests.exceptions.JSONDecodeError as e:
            self.logger.exception(f"Error decoding JSON: {e}")
            self.logger.debug("Response text:", response.text)
            sys.stdout.flush()

