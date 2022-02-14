# This is a simplified version of Cases2Beds with generic parameters.
# DELPHI Group
# Contact aajoshi@andrew.cmu.edu if you have any questions

# Input Files:
# Opt 1a. distribution_example.csv:
# A set of values to split the cases by age group (no other functionality)
# Opt 1b. cases_example.csv:
# The number of cases per day - can be output from covidcast.

# OR  2. Cases_Split.csv: Case inputs broken down by demographics
# 3.  input_parameters_generic.csv: Sample parameters for C2B

# Output Files:
# 1. C2B_output.csv: The Cases2Beds Outputs as a csv file
# 2. Plot of C2B


from datetime import date
from datetime import datetime
from datetime import timedelta
from scipy.stats import expon
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import datetime
import getopt
import glob
import math
import numpy as np
import os
import os.path
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import psutil
import random
import requests
import sys
import scipy.stats as st
import scipy.stats as stats
import statsmodels.api as sm
import time


def o_d(p1, p2, p3):
    """ Creating the offset dictionary with the three parameters:
    1. Fraction of COVID Admissions that tested positive on
    Admission Date (or one day before/after)
    2. Fraction of COVID Admissions that tested positive more than
     one day AFTER being admitted
    3. Mean number of days from testing date to admission date,
    among admissions that happened more than one day after testing
    Outputs a dictionary
    These three parameters can be modified in the file,
    "distribution_example.csv"
    """
    negative_mean = 4  # hardcoded
    offset_dict = {}
    off_bott = p2
    off_top = 1-off_bott-p1
    offset_dict[-1] = round(p1/3, 10)
    offset_dict[0] = round(p1/3, 10)
    offset_dict[1] = round(p1/3, 10)
    exp_sum = 0
    exp_sum_2 = 0
    if p2 != 0:
        for i in range(0, 9):
            exp_sum += round(expon.pdf(i, 0, negative_mean), 10)
        for j in range(-10, -1):
            offset_dict[j] = round(round(expon.pdf(
                (j * -1)-2, 0, negative_mean), 10) * off_bott * round(
                (1 / exp_sum), 10), 10)
    if p3 != 0:
        for i in range(0, 29):
            exp_sum_2 += round(expon.pdf(i, 0, p3), 10)
        for j in range(2, 31):
            offset_dict[j] = round(round(expon.pdf(
                j-2, 0, p3), 10) * off_top * round((1 / exp_sum_2), 10), 10)
    return offset_dict


def d_d(p1, p2):
    """ Creating the duration dictionary with the two parameters:
    1. Mean Length of Stay
    2. Fraction of Hospitalizations that last 0 days [Optional]
    Outputs a dictionary
    These parameters can be modified in the file, "distribution_example.csv"
    """
    duration_dict = {}
    duration_dict[0] = p2
    exp_sum = 0
    if p1 > 0:
        for j in range(0, 40):
            exp_sum = round(exp_sum + expon.pdf(j, 0, p1), 10)
        for j in range(1, 41):
            duration_dict[j] = round(round(expon.pdf(
                j-1, 0, p1), 10) * (1-p2) * round((1/exp_sum), 10), 10)
    return duration_dict


def gen_distribution_params(age, race, gender, parameters):
    """
    A method with inputs of a specific age group, race, gender,
    and pre-set parameters list. This method returns either the
    parameters row that's an exact match to the queried age group,
    race, and gender, or extrapolates it from
    other rows in the parameters list.

    It outputs a data frame row of parameters.
    """
    delta = 0.0000000000001
    col_list = ["hosp_rate", "pos_one_day",
                "frac_less_1", "frac_0"]
    if [age, race, gender].count("unspecified") >= 2:
        age_l = parameters[parameters["age"] == age]
        race_l = age_l[age_l["race"] == race]
        new_params = race_l[race_l["gender"] == gender].reset_index(
            drop=True).iloc[:, 3:]
    else:  # at least one of the following is unspecified
        new_params = pd.DataFrame()
        age_params = parameters[(parameters["age"] == age) & (
            parameters["race"] == "unspecified") & (
            parameters["gender"] == "unspecified")].reset_index(drop=True)
        if race != "unspecified" and gender == "unspecified":
            race_val = parameters[parameters["race"] == race].reset_index(
                drop=True)
            race_list = parameters[parameters["race"] != "unspecified"]
            race_average = race_list.mean()
            for index, col in enumerate(race_val.columns[3:]):
                rel_gain = (age_params[col].values[0] *
                            race_val[col].values[0]) / [race_average[col].mean(
                            ) + delta]
                if col in (col_list and rel_gain >= 1):
                    rel_gain = 0.99
                new_params.loc[0, col] = round(rel_gain[0], 10)
        elif gender != "unspecified" and race == "unspecified":
            gen_val = parameters[parameters["gender"] == gender].reset_index(
                drop=True)
            gen_average = parameters.query(
                "gender!='unspecified'").mean()
            for index, col in enumerate(gen_val.columns[3:]):
                rel_gain_2 = age_params[col].values[0]
                rel_gain_num = (rel_gain_2*gen_val[col].values[0])
                rel_gain = rel_gain_num/[gen_average[col].mean() + delta]
                if col in col_list and rel_gain >= 1:
                    rel_gain = 0.99
                new_params.loc[0, col] = round(rel_gain[0], 10)
        else:  # both are specified
            gen_val = parameters[parameters["gender"] == gender].reset_index(
                drop=True)
            gen_average = parameters.query("gender!='unspecified'").mean()
            race_val = parameters[parameters["race"] == race].reset_index(
                drop=True)
            race_l = parameters[parameters["race"] != "unspecified"]
            race_average = race_l.mean()
            for index, col in enumerate(gen_val.columns[3:]):
                rel_gain = age_params[col].values[0] * (
                                gen_val[col].values[0] /
                                [gen_average[col].mean() + delta] *
                                race_val[col].values[0] /
                                [race_average[col].mean() + delta]) ** 0.5
                if col in col_list and rel_gain >= 1:
                    rel_gain = 0.99
                new_params.loc[0, col] = round(rel_gain[0], 10)
    return new_params


def gen_distribution(age, race, gender, params):
    """
    A method with inputs of a specific age group, race, gender,
    and pre-set parameters list, which are passed to the
    gen_distribution_params function. With the
    respective parameters for this specific age group, race, and gender,
    we create the offset_dict and duration_dict.

    The method returns the parameters, the offset dictionary, and
    the duration dictionary for a specific row.
    """
    new_params = gen_distribution_params(age, race, gender, params)
    offset_dict = o_d(new_params["pos_one_day"].values[0],
                      new_params["frac_less_1"].values[0],
                      new_params["mean_greater_1"].values[0])
    duration_dict = d_d(new_params["mean"].values[0],
                        new_params["frac_0"].values[0])
    return(duration_dict, offset_dict, new_params)


def gen_mean_var(params_in, cases_in):
    """
    The inputs are the parameters and the cases.
    This method creates the mean and variance
    dictionaries by iterating over every row
    in the cases file.
    For more details on the methodology, see:
    https://delphi.cmu.edu/blog/2021/03/10/
    cases2beds-a-case-study-in-actionable-intelligence/

    The outputs are two dictionaries that
    correspond to the day and the mean beds for that day
    or the variance as calculated by C2B.
    """
    mean_d = {}
    var_d = {}
    for i, case in cases_in.iterrows():
        if i >= 0:
            OF = [0] * 41
            age_group = case["Age Group (optional but highly desirable)"]
            race = case["Race/Ethnicity (Black, White," +
                        " Asian, Hispanic, or unspecified)"]
            gen = case["Gender (F, M, or unspecified)"]
            cases = case["Number of such Cases (optional, defaults to 1)"]
            date = pd.to_datetime(case["Specimen Testing Date (mandatory)"])
            if not pd.isnull(date):
                D, O, params = gen_distribution(
                    age_group, race, gen, params_in)
                hosp_factor = float(params["hosp_rate"])
                for o in range(min(list(O.keys())), max(list(O.keys()))+1):
                    curr_o = O[o]
                    for d in range(min(list(D.keys())), max(list(D.keys()))+1):
                        t_val = D[d]*curr_o*hosp_factor
                        for k in range(o, o+d+1):
                            if (k < 31):
                                OF[k+10] += t_val
                for j in range(-10, 31):
                    curr_date = date + timedelta(days=j)
                    p = OF[j+10]
                    mean_d[curr_date] = mean_d.get(curr_date, 0) + (p*cases)
                    var_d[curr_date] = var_d.get(curr_date, 0) + (p*cases*(1-p))
    return mean_d, var_d


def C2B(params_in, cases_in):
    """
    The inputs are the parameters and the cases.
    This method takes the generated mean and variance dictionaries to
    calculate the values of the (5, 25, 50, 75, and 95) quantiles.
    The method outputs a data frame with these quantiles, as well as the
    mean and variance data frames.
    """
    mean_d, var_d = gen_mean_var(params_in, cases_in)
    mean_df = pd.DataFrame.from_dict(mean_d, orient='index')
    var_df = pd.DataFrame.from_dict(var_d, orient='index')
    plotting_df = pd.DataFrame()
    plotting_df["dates"] = list(mean_d.keys())
    plotting_df["mean"] = list(mean_d.values())
    plotting_df["var"] = list(var_d.values())
    plotting_df["stdev"] = [math.sqrt(x) for x in var_d.values()]
    plotting_df["Q5"] = (-1.645*plotting_df["stdev"]) + plotting_df["mean"]
    plotting_df["Q25"] = (-0.675*plotting_df["stdev"]) + plotting_df["mean"]
    plotting_df["Q75"] = (0.675*plotting_df["stdev"]) + plotting_df["mean"]
    plotting_df["Q95"] = (1.645*plotting_df["stdev"]) + plotting_df["mean"]
    plotting_df["Q5"][plotting_df["Q5"] < 0] = 0
    plotting_df["Q25"][plotting_df["Q25"] < 0] = 0
    return plotting_df, mean_df, var_df


def ae_plot(plotting_df):
    """
    This method takes the data frame with the
    quantiles and returns a Plotly figure.
    """
    x = plotting_df["dates"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=plotting_df["Q5"],
                             name="Q5", fill=None, mode='lines',
                             line_color='lightblue'))
    fig.add_trace(go.Scatter(x=x, y=plotting_df["Q25"],
                             showlegend=False,  fill='tonexty', mode='lines',
                             line_color='lightblue'))
    fig.add_trace(go.Scatter(x=x, y=plotting_df["Q25"],
                             name="Q25", fill=None, mode='lines',
                             line_color='blue'))
    fig.add_trace(go.Scatter(x=x, y=plotting_df["mean"],
                             showlegend=False,  fill='tonexty',
                             mode='lines', line_color='blue'))
    fig.add_trace(go.Scatter(x=x, y=plotting_df["mean"],
                             name="Q50",  fill=None, mode='lines',
                             line_color='black'))
    fig.add_trace(go.Scatter(x=x, y=plotting_df["Q75"],
                             showlegend=False, fill='tonexty', mode='lines',
                             line_color='blue'))
    fig.add_trace(go.Scatter(x=x, y=plotting_df["Q75"],
                             name="Q75", fill=None, mode='lines',
                             line_color='blue'))
    fig.add_trace(go.Scatter(x=x, y=plotting_df["Q95"],
                             showlegend=False, fill='tonexty',
                             mode='lines', line_color='lightblue'))
    fig.add_trace(go.Scatter(x=x, y=plotting_df["Q95"],
                             name="Q95", fill=None, mode='lines',
                             line_color='lightblue'))
    fig.update_layout(title_text='Cases2Beds Outputs',
                      xaxis_title="Dates", yaxis_title="Beds Needed")
    return fig


def C2B_outputs_w_input_params(cases, dem_breakdown):
    """
    This method replicates the "Easy Input" tab on the Excel worksheet.
    If the user inputs case numbers and the demographics they are using, then
    this method splits the total cases by the demographic breakdowns.
    Currently, only age is considered. This method creates a "Cases_Split.csv"
    file, which is then used with the rest of C2B.
    """
    cases["date"] = pd.to_datetime(cases["date"])
    age_list = []
    for index, row in cases[["cases", "date"]].iterrows():
        left_over = row["cases"]
        for age_gr in ["0-9", "10-19", "20-29", "30-39",
                       "40-49", "50-59", "60-69", "70-79", "80+",
                       "unspecified"]:
            age_dict = {}
            p_change = dem_breakdown.loc[:, age_gr].mean()
            age_dict["Specimen Testing Date" +
                     " (mandatory)"] = row["date"].strftime(
                '%m/%d/%Y')
            cases_used = round(
                        row["cases"] * p_change)
            left_over = left_over - cases_used
            if age_gr == "unspecified":
                cases_used += left_over
            age_dict["Number of such Cases" +
                     " (optional, defaults to 1)"] = cases_used
            age_dict["Age Group (optional but highly desirable)"] = age_gr
            age_dict["Race/Ethnicity (Black, White," +
                     " Asian, Hispanic, or unspecified)"] = "unspecified"
            age_dict["Gender (F, M, or unspecified)"] = "unspecified"
            age_list.append(age_dict)
    age_df = pd.DataFrame(age_list).fillna("unspecified")
    age_df.to_csv("Cases_Split.csv", date_format='%m/%d/%Y')
    return(age_df)


def gen_plot_C2B(age_df, in_params):
    """
    This method creates the plot for Cases2Beds.
    The inputs are the cases data frame from the "Cases (the input)" tab
    and the parameters from the assumptions tab in the spreadsheet.
    The output is the output to C2B.
    """
    plot_df, mean, var = C2B(in_params, age_df)
    plot_df = plot_df.sort_values(by="dates").reset_index(drop=True)
    plot_df.to_csv("C2B_output.csv")
    fig = ae_plot(plot_df)
    fig.show()
    # fig.write_image("graphs/C2B.png")
    return plot_df


def main(argv):
    """
    This is the main method.
    There are two options to recreate the "Cases (the input)" tab.
    F1 and F2 correspond to .csv files that have the cases by day
    and the demographic breakdown, respectively ("Easy Input" tab).
    Alternatively, you can use -c to pass a file that has the cases split.

    -ip are the parameters for C2B that match the
    "Assumptions" tab in the worksheet.
    """
    cases = ""
    dem_breakdown = ""
    cases_split = ""
    in_params = ""
    cases_split_frame = pd.DataFrame()
    try:
        opts, args = getopt.getopt(argv, "hc:s:d:i:")
    except getopt.GetoptError:
        print('C2B_basic.py -c <cases> -d <dem_breakdown> -s ' +
              '<cases_split>  -i <input_params>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('C2B_basic.py -c <cases> -d <dem_breakdown> -s' +
                  ' <cases_split> -i <input_params>')
            sys.exit()
        elif opt in ("-c"):
            cases = arg
        elif opt in ("-d"):
            dem_breakdown = arg
        elif opt in ("-s"):
            cases_split = arg
        elif opt in ("-i"):
            in_params = arg
    if cases != "" and dem_breakdown != "":
        cases = pd.read_csv(cases)
        dem_breakdown = pd.read_csv(dem_breakdown, index_col=0).reset_index()
        if abs(1-dem_breakdown.sum(axis=1).values[0]) > 1e-5:
            print("The age distribution " +
                  "doesn't sum to 1. Err:", 1-dem_breakdown.sum(
                    axis=1).values)
            sys.exit(2)
        _ = C2B_outputs_w_input_params(cases, dem_breakdown)
    if cases_split_frame.empty:
        if cases_split != "":
            cases_split_frame = pd.read_csv(cases_split,
                                            index_col=0, header=0).dropna(
                                            how="all")
        else:
            cases_split_frame = pd.read_csv("Cases_Split.csv",
                                            index_col=0, header=0).dropna(
                                            how="all")
            cases_split_frame["Specimen Testing Date " +
                              "(mandatory)"] = pd.to_datetime(
                                cases_split_frame["Specimen " +
                                                  "Testing Date (mandatory)"])
    cases_split_frame["Number of such Cases " +
                      "(optional, defaults to 1)"] = cases_split_frame[
                      "Number of such Cases" +
                      " (optional, defaults to 1)"].replace(
                np.nan, 1)
    cases_split_frame["Age Group (optional but" +
                      " highly desirable)"] = cases_split_frame[
                      "Age Group (optional but" +
                      " highly desirable)"].replace(
                np.nan, "unspecified")
    cases_split_frame["Race/Ethnicity " +
                      "(Black, White, " +
                      "Asian, Hispanic, or" +
                      " unspecified)"] = cases_split_frame[
                      "Race/Ethnicity " +
                      "(Black, White, " +
                      "Asian, Hispanic, " +
                      "or unspecified)"].replace(
                np.nan, "unspecified")
    cases_split_frame["Gender (F, M, " +
                      "or unspecified)"] = cases_split_frame[
                                                             "Gender (F, M, " +
                                                             "or unspecif" +
                                                             "ied)"].replace(
                                        np.nan, "unspecified")
    if in_params == "":
        in_params = pd.read_csv("input_parameters_generic.csv")
    else:
        in_params = pd.read_csv(in_params)
    _ = gen_plot_C2B(cases_split_frame, in_params)


if __name__ == "__main__":
    main(sys.argv[1:])
