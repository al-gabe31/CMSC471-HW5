from numpy import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from scipy import stats
import csv

# Set Global Constants Here
DECIMAL_PLACE = 6
RANDOM_INITIALIZATION = 0
LEARNING_RATE = 0.2
CONVERGENCE = pow(10, -5)

# Settings for plot customization
plt.style.use('fivethirtyeight') # My favorite plot theme <3
LINE_WIDTH = 2 # How thick each lines are

# Returns a tuple of all scaled means (except mpg)
# Index:
#   0: cyl
#   1: disp
#   2: hp
#   3: drat
#   4: wt
#   5: qsec
#   6: vs
#   7: am
#   8: gear
#   9: carb
def get_scaled_values(csv_file):
    # This is where all data will be stored
    mpg = []
    cyl = []
    disp = []
    hp = []
    drat = []
    wt = []
    qsec = []
    vs = []
    am = []
    gear = []
    carb = []

    # Read all data from the CSV file
    with open(csv_file, newline='') as csvfile:
        line = csv.reader(csvfile, delimiter=',')

        line_num = 0
        for row in line:
            if line_num > 0: # We ignore the first line since that's just the header
                # Takes all values from the csv and puts them into their respective array
                mpg.append(float(row[0]))
                cyl.append(float(row[1]))
                disp.append(float(row[2]))
                hp.append(float(row[3]))
                drat.append(float(row[4]))
                wt.append(float(row[5]))
                qsec.append(float(row[6]))
                vs.append(float(row[7]))
                am.append(float(row[8]))
                gear.append(float(row[9]))
                carb.append(float(row[10]))
            line_num += 1
    
    # Stores all the mean values for each data
    MEANS = {
        'mpg': mean(mpg),
        'cyl': mean(cyl),
        'disp': mean(disp),
        'hp': mean(hp),
        'drat': mean(drat),
        'wt': mean(wt),
        'qsec': mean(qsec),
        'vs': mean(vs),
        'am': mean(am),
        'gear': mean(gear),
        'carb': mean(carb)
    }

    VARIANCES = {
        'mpg': var(mpg),
        'cyl': var(cyl),
        'disp': var(disp),
        'hp': var(hp),
        'drat': var(drat),
        'wt': var(wt),
        'qsec': var(qsec),
        'vs': var(vs),
        'am': var(am),
        'gear': var(gear),
        'carb': var(carb)
    }

    # This is where all scaled data will be stored
    scaled_mpg = [round((data - MEANS['mpg']) / sqrt(VARIANCES['mpg']), DECIMAL_PLACE) for data in mpg]
    scaled_cyl = [round((data - MEANS['cyl']) / sqrt(VARIANCES['cyl']), DECIMAL_PLACE) for data in cyl]
    scaled_disp = [round((data - MEANS['disp']) / sqrt(VARIANCES['disp']), DECIMAL_PLACE) for data in disp]
    scaled_hp = [round((data - MEANS['hp']) / sqrt(VARIANCES['hp']), DECIMAL_PLACE) for data in hp]
    scaled_drat = [round((data - MEANS['drat']) / sqrt(VARIANCES['drat']), DECIMAL_PLACE) for data in drat]
    scaled_wt = [round((data - MEANS['wt']) / sqrt(VARIANCES['wt']), DECIMAL_PLACE) for data in wt]
    scaled_qsec = [round((data - MEANS['qsec']) / sqrt(VARIANCES['qsec']), DECIMAL_PLACE) for data in qsec]
    scaled_vs = [round((data - MEANS['vs']) / sqrt(VARIANCES['vs']), DECIMAL_PLACE) for data in vs]
    scaled_am = [round((data - MEANS['am']) / sqrt(VARIANCES['am']), DECIMAL_PLACE) for data in am]
    scaled_gear = [round((data - MEANS['gear']) / sqrt(VARIANCES['gear']), DECIMAL_PLACE) for data in gear]
    scaled_carb = [round((data - MEANS['carb']) / sqrt(VARIANCES['carb']), DECIMAL_PLACE) for data in carb]

    return (scaled_cyl, scaled_disp, scaled_hp, scaled_drat, scaled_wt, scaled_qsec, scaled_vs, scaled_am, scaled_gear, scaled_carb)

def print_scaled_values(scaled_values, headers, num_rows = 5):
    # Print the header
    header_str = ""
    for i in range(len(headers)):
        header_str += headers[i]

        while len(header_str) < (i + 1) * 15:
            header_str += " "
    
    print(header_str)
    

    # We then print the scaled values in a proper table format
    for i in range(num_rows):
        data_row = ""

        for j in range(len(headers)):
            data_row += str(scaled_values[j][i])

            while len(data_row) < (j + 1) * 15:
                data_row += " "
            
        print(data_row)

def report_scaled_mean_and_variance(scaled_values, headers, num_rows = 5):
    # Stores all the mean values for each data
    MEANS = {
        'cyl': mean(scaled_values[0]),
        'disp': mean(scaled_values[1]),
        'hp': mean(scaled_values[2]),
        'drat': mean(scaled_values[3]),
        'wt': mean(scaled_values[4]),
        'qsec': mean(scaled_values[5]),
        'vs': mean(scaled_values[6]),
        'am': mean(scaled_values[7]),
        'gear': mean(scaled_values[8]),
        'carb': mean(scaled_values[9])
    }

    VARIANCES = {
        'cyl': var(scaled_values[0]),
        'disp': var(scaled_values[1]),
        'hp': var(scaled_values[2]),
        'drat': var(scaled_values[3]),
        'wt': var(scaled_values[4]),
        'qsec': var(scaled_values[5]),
        'vs': var(scaled_values[6]),
        'am': var(scaled_values[7]),
        'gear': var(scaled_values[8]),
        'carb': var(scaled_values[9])
    }

    # We then print the table
    print("Scaled Means and Variances")
    
    # First, print the header
    header_str = " " * 7
    for i in range(len(headers)):
        header_str += headers[i]

        while len(header_str) < (i + 1) * 15 + 7:
            header_str += " "
    
    print(header_str)

    # Report the MEANS here
    mean_str = "MEAN   "
    for i in range(len(headers)):
        mean_str += str(round(MEANS[headers[i]], DECIMAL_PLACE))

        while len(mean_str) < (i + 1) * 15 + 7:
            mean_str += " "
    print(mean_str)


    
    # Report VAR here
    var_str = "VAR    "
    for i in range(len(headers)):
        var_str += str(round(VARIANCES[headers[i]], DECIMAL_PLACE))

        while len(var_str) < (i + 1) * 15 + 7:
            var_str += " "
    print(var_str)

def make_cars_plot(csv_file):
    num_lines = 0
    
    # This is where all data will be stored
    mpg = []
    cyl = []
    disp = []
    hp = []
    drat = []
    wt = []
    qsec = []
    vs = []
    am = []
    gear = []
    carb = []

    # Read all data from the CSV file
    with open(csv_file, newline='') as csvfile:
        line = csv.reader(csvfile, delimiter=',')

        line_num = 0
        for row in line:
            if line_num > 0: # We ignore the first line since that's just the header
                # Takes all values from the csv and puts them into their respective array
                mpg.append(float(row[0]))
                cyl.append(float(row[1]))
                disp.append(float(row[2]))
                hp.append(float(row[3]))
                drat.append(float(row[4]))
                wt.append(float(row[5]))
                qsec.append(float(row[6]))
                vs.append(float(row[7]))
                am.append(float(row[8]))
                gear.append(float(row[9]))
                carb.append(float(row[10]))
            line_num += 1
            num_lines += 1
    
    # Stores all the mean values for each data
    MEANS = {
        'mpg': mean(mpg),
        'cyl': mean(cyl),
        'disp': mean(disp),
        'hp': mean(hp),
        'drat': mean(drat),
        'wt': mean(wt),
        'qsec': mean(qsec),
        'vs': mean(vs),
        'am': mean(am),
        'gear': mean(gear),
        'carb': mean(carb)
    }

    VARIANCES = {
        'mpg': var(mpg),
        'cyl': var(cyl),
        'disp': var(disp),
        'hp': var(hp),
        'drat': var(drat),
        'wt': var(wt),
        'qsec': var(qsec),
        'vs': var(vs),
        'am': var(am),
        'gear': var(gear),
        'carb': var(carb)
    }

    # This is where all scaled data will be stored
    scaled_mpg = [round((data - MEANS['mpg']) / sqrt(VARIANCES['mpg']), DECIMAL_PLACE) for data in mpg]
    scaled_cyl = [round((data - MEANS['cyl']) / sqrt(VARIANCES['cyl']), DECIMAL_PLACE) for data in cyl]
    scaled_disp = [round((data - MEANS['disp']) / sqrt(VARIANCES['disp']), DECIMAL_PLACE) for data in disp]
    scaled_hp = [round((data - MEANS['hp']) / sqrt(VARIANCES['hp']), DECIMAL_PLACE) for data in hp]
    scaled_drat = [round((data - MEANS['drat']) / sqrt(VARIANCES['drat']), DECIMAL_PLACE) for data in drat]
    scaled_wt = [round((data - MEANS['wt']) / sqrt(VARIANCES['wt']), DECIMAL_PLACE) for data in wt]
    scaled_qsec = [round((data - MEANS['qsec']) / sqrt(VARIANCES['qsec']), DECIMAL_PLACE) for data in qsec]
    scaled_vs = [round((data - MEANS['vs']) / sqrt(VARIANCES['vs']), DECIMAL_PLACE) for data in vs]
    scaled_am = [round((data - MEANS['am']) / sqrt(VARIANCES['am']), DECIMAL_PLACE) for data in am]
    scaled_gear = [round((data - MEANS['gear']) / sqrt(VARIANCES['gear']), DECIMAL_PLACE) for data in gear]
    scaled_carb = [round((data - MEANS['carb']) / sqrt(VARIANCES['carb']), DECIMAL_PLACE) for data in carb]



    # Customize plot here
    plt.title("Scaled Weight vs Miles per Gallon")
    plt.xlabel("Scaled Weight")
    plt.ylabel("Miles per Gallon")

    # 0 --> theta_intercept
    # 1 --> theta_scaled_wt
    # thetas = [18, -5.25]
    thetas = [RANDOM_INITIALIZATION for i in range(2)]

    def mpg_scaled_wt_prediction(x, theta_values = thetas):
        result = theta_values[0]

        for i in range(1, len(theta_values)):
            result += theta_values[i] * x
        
        return result

    def get_mpg_scaled_wt_regression_sum(theta_values = thetas):
        regression_sum = 0

        # Adds up the regression for each row
        for i in range(len(mpg)):
            regression_sum += pow(mpg_scaled_wt_prediction(scaled_wt[i], theta_values) - mpg[i], 2)
        
        return regression_sum / (2 * (num_lines - 0))
    
    def cost_derivative(theta_values = thetas, index = 0):
        regression_sum = 0

        # Adds up the regression for each row
        for i in range(len(mpg)):
            if index == 0:
                regression_sum += mpg_scaled_wt_prediction(scaled_wt[i], theta_values) - mpg[i]
            else:
                regression_sum += (mpg_scaled_wt_prediction(scaled_wt[i], theta_values) - mpg[i]) * scaled_wt[i]
        
        return regression_sum / (num_lines - 0)
    
    # Keep updating thetas until the change in all delta_regression is negligable
    SAFETY = 10000
    iteration = 0
    while iteration < SAFETY:
        base_regression = get_mpg_scaled_wt_regression_sum()
        
        new_thetas = []

        if base_regression - get_mpg_scaled_wt_regression_sum([thetas[0] - LEARNING_RATE * cost_derivative(), thetas[1]]) > base_regression - get_mpg_scaled_wt_regression_sum([thetas[0] + LEARNING_RATE * cost_derivative(), thetas[1]]):
            new_thetas.append(thetas[0] - LEARNING_RATE * cost_derivative())
        else:
            new_thetas.append(thetas[0] + LEARNING_RATE * cost_derivative())
        if base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1] - LEARNING_RATE * cost_derivative(index=1)]) > base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1] + LEARNING_RATE * cost_derivative(index=1)]):
            new_thetas.append(thetas[1] - LEARNING_RATE * cost_derivative(index=1))
        else:
            new_thetas.append(thetas[1] + LEARNING_RATE * cost_derivative(index=1))

        delta_regression = []

        # Populates delta_regression
        for i in range(len(new_thetas)):
            stuff = thetas.copy()
            stuff[i] = new_thetas[i]
            delta_regression.append(base_regression - get_mpg_scaled_wt_regression_sum(stuff))

        # We then check if all values in delta_regression is less than the CONVERGENCE threshold

        # Otherwise, we take action of whichever change lead to largest change in the regression sum
        index_delta_regression = delta_regression.index(max(delta_regression))
        thetas[index_delta_regression] = new_thetas[index_delta_regression]

        iteration += 1 # End of while loop
    print(f"Theta0 = {thetas[0]}")
    print(f"Theta1 = {thetas[1]}")
    
    # Once we get the proper theta values, we just plot it on the graph
    plt.scatter(scaled_wt, mpg)
    regression_line = [mpg_scaled_wt_prediction(x) for x in scaled_wt]
    plt.plot(scaled_wt, regression_line, color='red')
    plt.show()


def get_mpg_multi_linear_reg(csv_file):
    num_lines = 0
    
    # This is where all data will be stored
    mpg = []
    cyl = []
    disp = []
    hp = []
    drat = []
    wt = []
    qsec = []
    vs = []
    am = []
    gear = []
    carb = []

    # Read all data from the CSV file
    with open(csv_file, newline='') as csvfile:
        line = csv.reader(csvfile, delimiter=',')

        line_num = 0
        for row in line:
            if line_num > 0: # We ignore the first line since that's just the header
                # Takes all values from the csv and puts them into their respective array
                mpg.append(float(row[0]))
                cyl.append(float(row[1]))
                disp.append(float(row[2]))
                hp.append(float(row[3]))
                drat.append(float(row[4]))
                wt.append(float(row[5]))
                qsec.append(float(row[6]))
                vs.append(float(row[7]))
                am.append(float(row[8]))
                gear.append(float(row[9]))
                carb.append(float(row[10]))
            line_num += 1
            num_lines += 1
    
    # Stores all the mean values for each data
    MEANS = {
        'mpg': mean(mpg),
        'cyl': mean(cyl),
        'disp': mean(disp),
        'hp': mean(hp),
        'drat': mean(drat),
        'wt': mean(wt),
        'qsec': mean(qsec),
        'vs': mean(vs),
        'am': mean(am),
        'gear': mean(gear),
        'carb': mean(carb)
    }

    VARIANCES = {
        'mpg': var(mpg),
        'cyl': var(cyl),
        'disp': var(disp),
        'hp': var(hp),
        'drat': var(drat),
        'wt': var(wt),
        'qsec': var(qsec),
        'vs': var(vs),
        'am': var(am),
        'gear': var(gear),
        'carb': var(carb)
    }

    # This is where all scaled data will be stored
    scaled_mpg = [round((data - MEANS['mpg']) / sqrt(VARIANCES['mpg']), DECIMAL_PLACE) for data in mpg]
    scaled_cyl = [round((data - MEANS['cyl']) / sqrt(VARIANCES['cyl']), DECIMAL_PLACE) for data in cyl]
    scaled_disp = [round((data - MEANS['disp']) / sqrt(VARIANCES['disp']), DECIMAL_PLACE) for data in disp]
    scaled_hp = [round((data - MEANS['hp']) / sqrt(VARIANCES['hp']), DECIMAL_PLACE) for data in hp]
    scaled_drat = [round((data - MEANS['drat']) / sqrt(VARIANCES['drat']), DECIMAL_PLACE) for data in drat]
    scaled_wt = [round((data - MEANS['wt']) / sqrt(VARIANCES['wt']), DECIMAL_PLACE) for data in wt]
    scaled_qsec = [round((data - MEANS['qsec']) / sqrt(VARIANCES['qsec']), DECIMAL_PLACE) for data in qsec]
    scaled_vs = [round((data - MEANS['vs']) / sqrt(VARIANCES['vs']), DECIMAL_PLACE) for data in vs]
    scaled_am = [round((data - MEANS['am']) / sqrt(VARIANCES['am']), DECIMAL_PLACE) for data in am]
    scaled_gear = [round((data - MEANS['gear']) / sqrt(VARIANCES['gear']), DECIMAL_PLACE) for data in gear]
    scaled_carb = [round((data - MEANS['carb']) / sqrt(VARIANCES['carb']), DECIMAL_PLACE) for data in carb]

    thetas = [RANDOM_INITIALIZATION for i in range(11)]

    def mpg_scaled_wt_prediction(x, theta_values = thetas):
        result = theta_values[0]

        for i in range(1, len(theta_values)):
            result += theta_values[i] * x[i]
        
        return result
    
    def get_mpg_scaled_wt_regression_sum(theta_values = thetas):
        regression_sum = 0

        # Adds up the regression for each row
        for i in range(len(mpg)):
            regression_sum += pow(mpg_scaled_wt_prediction([scaled_mpg[i], scaled_cyl[i], scaled_disp[i], scaled_hp[i], scaled_drat[i], scaled_wt[i], scaled_qsec[i], scaled_vs[i], scaled_am[i], scaled_gear[i], scaled_carb[i]], theta_values) - mpg[i], 2)
        
        return regression_sum / (2 * (num_lines - 0))
    
    def cost_derivative(theta_values = thetas, index = 0):
        regression_sum = 0

        # Adds up the regression for each row
        for i in range(len(mpg)):
            if index == 0:
                regression_sum += mpg_scaled_wt_prediction([scaled_mpg[i], scaled_cyl[i], scaled_disp[i], scaled_hp[i], scaled_drat[i], scaled_wt[i], scaled_qsec[i], scaled_vs[i], scaled_am[i], scaled_gear[i], scaled_carb[i]], theta_values) - mpg[i]
            else:
                # 1 --> scaled_cyl
                # 2 --> scaled_disp
                # 3 --> scaled_hp
                # 4 --> scaled_drat
                # 5 --> scaled_wt
                # 6 --> scaled_qsec
                # 7 --> scaled_vs
                # 8 --> scaled_am
                # 9 --> scaled_gear
                # 10 --> scaled_carb
                sample = []

                if index == 1:
                    sample = scaled_cyl
                elif index == 2:
                    sample = scaled_disp
                elif index == 3:
                    sample = scaled_hp
                elif index == 4:
                    sample = scaled_drat
                elif index == 5:
                    sample = scaled_wt
                elif index == 6:
                    sample = scaled_qsec
                elif index == 7:
                    sample = scaled_vs
                elif index == 8:
                    sample = scaled_am
                elif index == 9:
                    sample = scaled_gear
                elif index == 10:
                    sample = scaled_carb
                
                regression_sum += (mpg_scaled_wt_prediction([scaled_mpg[i], scaled_cyl[i], scaled_disp[i], scaled_hp[i], scaled_drat[i], scaled_wt[i], scaled_qsec[i], scaled_vs[i], scaled_am[i], scaled_gear[i], scaled_carb[i]], theta_values) - mpg[i]) * sample[i]
        
        return regression_sum / (num_lines - 0)
    
    # Keep updating thetas until the change in all delta_regression is negligable
    SAFETY = 10000
    iteration = 0
    while iteration < SAFETY:
        base_regression = get_mpg_scaled_wt_regression_sum()

        new_thetas = []

        if base_regression - get_mpg_scaled_wt_regression_sum([thetas[0] - LEARNING_RATE * cost_derivative(), thetas[1], thetas[2], thetas[3], thetas[4], thetas[5], thetas[6], thetas[7], thetas[8], thetas[9], thetas[10]]) > base_regression - get_mpg_scaled_wt_regression_sum([thetas[0] + LEARNING_RATE * cost_derivative(), thetas[1], thetas[2], thetas[3], thetas[4], thetas[5], thetas[6], thetas[7], thetas[8], thetas[9], thetas[10]]):
            new_thetas.append(thetas[0] - LEARNING_RATE * cost_derivative())
        else:
            new_thetas.append(thetas[0] + LEARNING_RATE * cost_derivative())
        if base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1] - LEARNING_RATE * cost_derivative(index=1), thetas[2], thetas[3], thetas[4], thetas[5], thetas[6], thetas[7], thetas[8], thetas[9], thetas[10]]) > base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1] + LEARNING_RATE * cost_derivative(index=1), thetas[2], thetas[3], thetas[4], thetas[5], thetas[6], thetas[7], thetas[8], thetas[9], thetas[10]]):
            new_thetas.append(thetas[1] - LEARNING_RATE * cost_derivative(index=1))
        else:
            new_thetas.append(thetas[1] + LEARNING_RATE * cost_derivative(index=1))
        if base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1], thetas[2] - LEARNING_RATE * cost_derivative(index=2), thetas[3], thetas[4], thetas[5], thetas[6], thetas[7], thetas[8], thetas[9], thetas[10]]) > base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1], thetas[2] + LEARNING_RATE * cost_derivative(index=2), thetas[3], thetas[4], thetas[5], thetas[6], thetas[7], thetas[8], thetas[9], thetas[10]]):
            new_thetas.append(thetas[2] - LEARNING_RATE * cost_derivative(index=2))
        else:
            new_thetas.append(thetas[2] + LEARNING_RATE * cost_derivative(index=2))
        if base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1], thetas[2], thetas[3] - LEARNING_RATE * cost_derivative(index=3), thetas[4], thetas[5], thetas[6], thetas[7], thetas[8], thetas[9], thetas[10]]) > base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1], thetas[2], thetas[3] + LEARNING_RATE * cost_derivative(index=3), thetas[4], thetas[5], thetas[6], thetas[7], thetas[8], thetas[9], thetas[10]]):
            new_thetas.append(thetas[3] - LEARNING_RATE * cost_derivative(index=3))
        else:
            new_thetas.append(thetas[3] + LEARNING_RATE * cost_derivative(index=3))
        if base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1], thetas[2], thetas[3], thetas[4] - LEARNING_RATE * cost_derivative(index=4), thetas[5], thetas[6], thetas[7], thetas[8], thetas[9], thetas[10]]) > base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1], thetas[2], thetas[3], thetas[4] + LEARNING_RATE * cost_derivative(index=4), thetas[5], thetas[6], thetas[7], thetas[8], thetas[9], thetas[10]]):
            new_thetas.append(thetas[4] - LEARNING_RATE * cost_derivative(index=4))
        else:
            new_thetas.append(thetas[4] + LEARNING_RATE * cost_derivative(index=4))
        if base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1], thetas[2], thetas[3], thetas[4], thetas[5] - LEARNING_RATE * cost_derivative(index=5), thetas[6], thetas[7], thetas[8], thetas[9], thetas[10]]) > base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1], thetas[2], thetas[3], thetas[4], thetas[5] + LEARNING_RATE * cost_derivative(index=5), thetas[6], thetas[7], thetas[8], thetas[9], thetas[10]]):
            new_thetas.append(thetas[5] - LEARNING_RATE * cost_derivative(index=5))
        else:
            new_thetas.append(thetas[5] + LEARNING_RATE * cost_derivative(index=5))
        if base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1], thetas[2], thetas[3], thetas[4], thetas[5], thetas[6] - LEARNING_RATE * cost_derivative(index=6), thetas[7], thetas[8], thetas[9], thetas[10]]) > base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1], thetas[2], thetas[3], thetas[4], thetas[5], thetas[6] + LEARNING_RATE * cost_derivative(index=6), thetas[7], thetas[8], thetas[9], thetas[10]]):
            new_thetas.append(thetas[6] - LEARNING_RATE * cost_derivative(index=6))
        else:
            new_thetas.append(thetas[6] + LEARNING_RATE * cost_derivative(index=6))
        if base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1], thetas[2], thetas[3], thetas[4], thetas[5], thetas[6], thetas[7] - LEARNING_RATE * cost_derivative(index=7), thetas[8], thetas[9], thetas[10]]) > base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1], thetas[2], thetas[3], thetas[4], thetas[5], thetas[6], thetas[7] + LEARNING_RATE * cost_derivative(index=7), thetas[8], thetas[9], thetas[10]]):
            new_thetas.append(thetas[7] - LEARNING_RATE * cost_derivative(index=7))
        else:
            new_thetas.append(thetas[7] + LEARNING_RATE * cost_derivative(index=7))
        if base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1], thetas[2], thetas[3], thetas[4], thetas[5], thetas[6], thetas[7], thetas[8] - LEARNING_RATE * cost_derivative(index=8), thetas[9], thetas[10]]) > base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1], thetas[2], thetas[3], thetas[4], thetas[5], thetas[6], thetas[7], thetas[8] + LEARNING_RATE * cost_derivative(index=8), thetas[9], thetas[10]]):
            new_thetas.append(thetas[8] - LEARNING_RATE * cost_derivative(index=8))
        else:
            new_thetas.append(thetas[8] + LEARNING_RATE * cost_derivative(index=8))
        if base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1], thetas[2], thetas[3], thetas[4], thetas[5], thetas[6], thetas[7], thetas[8], thetas[9] - LEARNING_RATE * cost_derivative(index=9), thetas[10]]) > base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1], thetas[2], thetas[3], thetas[4], thetas[5], thetas[6], thetas[7], thetas[8], thetas[9] + LEARNING_RATE * cost_derivative(index=9), thetas[10]]):
            new_thetas.append(thetas[9] - LEARNING_RATE * cost_derivative(index=9))
        else:
            new_thetas.append(thetas[9] + LEARNING_RATE * cost_derivative(index=9))
        if base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1], thetas[2], thetas[3], thetas[4], thetas[5], thetas[6], thetas[7], thetas[8], thetas[9], thetas[10] - LEARNING_RATE * cost_derivative(index=10)]) > base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1], thetas[2], thetas[3], thetas[4], thetas[5], thetas[6], thetas[7], thetas[8], thetas[9], thetas[10] + LEARNING_RATE * cost_derivative(index=10)]):
            new_thetas.append(thetas[10] - LEARNING_RATE * cost_derivative(index=10))
        else:
            new_thetas.append(thetas[10] + LEARNING_RATE * cost_derivative(index=10))

        delta_regression = []
        # Populates delta_regression
        for i in range(len(new_thetas)):
            stuff = thetas.copy()
            stuff[i] = new_thetas[i]
            delta_regression.append(base_regression - get_mpg_scaled_wt_regression_sum(stuff))

        # Otherwise, we take action of whichever change lead to largest change in the regression sum
        index_delta_regression = delta_regression.index(max(delta_regression))
        thetas[index_delta_regression] = new_thetas[index_delta_regression]
        
        iteration += 1 # End of while loop
    print(f"Theta0 = {round(thetas[0], 5)}")
    print(f"Theta1 = {round(thetas[1], 5)}")
    print(f"Theta2 = {round(thetas[2], 5)}")
    print(f"Theta3 = {round(thetas[3], 5)}")
    print(f"Theta4 = {round(thetas[4], 5)}")
    print(f"Theta5 = {round(thetas[5], 5)}")
    print(f"Theta6 = {round(thetas[6], 5)}")
    print(f"Theta7 = {round(thetas[7], 5)}")
    print(f"Theta8 = {round(thetas[8], 5)}")
    print(f"Theta9 = {round(thetas[9], 5)}")
    print(f"Theta10 = {round(thetas[10], 5)}")

# Reports the theta values for cancer data
def get_cancer_thetas(csv_file = 'cancer.csv'):
    num_lines = 0

    # This is where all data will be stored
    lung_cancer = [] # Dependent Variable
    smoking = [] # Independent Variable

    # Read all data from the CSV file
    with open(csv_file, newline='') as csvfile:
        line = csv.reader(csvfile, delimiter=',')

        line_num = 0
        for row in line:
            if line_num > 0: # We ignore the first line since that's just the header
                # Take all values from the csv and puts them into their respective array
                lung_cancer.append(int(row[0]))
                smoking.append(int(row[1]))
            line_num += 1
            num_lines += 1
        
    thetas = [RANDOM_INITIALIZATION for i in range(2)]

    def lung_cancer_prediction(x, theta_values = thetas):
        result = theta_values[0]

        for i in range(1, len(theta_values)):
            result += theta_values[i] * x[i]
        
        return 1 / (1 + pow(math.e, -1 * result)) # We need the logistic regression form instead
    
    def get_regression_sum(theta_values = thetas):
        regression_sum = 0

        # Adds up the regression for each row
        for i in range(len(lung_cancer)):
            regression_sum += pow(lung_cancer_prediction([lung_cancer[i], smoking[i]], theta_values) - lung_cancer[i], 2)
        
        return regression_sum / (2 * num_lines)
    
    def cost_derivative(theta_values = thetas, index = 0):
        regression_sum = 0

        # Adds up the regression for each row
        for i in range(len(lung_cancer)):
            if index == 0:
                regression_sum += lung_cancer_prediction([lung_cancer[i], smoking[i]], theta_values) - lung_cancer[i]
            else:
                regression_sum += (lung_cancer_prediction([lung_cancer[i], smoking[i]], theta_values) - lung_cancer[i]) * smoking[i]
        
        return regression_sum / num_lines

    # Keep updating theta values until changes to all leads to less of a change in regression sum than the convergence threshold
    SAFETY = 50000
    iteration = 0
    while iteration < SAFETY:
        base_regression = get_regression_sum()

        new_thetas = []

        if base_regression - get_regression_sum([thetas[0] - LEARNING_RATE * cost_derivative(), thetas[1]]) > base_regression - get_regression_sum([thetas[0] + LEARNING_RATE * cost_derivative(), thetas[1]]):
            new_thetas.append(thetas[0] - LEARNING_RATE * cost_derivative())
        else:
            new_thetas.append(thetas[0] + LEARNING_RATE * cost_derivative())
        if base_regression - get_regression_sum([thetas[0], thetas[1] - LEARNING_RATE * cost_derivative(index=1)]) > base_regression - get_regression_sum([thetas[0], thetas[1] + LEARNING_RATE * cost_derivative(index=1)]):
            new_thetas.append(thetas[1] - LEARNING_RATE * cost_derivative(index=1))
        else:
            new_thetas.append(thetas[1] + LEARNING_RATE * cost_derivative(index=1))
        
        delta_regression = []
        # Populates delta_regression
        for i in range(len(new_thetas)):
            stuff = thetas.copy()
            stuff[i] = new_thetas[i]
            delta_regression.append(base_regression - get_regression_sum(stuff))
        
        # Otherwise, we take action of whichever change lead to largest change in the regression sum
        index_delta_regression = delta_regression.index(max(delta_regression))
        thetas[index_delta_regression] = new_thetas[index_delta_regression]
        
        iteration += 1 # End of while loop
    print(f"Theta0 = {round(thetas[0], 5)}")
    print(f"Theta1 = {round(thetas[1], 5)}")
    print("")

    # Print the confusion matrix and accuracy here
    # Correct Predictions
    oc_pc = 0 # Observed Cancer / Predicted Cancer
    onc_pnc = 0 # Observed No Cancer / Predicted No Cancer

    # Incorrect Predictions
    oc_pnc = 0 # Observed Cancer / Predicted No Cancer
    onc_pc = 0 # Observed No Cancer / Predicted Cancer

    num_entries = 0
    num_correct = 0

    num_pc = 0 # Number of times it Predicted Cancer
    num_pc_correct = 0 # Number of times it correctly Predicted Cancer

    num_pnc = 0 # Number of times it Predicted No Cancer
    num_pnc_correct = 0 # Number of times it correctly Predicted No Cancer

    # Gets prediction from a logistic probability curve
    def collapse(x):
        result = thetas[0]

        for i in range(1, len(thetas)):
            result += thetas[i] * x
        
        result = 1 / (1 + pow(math.e, -1 * result))


        return result
    
    # Classify Individuals Here
    print("Classifying Individuals:")

    for i in range(len(smoking)):
        line = "Patient " + str(i + 1) + ": Smoking = " + str(smoking[i]) + " --> " + str(p := round(collapse(smoking[i]) * 100, 2)) + "% has cancer --> "

        if p < 50: # Decides it (probably) doesn't have cancer
            line += "Doesn't have cancer"
        else:
            line += "Has cancer"
        
        print(line)
    
    print("\n")
    
    for i in range(len(lung_cancer)):
        predicted = smoking[i]
        observed = lung_cancer[i]
        
        # Needed to get confusion matrix
        if observed == 1 and predicted == 1: # Observed Cancer / Predicted Cancer
            oc_pc += 1
        elif observed == 0 and predicted == 0: # Observed No Cancer / Predicted No Cancer
            onc_pnc += 1
        elif observed == 1 and predicted == 0: # Observed Cancer / Predicted No Cancer
            oc_pnc += 1
        elif observed == 0 and predicted == 1: # Observed No Cancer / Predicted Cancer
            onc_pc += 1
        
        # Needed to calculate overall accuracy
        if predicted == observed:
            num_correct += 1

        # Needed to calculate category-wise percentages
        # Predicted Cancer
        if predicted == 1: # Predicted Cancer
            num_pc += 1
            if observed == 1: # Observed Cancer
                num_pc_correct += 1
        elif predicted == 0: # Predicted No Cancer
            num_pnc += 1
            if observed == 0: # Observed No Cancer
                num_pnc_correct += 1

        num_entries += 1
    
    # Print Confusion Matrix Here
    print(f"{' ' * 30}Predicted")
    print(f"{' ' * 25}Cancer{' ' * 5}No Cancer")

    first = "Observed" + (' ' * 5) + "Cancer" + (' ' * 6) + str(oc_pc)
    second = (' ' * 13) + "No Cancer" + (' ' * 3) + str(onc_pc)

    while len(first) < 36:
        first += ' '
    while len(second) < 36:
        second += ' '
    
    first += str(oc_pnc)
    second += str(onc_pnc)
    
    print(first)
    print(second)
    print("")
    
    # Print Accuracies Here
    print(f"Overall Accuracy = {round((num_correct / num_entries) * 100, 2)}%")
    print(f"Correctly Predicted Cancer {round((num_pc_correct / num_pc) * 100, 2)}% of the time")
    print(f"Correctly No Predicted Cancer {round((num_pnc_correct / num_pnc) * 100, 2)}% of the time")

# Runs everything for question 4
def question4(csv_file = 'emails.csv'):
    num_lines = 0

    # This is where all data will be stored
    spam_flag = [] # Dependent Variable
    subject_length = [] # Independent Variable
    suspicious_word_count = [] # Independent Variable

    # Read all data from the CSV file
    with open(csv_file, newline='') as csvfile:
        line = csv.reader(csvfile, delimiter=',')

        line_num = 0
        for row in line:
            if line_num > 0: # We ignore the first line since that's just the header
                # Take all values from the csv and puts them into their respective array
                spam_flag.append(int(row[0]))
                subject_length.append(int(row[1]))
                suspicious_word_count.append(int(row[2]))
            line_num += 1
            num_lines += 1
        
    # Let's then make our scatter plot
    spam_subject_length = []
    spam_suspicious_word_count = []
    
    not_spam_subject_length = []
    not_spam_suspicious_word_count = []

    for i in range(len(spam_flag)):
        if spam_flag[i] == 0: # Not Spam
            not_spam_subject_length.append(subject_length[i])
            not_spam_suspicious_word_count.append(suspicious_word_count[i])
        elif spam_flag[i] == 1: # Spam
            spam_subject_length.append(subject_length[i])
            spam_suspicious_word_count.append(suspicious_word_count[i])
    
    plt.title("Subject Length & Suspicious Word Count of Spam vs Not Spam Emails")
    plt.xlabel("Suspicious Word Count")
    plt.ylabel("Subject Length")
    
    plt.scatter(spam_suspicious_word_count, spam_subject_length)
    plt.scatter(not_spam_suspicious_word_count, not_spam_subject_length)
    plt.legend(['Spam', 'Not Spam'], loc='best')

    # Resume with learning the logistic regression
    thetas = [RANDOM_INITIALIZATION for i in range(3)]

    def spam_prediction(x, theta_values = thetas):
        result = theta_values[0]

        for i in range(1, len(theta_values)):
            result += theta_values[i] * x[i]
        
        return 1 / (1 + pow(math.e, -1 * result)) # We need the logistic regression form instead
    
    def get_regression_sum(theta_values = thetas):
        regression_sum = 0

        # Adds up the regression for each row
        for i in range(len(spam_flag)):
            regression_sum += pow(spam_prediction([spam_flag[i], subject_length[i], suspicious_word_count[i]], theta_values) - spam_flag[i], 2)
        
        return regression_sum / (2 * num_lines)
    
    thetas = [-35.68618463186144, 0.23549743621369503, 3.892160533123672]
    
    def cost_derivative(theta_values = thetas, index = 0):
        regression_sum = 0

        # Adds up the regression for each row
        for i in range(len(spam_flag)):
            if index == 0:
                regression_sum += abs(spam_prediction([spam_flag[i], subject_length[i], suspicious_word_count[i]], theta_values) - spam_flag[i])
            else:
                # 1 --> subject_length
                # 2 --> suspicious_word_count
                sample = []

                if index == 1:
                    sample = subject_length
                elif index == 2:
                    sample = suspicious_word_count
                
                regression_sum += abs((spam_prediction([spam_flag[i], subject_length[i], suspicious_word_count[i]], theta_values) - spam_flag[i]) * sample[i])
        
        return regression_sum / num_lines
    
    # Keep updating thetas until the change in all delta_regression is negligable
    SAFETY = 500000
    iteration = 0
    while iteration < SAFETY:
        base_regresssion = get_regression_sum()

        new_thetas = []

        if base_regresssion - get_regression_sum([thetas[0] - LEARNING_RATE * cost_derivative(), thetas[1], thetas[2]]) > base_regresssion - get_regression_sum([thetas[0] + LEARNING_RATE * cost_derivative(), thetas[1], thetas[2]]):
            new_thetas.append(thetas[0] - LEARNING_RATE * cost_derivative())
        else:
            new_thetas.append(thetas[0] + LEARNING_RATE * cost_derivative())
        if base_regresssion - get_regression_sum([thetas[0], thetas[1] - LEARNING_RATE * cost_derivative(index=1), thetas[2]]) > base_regresssion - get_regression_sum([thetas[0], thetas[1] + LEARNING_RATE * cost_derivative(index=1), thetas[2]]):
            new_thetas.append(thetas[1] - LEARNING_RATE * cost_derivative(index=1))
        else:
            new_thetas.append(thetas[1] + LEARNING_RATE * cost_derivative(index=1))
        if base_regresssion - get_regression_sum([thetas[0], thetas[1], thetas[2] - LEARNING_RATE * cost_derivative(index=2)]) > base_regresssion - get_regression_sum([thetas[0], thetas[1], thetas[2] + LEARNING_RATE * cost_derivative(index=2)]):
            new_thetas.append(thetas[2] - LEARNING_RATE * cost_derivative(index=2))
        else:
            new_thetas.append(thetas[2] + LEARNING_RATE * cost_derivative(index=2))
        
        delta_regression = []
        # Populates delta_regression
        for i in range(len(new_thetas)):
            stuff = thetas.copy()
            stuff[i] = new_thetas[i]
            delta_regression.append(1 * (base_regresssion - get_regression_sum(stuff)))
        
        # Otherwise, we take action of whichever chagne lead to largest change in the regression sum
        index_delta_regression = delta_regression.index(max(delta_regression))
        thetas[index_delta_regression] = new_thetas[index_delta_regression]

        iteration += 1 # End of while loop
    print(f"Theta0 = {round(thetas[0], 5)}")
    print(f"Theta1 = {round(thetas[1], 5)}")
    print(f"Theta2 = {round(thetas[2], 5)}")
    print("")

    # Print the confusion matrix and accuracy here
    # Correct Predictions
    os_ps = 0 # Observed Spam / Predicted Spam
    ons_pns = 0 # Observed Not Spam / Predicted Not Spam

    # Incorrect Predictions
    os_pns = 0 # Observed Spam / Predicted Not Spam
    ons_ps = 0 # Observed Not Spam / Predicted Not Spam

    num_entries = 0
    num_correct = 0

    num_ps = 0 # Number of times it Predicted Spam
    num_ps_correct = 0 # Number of times it correclty Predicted Spam

    num_pns = 0 # Number of times it Predicted Not Spam
    num_pns_correct = 0 # Number of times it correctly Predicted Not Spam

    # Gets prediction from a logistic probability curve
    def collapse(x):
        result = thetas[0]

        for i in range(1, len(thetas)):
            result += thetas[i] * x[i]
        
        result = 1 / (1 + pow(math.e, -1 * result))

        return result

    # Classify Individuals Here
    print("Classifying Individuals:")

    for i in range(len(spam_flag)):
        line = "Email " + str(i + 1) + ": Sub_Len = " + str(subject_length[i]) + " | Sus_Count = " + str(suspicious_word_count[i]) + " --> " + str(p := round(collapse([spam_flag[i], subject_length[i], suspicious_word_count[i]]) * 100, 2)) + "% is spam --> "

        if p < 50: # Decides it (probably) isn't spam
            line += "NO, it's not spam"
        else:
            line += "YES, it's spam"
        
        print(line)
    
    print("\n")

    for i in range(len(spam_flag)):
        predicted = 0

        if round(collapse([spam_flag[i], subject_length[i], suspicious_word_count[i]]) * 100, 2) >= 50:
            predicted = 1
        
        observed = spam_flag[i]

        # Needed to get confusion matrix
        if observed == 1 and predicted == 1: # Observed Spam / Predicted Spam
            os_ps += 1
        elif observed == 0 and predicted == 0: # Observed No Spam / Predicted No Spam
            ons_pns += 1
        elif observed == 1 and predicted == 0: # Observed Spam / Predicted No Spam
            os_pns += 1
        elif observed == 0 and predicted == 1: # Observed No Spam / Predicted Spam
            ons_ps += 1
        
        # Needed to calculate overall accuracy
        if predicted == observed:
            num_correct += 1

        # Needed to calculate category-wise percentages
        # Predicted Spam
        if predicted == 1: # Predicted Spam
            num_ps += 1
            if observed == 1: # Observed Spam
                num_ps_correct += 1
        elif predicted == 0: # Predicted No Spam
            num_pns += 1
            if observed == 0: # Observed No Spam
                num_pns_correct += 1

        num_entries += 1
    
    # Print Confusion Matrix Here
    print(f"{' ' * 30}Predicted")
    print(f"{' ' * 25}Spam{' ' * 5}No Spam")

    first = "Observed" + (' ' * 5) + "Spam" + (' ' * 8) + str(os_ps)
    second = (' ' * 13) + "No Spam" + (' ' * 5) + str(ons_ps)

    while len(first) < 34:
        first += ' '
    while len(second) < 34:
        second += ' '
    
    first += str(os_pns)
    second += str(ons_pns)
    
    print(first)
    print(second)
    print("")
    
    # Print Accuracies Here
    print(f"Overall Accuracy = {round((num_correct / num_entries) * 100, 2)}%")
    print(f"Correctly Predicted Spam {round((num_ps_correct / num_ps) * 100, 2)}% of the time")
    print(f"Correctly No Predicted Spam {round((num_pns_correct / num_pns) * 100, 2)}% of the time")

    # End of function


# END OF FILE