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
        'carb': mean(carb)
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

def report_mean_and_variance(csv_file, headers):
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
        'carb': mean(carb)
    }

    

    # We then print the table
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
        'carb': mean(carb)
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
    
    # plt.scatter(mpg, scaled_wt)
    # plt.show()



    # We then plot the simple linear hypothesis here
    # theta_intercept = RANDOM_INITIALIZATION
    # theta_scaled_wt = RANDOM_INITIALIZATION

    # theta_intercept = 18
    # theta_scaled_wt = -5.25

    # 0 --> theta_intercept
    # 1 --> theta_scaled_wt
    # thetas = [18, -5.25]
    thetas = [RANDOM_INITIALIZATION for i in range(2)]

    def mpg_scaled_wt_prediction(x, theta_values = thetas):
        # return theta_values[0] + (theta_values[1] * x)
        result = theta_values[0]

        for  i in range(1, len(theta_values)):
            result += theta_values[i] * x
        
        return result

    def get_mpg_scaled_wt_regression_sum(theta_values = thetas):
        regression_sum = 0

        # Adds up the regression for each row
        for i in range(len(mpg)):
            regression_sum += pow(mpg[i] - mpg_scaled_wt_prediction(scaled_wt[i], theta_values), 2)
        
        return regression_sum

    # Keep updating theta_intercept and/or theta_scaled_wt until changes to both leads to less of a change in regression sum than the convergence threshold
    SAFETY = 1000
    iteration = 0
    while iteration < SAFETY:
        base_regression = get_mpg_scaled_wt_regression_sum()
        
        # Index
        #   0 --> theta0--
        #   1 --> theta1--
        #   2 --> theta0++
        #   3 --> theta1++
        delta_regression = []

        # Populate delta_regression
        # SUBTRACTING LEARNING RATE
        delta_regression.append(get_mpg_scaled_wt_regression_sum([thetas[0] - LEARNING_RATE, thetas[1]]))
        delta_regression.append(get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1] - LEARNING_RATE]))
        
        # ADDING LEARNING RATE
        delta_regression.append(get_mpg_scaled_wt_regression_sum([thetas[0] + LEARNING_RATE, thetas[1]]))
        delta_regression.append(get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1] + LEARNING_RATE]))

        

        # We then turn any negatives into 0
        delta_regression = [0 if num < 0 else num for num in delta_regression]



        # We then check if all values in delta_regression is less than the CONVERGENCE threshold

        # Checks if all delta_regression values are less than the CONVERGENEC threshold
        def all_less(nums):
            flag = True
            
            for num in nums:
                if num > CONVERGENCE:
                    flag = False
            
            return flag

        # If all delta_regression values are less than the CONVERGENCE threshold, we stop iterating
        if all_less(delta_regression):
            break
        

        # Otherwise, we take action of whichever change lead to largest change in the regression sum
        factor = -1
        index_delta_regression = delta_regression.index(max(delta_regression))

        if index_delta_regression < len(delta_regression) / 2:
            factor = 1
        
        index_delta_regression %= len(delta_regression) / 2

        # Index:
        #   0 --> theta0
        #   1 --> theta1
        thetas[int(index_delta_regression)] += factor * LEARNING_RATE

        iteration += 1 # End of while loop
    print(f"While loop ended at iteration {iteration}")
    print(f"Theta0 = {round(thetas[0], 2)}")
    print(f"Theta1 = {round(thetas[1], 2)}")

    # Once we get the proper theta values, we just plot it on the graph
    mpg_scaled_wt_regression = list(map(mpg_scaled_wt_prediction, scaled_wt))
    plt.scatter(scaled_wt, mpg)
    plt.plot(scaled_wt, mpg_scaled_wt_regression)
    plt.show()
    print(f"Sum Regression {round(get_mpg_scaled_wt_regression_sum(), 2)}")


# RESUME