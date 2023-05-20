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
        
        return regression_sum / (2 * num_lines)

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
        delta_regression.append(base_regression - get_mpg_scaled_wt_regression_sum([thetas[0] - LEARNING_RATE, thetas[1]]))
        delta_regression.append(base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1] - LEARNING_RATE]))
        
        # ADDING LEARNING RATE
        delta_regression.append(base_regression - get_mpg_scaled_wt_regression_sum([thetas[0] + LEARNING_RATE, thetas[1]]))
        delta_regression.append(base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1] + LEARNING_RATE]))

        

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

        if index_delta_regression >= len(delta_regression) / 2:
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
    plt.plot(scaled_wt, mpg_scaled_wt_regression, color='red')
    plt.show()
    print(f"Sum Regression {round(get_mpg_scaled_wt_regression_sum(), 2)}")

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

    def mpg_scaled_prediction(x, theta_values = thetas):
        # return theta_values[0] + (theta_values[1] * x)
        result = theta_values[0]

        for i in range(1, len(theta_values)):
            result += theta_values[i] * x[i]
        
        return result

    def get_mpg_scaled_wt_regression_sum(theta_values = thetas):
        regression_sum = 0

        # Adds up the regression for each row
        for i in range(len(mpg)):
            regression_sum += pow(mpg[i] - mpg_scaled_prediction([scaled_mpg[i], scaled_cyl[i], scaled_disp[i], scaled_hp[i], scaled_drat[i], scaled_wt[i], scaled_qsec[i], scaled_vs[i], scaled_am[i], scaled_gear[i], scaled_carb[i]], theta_values), 2)
        
        return regression_sum / (2 * num_lines)
    
    # Keep updating theta values until changes to all leads to less of a change in regression sum than the convergence threshold
    SAFETY = 10000
    iteration = 0
    while iteration < SAFETY:
        base_regression = get_mpg_scaled_wt_regression_sum()
        
        # Index
        #   0 --> theta0--
        #   1 --> theta1--
        #   2 --> theta2--
        #   3 --> theta3--
        #   4 --> theta4--
        #   5 --> theta5--
        #   6 --> theta6--
        #   7 --> theta7--
        #   8 --> theta8--
        #   9 --> theta9--
        #   10 --> theta10--
        #   11 --> theta0++
        #   12 --> theta1++
        #   13 --> theta2++
        #   14 --> theta3++
        #   15 --> theta4++
        #   16 --> theta5++
        #   17 --> theta6++
        #   18 --> theta7++
        #   19 --> theta8++
        #   20 --> theta9++
        #   21 --> theta10++
        delta_regression = []

        # Populate delta_regression (also, yikes)
        # SUBTRACTING LEARNING RATE
        delta_regression.append(base_regression - get_mpg_scaled_wt_regression_sum([thetas[0] - LEARNING_RATE, thetas[1], thetas[2], thetas[3], thetas[4], thetas[5], thetas[6], thetas[7], thetas[8], thetas[9], thetas[10]]))
        delta_regression.append(base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1] - LEARNING_RATE, thetas[2], thetas[3], thetas[4], thetas[5], thetas[6], thetas[7], thetas[8], thetas[9], thetas[10]]))
        delta_regression.append(base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1], thetas[2] - LEARNING_RATE, thetas[3], thetas[4], thetas[5], thetas[6], thetas[7], thetas[8], thetas[9], thetas[10]]))
        delta_regression.append(base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1], thetas[2], thetas[3] - LEARNING_RATE, thetas[4], thetas[5], thetas[6], thetas[7], thetas[8], thetas[9], thetas[10]]))
        delta_regression.append(base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1], thetas[2], thetas[3], thetas[4] - LEARNING_RATE, thetas[5], thetas[6], thetas[7], thetas[8], thetas[9], thetas[10]]))
        delta_regression.append(base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1], thetas[2], thetas[3], thetas[4], thetas[5] - LEARNING_RATE, thetas[6], thetas[7], thetas[8], thetas[9], thetas[10]]))
        delta_regression.append(base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1], thetas[2], thetas[3], thetas[4], thetas[5], thetas[6] - LEARNING_RATE, thetas[7], thetas[8], thetas[9], thetas[10]]))
        delta_regression.append(base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1], thetas[2], thetas[3], thetas[4], thetas[5], thetas[6], thetas[7] - LEARNING_RATE, thetas[8], thetas[9], thetas[10]]))
        delta_regression.append(base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1], thetas[2], thetas[3], thetas[4], thetas[5], thetas[6], thetas[7], thetas[8] - LEARNING_RATE, thetas[9], thetas[10]]))
        delta_regression.append(base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1], thetas[2], thetas[3], thetas[4], thetas[5], thetas[6], thetas[7], thetas[8], thetas[9] - LEARNING_RATE, thetas[10]]))
        delta_regression.append(base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1], thetas[2], thetas[3], thetas[4], thetas[5], thetas[6], thetas[7], thetas[8], thetas[9], thetas[10] - LEARNING_RATE]))
        
        # ADDING LEARNING RATE
        delta_regression.append(base_regression - get_mpg_scaled_wt_regression_sum([thetas[0] + LEARNING_RATE, thetas[1], thetas[2], thetas[3], thetas[4], thetas[5], thetas[6], thetas[7], thetas[8], thetas[9], thetas[10]]))
        delta_regression.append(base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1] + LEARNING_RATE, thetas[2], thetas[3], thetas[4], thetas[5], thetas[6], thetas[7], thetas[8], thetas[9], thetas[10]]))
        delta_regression.append(base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1], thetas[2] + LEARNING_RATE, thetas[3], thetas[4], thetas[5], thetas[6], thetas[7], thetas[8], thetas[9], thetas[10]]))
        delta_regression.append(base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1], thetas[2], thetas[3] + LEARNING_RATE, thetas[4], thetas[5], thetas[6], thetas[7], thetas[8], thetas[9], thetas[10]]))
        delta_regression.append(base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1], thetas[2], thetas[3], thetas[4] + LEARNING_RATE, thetas[5], thetas[6], thetas[7], thetas[8], thetas[9], thetas[10]]))
        delta_regression.append(base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1], thetas[2], thetas[3], thetas[4], thetas[5] + LEARNING_RATE, thetas[6], thetas[7], thetas[8], thetas[9], thetas[10]]))
        delta_regression.append(base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1], thetas[2], thetas[3], thetas[4], thetas[5], thetas[6] + LEARNING_RATE, thetas[7], thetas[8], thetas[9], thetas[10]]))
        delta_regression.append(base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1], thetas[2], thetas[3], thetas[4], thetas[5], thetas[6], thetas[7] + LEARNING_RATE, thetas[8], thetas[9], thetas[10]]))
        delta_regression.append(base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1], thetas[2], thetas[3], thetas[4], thetas[5], thetas[6], thetas[7], thetas[8] + LEARNING_RATE, thetas[9], thetas[10]]))
        delta_regression.append(base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1], thetas[2], thetas[3], thetas[4], thetas[5], thetas[6], thetas[7], thetas[8], thetas[9] + LEARNING_RATE, thetas[10]]))
        delta_regression.append(base_regression - get_mpg_scaled_wt_regression_sum([thetas[0], thetas[1], thetas[2], thetas[3], thetas[4], thetas[5], thetas[6], thetas[7], thetas[8], thetas[9], thetas[10] + LEARNING_RATE]))

        

        # We then turn any negatives into 0
        delta_regression = [0 if num < 0 else num for num in delta_regression]



        # We then check if all values in delta_regression is less than the CONVERGENCE threshold

        # Checks if all delta_regression values are less than the CONVERGENCE threshold
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

        if index_delta_regression >= len(delta_regression) / 2:
            factor = 1
        
        index_delta_regression %= len(delta_regression) / 2

        # Index:
        #   0 --> theta0
        #   1 --> theta1
        #   2 --> theta2
        #   3 --> theta3
        #   4 --> theta4
        #   5 --> theta5
        #   6 --> theta6
        #   7 --> theta7
        #   8 --> theta8
        #   9 --> theta9
        #   10 --> theta10
        thetas[int(index_delta_regression)] += factor * LEARNING_RATE

        iteration += 1 # End of while loop
    print(f"While loop ended at iteration {iteration}")
    print(f"Theta0 = {round(thetas[0], 2)}")
    print(f"Theta1 = {round(thetas[1], 2)}")
    print(f"Theta2 = {round(thetas[2], 2)}")
    print(f"Theta3 = {round(thetas[3], 2)}")
    print(f"Theta4 = {round(thetas[4], 2)}")
    print(f"Theta5 = {round(thetas[5], 2)}")
    print(f"Theta6 = {round(thetas[6], 2)}")
    print(f"Theta7 = {round(thetas[7], 2)}")
    print(f"Theta8 = {round(thetas[8], 2)}")
    print(f"Theta9 = {round(thetas[9], 2)}")
    print(f"Theta10 = {round(thetas[10], 2)}")
    
    print(f"The Thetas {thetas}")
    # mpg_scaled_wt_regression = list(map(mpg_scaled_prediction, scaled_wt))
    # plt.scatter(scaled_wt, mpg)
    # plt.plot(scaled_wt, mpg_scaled_wt_regression)
    # plt.show()

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
            regression_sum += pow(lung_cancer[i] - lung_cancer_prediction([lung_cancer[i], smoking[i]], theta_values), 2)
        
        return regression_sum / (2 * num_lines)

    # Keep updating theta values until changes to all leads to less of a change in regression sum than the convergence threshold
    SAFETY = 10000
    iteration = 0
    while iteration < SAFETY:
        base_regression = get_regression_sum()

        # Index
        #   0 --> theta0--
        #   1 --> theta1--
        #   2 --> theta0++
        #   3 --> theta1++
        delta_regression = []

        # Populate delta_regression
        # SUBTRACTING LEARNING RATE
        delta_regression.append(base_regression - get_regression_sum([thetas[0] - LEARNING_RATE, thetas[1]]))
        delta_regression.append(base_regression - get_regression_sum([thetas[0], thetas[1] - LEARNING_RATE]))

        # ADDING LEARNING RATE
        delta_regression.append(base_regression - get_regression_sum([thetas[0] + LEARNING_RATE, thetas[1]]))
        delta_regression.append(base_regression - get_regression_sum([thetas[0], thetas[1] + LEARNING_RATE]))



        # We then turn any negatives into 0
        delta_regression = [0 if num < 0 else num for num in delta_regression]



        # We then check if all values in delta_regression is less than the CONVERGENCE threshold

        # Checks if all delta_regression values are less than the CONVERGENCE threshold
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

        if index_delta_regression >= len(delta_regression) / 2:
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

    print(f"The Thetas {thetas}")
    print("\n")

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

        # if 0 <= result and result < 0.5:
        #     return 0 # Predicts No Cancer
        # elif 0.5 <= result and result <= 1:
        #     return 1 # Predicts Cancer
        # return -1 # Invalid input
    
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
        
    thetas = [RANDOM_INITIALIZATION for i in range(3)]

    def spam_flag_prediction(x, theta_values = thetas):
        result = theta_values[0]

        for i in range(1, len(theta_values)):
            result += theta_values[i] * x[i]
        
        return 1 / (1 + pow(math.e, -1 * result)) # We need the logistic regression form instead
    
    def get_regression_sum(theta_values = thetas):
        regression_sum = 0

        # Adds up the regression for each row
        for i in range(len(spam_flag)):
            regression_sum += pow(spam_flag[i] - spam_flag_prediction([spam_flag[i], subject_length[i], suspicious_word_count[i]], theta_values), 2)
        
        return regression_sum / (2 * num_lines)
    
    # Keep updating theta values until changes to all leads to less of a change in regression sum than the convergence threshold
    SAFETY = 10000
    iteration = 0
    while iteration < SAFETY:
        base_regression = get_regression_sum()

        # Index
        #   0 --> theta0--
        #   1 --> theta1--
        #   2 --> theta2--
        #   3 --> theta0++
        #   4 --> theta1++
        #   5 --> theta2++
        delta_regression = []

        # Populate delta_regression
        #SUBTRACTING LEARNING RATE
        delta_regression.append(base_regression - get_regression_sum([thetas[0] - LEARNING_RATE, thetas[1], thetas[2]]))
        delta_regression.append(base_regression - get_regression_sum([thetas[0], thetas[1] - LEARNING_RATE, thetas[2]]))
        delta_regression.append(base_regression - get_regression_sum([thetas[0], thetas[1], thetas[2] - LEARNING_RATE]))

        # ADDING LEARNING RATE
        delta_regression.append(base_regression - get_regression_sum([thetas[0] + LEARNING_RATE, thetas[1], thetas[2]]))
        delta_regression.append(base_regression - get_regression_sum([thetas[0], thetas[1] + LEARNING_RATE, thetas[2]]))
        delta_regression.append(base_regression - get_regression_sum([thetas[0], thetas[1], thetas[2] + LEARNING_RATE]))



        # We then turn any negatives into 0
        delta_regression = [0 if num < 0 else num for num in delta_regression]



        # We then check if all values in delta_regression is less than the CONVERGENCE threshold

        # Checks if all delta_regression values are less than the CONVERGENCE threshold
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

        if index_delta_regression >= len(delta_regression) / 2:
            factor = 1
        
        index_delta_regression %= len(delta_regression) / 2

        # Index:
        #   0 --> theta0
        #   1 --> theta1
        #   2 --> theta2
        thetas[int(index_delta_regression)] += factor * LEARNING_RATE

        iteration += 1 # End of while loop
    print(f"While loop ended at iteration {iteration}")
    print(f"Theta0 = {round(thetas[0], 2)}")
    print(f"Theta1 = {round(thetas[1], 2)}")
    print(f"Theta2 = {round(thetas[2], 2)}")

    print(f"The Thetas {thetas}")
    print("\n")


# END OF FILE