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

def make_mpg_scaled_wt_sp(csv_file):
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
    
    plt.scatter(mpg, scaled_wt)
    plt.show()


# RESUME