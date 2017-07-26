# Diane Nava
##Version 3.0 

from scipy import signal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import csv
import os
from scipy.signal import argrelextrema
import numpy as np
from scipy.interpolate import UnivariateSpline

# read each csv file
path = 'C:\\Users\\Diane\\Documents\\Diane1\\python working folder\working\\'

allfiles = sorted(glob('*Photopic*.txt'))

listoffiles = []
listoffiles.append(allfiles)

filtered = []
biggerarray1 = []
biggerarray2 = []
biggestarray1 = []
biggestarray2 = []

columna = [2, 7, 12]
columnb = [1, 5, 10]


# 1,4, and 7 for column b
# 2,5,8 for column a
# import xlwt
def filter(yy):
    xn = yy
    b, a = signal.butter(3, 0.05)
    zi = signal.lfilter_zi(b, a)
    z, _ = signal.lfilter(b, a, xn, zi=zi * xn[0])
    z2, _ = signal.lfilter(b, a, z, zi=zi * z[0])
    result = signal.filtfilt(b, a, xn)
    return result


def get_inflection_points(data):
    '''
    data is 2D. first column is time, the second is values
    this function gets the inflection points to find peaks
    '''
    # in the array called data, take all the rows, first column
    time = data[:, 0]

    # dake the derivative so that you can get the peaks (inflection points)
    deriv = np.diff(result)

    # interpolate with cubic
    interpolfun = interp1d(time[1:], deriv, kind='cubic')

    dt = 0.001
    finetime = np.arange(int(time[1]), int(time.max()), dt)
    upsampled = interpolfun(finetime)

    w = np.logical_and(upsampled[:-1] < 0, upsampled[1:] < 0)
    idx = np.where(np.diff(w.astype(int)))

    inflection_times = np.asarray([round(t) for t in finetime[idx]], np.int) - 1  # nearest milisec
    inflection_values = filtered_ts[inflection_times]
    return time, time_series, filtered_ts, inflection_times, inflection_values


def bamplitude_OG(fileto):
    # bamplitude gets the filtered version of your scotopic ERG when you have given it an input of afilename
    # files = pd.read_csv(fileto)
    files = pd.read_csv(fileto, sep='\t', encoding = "ISO-8859-1")
    filess = np.asarray(files)
    to = filess[67:, 6:]
    # change to 67
    # 56 and 6
    # extract the column responsible for step number 2

    # the things inside this will determine the traces
    plt.figure
    # change this to analyze diff 1, 4, 7 is oS, 2, 5, 8 is od

    for i in columnx:

        # cut the first 20 ms and cut the last bits
        # x = to[20:200,0]
        # let time be the first column
        time = to[20:400, 0]
        y = to[20:400, i]
        yy = y.astype(float)

        # filter this step number 2 with an input of xn 
        xn = yy
        b, a = signal.butter(3, 0.05)
        zi = signal.lfilter_zi(b, a)
        z, _ = signal.lfilter(b, a, xn, zi=zi * xn[0])
        z2, _ = signal.lfilter(b, a, z, zi=zi * z[0])
        result = signal.filtfilt(b, a, xn)
        # result = np.asarray(result)
        print("Filtered data for column")
        print(i)

        # plot things
        # axes.set_ylim([-40000,150000])
        # plt.plot(derivative2)
        # fig2 = plt.figure()
        # ax1 = fig2.add_subplot(1,1,1)
        # ax2 = fig2.add_subplot(2,1,1)
        # ax1.plot(yy)
        # ax2.plot(result)



        # different ways to get peaks
        timeframe = time[100:300]

        # counter to get step level based on column number
        if i <= 2:
            steplevel = 1
        elif i > 2 and i <= 7:
            steplevel = 2
        elif i >= 8:
            steplevel = 3

        window = result[100:300]

        peak = signal.argrelmax(result)
        malaki = np.amax(result)
        maxm = argrelextrema(result, np.greater)
        malaki2 = np.amax(window)
        implicit = timeframe[np.where(window == malaki2)]
        # return things to me
        print("Here is the amplitude for step level")
        print(steplevel)
        print("The maximum is:")
        print(malaki)
        print("The peak is at")
        print(peak)
        print("The implicit time is")
        print(implicit)
        towrite = [fileto, steplevel, malaki, implicit]

        # graph
        plt.plot(time, yy)
        plt.plot(time, result)
        plt.plot(malaki, 'p')
        plt.plot(timeframe, window, ':')
        # plt.plot(timeframe, maxm, "^")
        plt.title(fileto + "SCOTOPIC_OG")
        axes = plt.gca()
        axes.set_xlim([0, 250])
        axes.set_ylim([-400000, 600000])

        # print added_array_column
        added_array_column = np.column_stack((time.flatten(), result.flatten()))
        added_array_column = np.asarray(added_array_column)
        # filtered[i].append(result)

        # print added_array_column

        # label1 = "This is the peak for steplevel"
        # label2


##means I changed it on july 17 2017
        ##biggerarray1.append(added_array_column)
        # after appending you have to reshape the data
        ##banana = np.vstack(towrite)
        ##biggerarray2.append(banana)

        # legend should be the step level
        # star = steplevel.astype(str)
        # plt.legend(star)
        # how do we append multiple rows to valules/

        # legend should be the step level
        # star = steplevel.astype(str)
        # plt.legend(star)
        # how do we append multiple rows to valules/


# main code here
answer = []
eye = input("Left or right eye?")
print("Your answer is %s." % eye) 
neweye = answer.append(eye)
if eye == "L":
    columnx = columna
    for x in allfiles:
        print("You have opened file")
        print(x)
        bamplitude_OG(x)
        # save each file as a text file and as a figure
        # after all the columns have processed, write one big output file and save the figure
        plt.savefig(x + '_OG_PHOTOPIC.png', format='png')
        biggestarray1.append((x, biggerarray1))
        biggestarray2.append((x, biggerarray2))

        # np.savetxt('OG_ANALYZED'+str(x)+'.csv', biggerarray, delimiter=',', fmt='%d')
        outputfile = '%s_filteredfile_OG.txt' % x
        print('Writing results of "%s" to: %s'% (x, outputfile)) 
        myfile = open(outputfile, 'w')
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(biggerarray1)
        myfile.close()
        # alternative --> np.savetxt(x+'filteredfile_OG.txt', biggerarray1, delimiter=",", fmt='%s')
        # output(x, sheet, yy, result, x, y, z)

        plt.show()
        plt.close()
elif eye == "R":
    columnx = columnb
    for x in allfiles:
        print("You have opened file")
        print(x)
        bamplitude_OG(x)
        # save each file as a text file and as a figure
        # after all the columns have processed, write one big output file and save the figure
        plt.savefig(x + '_OD_PHOTOPIC.png', format='png')
        biggestarray1.append((x, biggerarray1))
        # biggestarray2.append((x, biggerarray2))





        # np.savetxt('OG_ANALYZED'+str(x)+'.csv', biggerarray, delimiter=',', fmt='%d')
        outputfile = '%s_filteredfile_OD.txt' % x
        print('Writing results of "%s" to: %s') % (x, outputfile)
        myfile = open(outputfile, 'w')
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(biggerarray1)
        myfile.close()
        # alternative --> np.savetxt(x+'filteredfile_OG.txt', biggerarray1, delimiter=",", fmt='%s')
        # output(x, sheet, yy, result, x, y, z)

        plt.show()
        plt.close()

print("Writing results")
output2 = pd.DataFrame(biggestarray2)
output2.to_csv('FINALPEAKS_Scotopic.csv')
# print "the whole result is this"
# return biggerarray

# outputfile = '%s_filteredvalues.txt' %x
# print 'Writing results of "%s" to: %s' % (x, outputfile)
# output = open(outputfile,'w')
# output.write((biggerarray))
# output.close()
##outfile = open('tmp_table.data', 'w')
# for row in data:
#   for column in row:
#      outfile.write('\n')
# outfile.close()
