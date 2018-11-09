#!/usr/bin/env python

__author__ = "Vivek M Agrawal"
__version__ = "1.0"
__email__ = "yatrie@gmail.com"

""" PhotoTracer
Python based step counting algorithm for automated analysis of single molecule
fluorescence data to infer the stoichiometry of binding events.

Usage: python3 photoTracer.py

Controls: User input parameters searchable by "# USER:"

Output: Writes output files for step transition data (csv formatted) and curve
tracing step-plots (multipage pdf)
"""

from decimal import *
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import pprint as pp

# function timer
import time
from functools import wraps

# multi-page pdf library
from matplotlib.backends.backend_pdf import PdfPages as pltPdf


def fn_timer(my_function):
    ''' decorator: function timer
    generic timer decorator for timing of functions, based on code snippet:
    http://www.marinamele.com/7-tips-to-time-python-scripts-and-control-memory-and-cpu-usage
    USAGE: Uncomment @fn_timer for function that needs to be timed
	'''
    @wraps(my_function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = my_function(*args, **kwargs)
        t1 = time.time()
        print ('Runtime for ' + my_function.__name__ + ' was: ' + str(round(1000.0*(t1-t0), 2)) + 'ms')
        return result
    return function_timer


#@fn_timer
def descDataframe(df, tOrig, pOrig):
	''' function: describe dataframe
	provides user with an overview of various statistical features of the input
	data and option to test the different smoothening functions for the particular
	data type (whether association or regular)

	!> calls {reviewSmooth -> smooth, }
	'''

	# print dataframe characteristics and basic phosphorescence stats
	print("Axes Characteristics: ", df.axes)

	print("Dataset Dimensions: ", df.shape[0], "*", df.shape[1])

	df_feat = dict(min = df['p'].min(), max = df['p'].max(), mad = df['p'].mad()\
		, mean = df['p'].mean(), var = df['p'].var(), std = df['p'].std()\
		, kurt = df['p'].kurt(), skew = df['p'].skew(), sem = df['p'].sem())

	pp.pprint(df_feat)

	# check with user if testing for appropriate smoothening window size is desired

	if (input('Plot test smoothening window sizes? (Y/y): ') == 'Y' or 'y'):
		# association plots have fewer data points so the window size needs to be smaller
		if (input('Are these association plots? (Y/y): ') == 'Y' or 'y'):
			wList = [3, 4, 5, 8, 10]
		else:
			wList = [5, 10, 15, 20, 25]

		# check with user if testing with different beta values is desired
		if (input('Test smoothening with recommended beta value (Y/y) or experiment with different standard values (any key): ') == 'Y' or 'y'):
			wBeta = [-0.5]
		else:
			# beta values for kaiser window to approximate other window functions
			# b		Window shape
			# -		------------------------------------
			# 0		Rectangular (working the best)
			# 5		Similar to a Hamming
			# 6		Similar to a Hanning
			# 8.6	Similar to a Blackman
			wBeta = [-0.5, 0.0, 5.0, 6.0, 8.6, 14.0]

		# display smoothening results for different test window sizes
		for wTest in wList:
			print('Plotting original and smooth values for window size of '+str(wTest))
			reviewSmooth(tOrig, pOrig, wTest, wBeta)


#@fn_timer
def reviewSmooth(tOrig, pOrig, wSize = 5, wBeta = [-0.5]):
	''' function: review smoothened waveform
	allows review of the result of different inbuilt window functions in numpy to
	identify the best/least lossy smoothening function based on the phosphorescence data

	!> calls {smooth -> {}, pltCompare -> {}, }
	'''

	# Note: After further consideration, 'kaiser' smoothening is found to be the
	# least lossy for the purpose of this tracing therefore defaulting to it

	# UNCOMMENT the following code to explore other window functions if
	# kaiser smoothening does not work well for your data
	## npWindows = ['flat', 'bartlett', 'blackman', 'hamming', 'hanning']
	## for w in npWindows:
		## pSmth = smooth(pOrig, wSize, w)
		## pltCompare(tOrig, pOrig, pSmth, wSize)

	for beta in wBeta:
		pSmth = smooth(pOrig, wSize, 'kaiser', beta)
		pltCompare(tOrig, pOrig, pSmth, wSize, beta)


#@fn_timer
def smooth(x, window_len = 10, window = 'kaiser', beta = 0):
	''' function: smoothen waveform
	computes the smoothened trace for the phosphorescence data using numpy window functions
	http://scipy-cookbook.readthedocs.io/items/SignalSmooth.html?highlight=signal%20smooth

	!> calls {, }
	'''

	s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]

	if window == 'flat':
		w = np.ones(window_len, 'd')
	elif window == 'kaiser':
		# smoothen pOrig using kaiser window function with different beta values
		# https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.kaiser.html
		w = eval('np.'+window+'(window_len, beta)')
	else:
		w = eval('np.'+window+'(window_len)')

	y = np.convolve(w/w.sum(), s, mode='valid')

	# NOTE: Due to convolving, length(output) != length(input), to correct this, return
	# y[(window_len/2-1):-(window_len/2)] instead of just y after convolving
	z = [float(Decimal("%.2f" % p)) for p in y][(window_len//2-1):-(window_len//2)]

	if len(z) > len(x):
		return z[:len(x)]
	elif len(z) == len(x):
		return z
	else:
		# rounding to two decimals places
		return [float(Decimal("%.2f" % p)) for p in y][:len(x)]


#@fn_timer
def pltCompare(tOrig, pOrig, pSmth, wSize = None, beta = None):
	''' function: compare waveforms
	plots the original and smoothened phosphorescence data against timestamp values
	with appropriate labels for beta and smoothening window sizes

	!> calls {, }
	'''

	# start the figure
	fig, ax = plt.subplots()

	# start the base figure with actual time series phosphorescence data
	plt.plot(tOrig, pOrig, label='Original values')

	# initialize empty string as holding pattern for figure legend
	s = ''
	if beta: # if the value for beta is provided
		s = s+' at beta '+str(beta)
	if wSize: # if the window size is provided
		s = s+' at window size '+str(wSize)

	# add smoothened time series phosphorescence data
	plt.plot(tOrig, pSmth, label='Smoothened values'+s)

	# show the resulting plot with legends
	legend = ax.legend(loc = 'upper right', shadow = True)
	plt.show()


#@fn_timer
def calcDiff(pStep = []):
	''' function: calcute neighbor distances
	computes the distance between neighboring phosphorescence step points and
	returns the diff as a new list. Also returns a list of indexes sortOrdrDiff
	on diff that would sort it in ascending order, this will allow picking the
	smallest distances first for smoothening operations

	!> calls {, }
	'''

	# initialize diff as list of zeros
	diff = [0]*len(pStep)

	# compute the distance (absolute) between successive steps
	for i in range(0, len(pStep)-1):
		diff[i] = abs(pStep[i] - pStep[i+1])

	# rounding to two decimals places
	diff = [float(Decimal("%.2f" % x)) for x in diff]

	# use np.argsort to returns indices that would sort the diff list
	sortOrdrDiff = np.argsort(diff)

	return diff, sortOrdrDiff


#@fn_timer
def spikeAttenuate(pStep = [], maxSpan = 10, minSpan = 0):
	''' function: attenuate spikes
	smoothens short spikes (peaks and valleys both) in the phosphorescence step
	data by examining the shortest spans of similar valued data points and then
	merging this spike with the nearest neighbor (in terms of absolute distance
	between the phosphorescence values of the two steps) and irrespective of the
	span of the 'to be merged to' spike (merging with longer span spike could
	result in faster convergence but might not be the best representation of the
	information contained in the spike)

	!> calls {calcWeights, }
	'''

	# break counter for run-away while loop
	brkCntr = 0

	# repeatedly search for spikes and normalize them to the nearest neighbor
	while (minSpan < maxSpan) and (brkCntr < len(pStep)//maxSpan):

		# update break counter for run-away while loop
		brkCntr += 1

		# re-calculate weights for the step values inside the while loop
		cntSmlr2Slf = calcWeights(pStep)

		# get the minimum span value to detect any spikes in the recalculated values
		minSpan = min(cntSmlr2Slf)

		# if the minimum span is greater than or equal to maximum span defined for a spike
		# break and return pStep
		if minSpan >= maxSpan:
			break

		# locate the index of the start of the spike
		i = cntSmlr2Slf.index(minSpan)
		# print(i, minSpan, len(pStep), len(cntSmlr2Slf), pStep[i-1], pStep[i+minSpan+1])
		# print(cntSmlr2Slf[i:i+minSpan+1], pStep[i:i+minSpan+1])

		# if at the beginning of the list, then default to the value on right
		# do this for this entire span of the spike, i.e. the range from i:(i+minSpan+1)
		if i == 0:
			pStep[i:(i+minSpan+1)] = [pStep[i+minSpan+1]]*(minSpan+1)
		# if at the end of the list, then default to value on left
		# do this for this entire span of the spike, i.e. the range from i:(i+minSpan+1)
		elif i+minSpan == len(pStep)-1:
			pStep[i:(i+minSpan+1)] = [pStep[i-1]]*(minSpan+1)
		else:
		# if somewhere in the middle of the list, then default to value closest in distance to the spike
		# do this for this entire span of the spike, i.e. the range from i:(i+minSpan+1)
			if abs(pStep[i] - pStep[i-1]) <= abs(pStep[i] - pStep[i+minSpan+1]):
				pStep[i:(i+minSpan+1)] = [pStep[i-1]]*(minSpan+1)
			else:
				pStep[i:(i+minSpan+1)] = [pStep[i+minSpan+1]]*(minSpan+1)

	# return the spike attenuated list
	pStep = [float(Decimal("%.2f" % x)) for x in pStep]
	return pStep


#@fn_timer
def calcWeights(pStep = []):
	''' function: cluster co-located samples
	for a list of smoothened phosphorescence values, returns a list of count of
	other co-located samples at the same value of phosphorescence as the sample
	at i'th position. This 'weight' helps with identifying stretches of co-located
	data that can be considered as falling under the same step level

	!> calls {, }
	'''

	# create a list of 0's representing count of data points similar to self
	cntSmlr2Slf = [0]*len(pStep)

	# compute count of other samples at the same phosphorescence value as running counter
	i, b = 0, 0 # b is a failsafe for the while loop running wild
	while i < len(pStep) and b < 2*len(pStep):
		j = i; b += 1
		while b < 2*len(pStep):
			b += 1
			if i == len(pStep)-1 or pStep[i] != pStep[i+1]:
				cntSmlr2Slf[j:i+1] = [i-j]*(i+1-j)
				#print(i, j, cntSmlr2Slf[j:i+1])
				i += 1
				break
			else:
				i += 1

	return cntSmlr2Slf[:len(pStep)]


#@fn_timer
def equalize(lbl = "asc", pStep = [], diff = [], sortOrdrDiff = [], cntSmlr2Slf = [], maxSteps = 5):
	''' function: eualize co-located samples
	for a list of phosphorescence values, iteratively performs the basis step conversion by
	combining co-located samples based on the following algorithm:
	1. For all samples in source data, take two co-located samples at a time and get
		the difference in phosphorescence levels between these samples
	2. Compute the min (the smallest transition between states) and max (the biggest
		leap or drop) of this running difference
	>>> Objective is to iteratively collect samples together where the state transition
		is the smallest delta. This should be safely be done for all samples that show a
		transition smaller than (max-min)//maxSteps
	3. Combine the collected samples together as the average value of the two samples
	4. Repeat for a fixed number of iterations

	!> calls {calcWeights, }
	'''

	# calculate max allowable transition delta
	mxDelta = max(diff)/float(maxSteps/2)

	# process in sorted order of diffs - smallest differences first
	for i in sortOrdrDiff:
		# ignore items with zero diff values - indicates equal values
		if diff[i] == 0.0:
			continue
		# for samples that are under the max allowable transition delta
		elif diff[i] <= mxDelta: # basically the value at i'th and i+1'th position differ
			# calculate the count of states with similar values
			cntSmlr2Slf = calcWeights(pStep)

			# get weights as count of similar valued samples
			wLeft = cntSmlr2Slf[i] + 1 # +1 since count does not include self
			wRight = cntSmlr2Slf[i+1] + 1 # +1 since count does not include self

			# compute weighted average
			pWeighted = ((wLeft*pStep[i])+(wRight*pStep[i+1]))/(wLeft + wRight)

			# update source sample values with the weighted average for all similar
			# items on the left and right
			pStep[(i-wLeft+1):(i+wRight+1)] = [pWeighted]*(wLeft + wRight)
		else: # delta greater than max allowable indicates reached end of safe looping
			pStep = [float(Decimal("%.2f" % x)) for x in pStep]
			return pStep

	# return the combined values list - this logically is unreachable code
	pStep = [float(Decimal("%.2f" % x)) for x in pStep]
	return pStep


#@fn_timer
def pullDown(pStep = [], maxSteps = 3):
	''' function: standardizes result sets
	after the step has been mostly fitted, pulls down the count of distinct steps to
	the value expected by the user, with the default at 3

	!> calls{calcWeights, }
	'''

	# calculate the weights of the samples based on co-location
	cntSmlr2Slf = calcWeights(pStep)

	# get all unique values of weights
	sWts = set(cntSmlr2Slf)

	# remove the edge values for start-at-zero or end-at-zero
	while 0 in sWts:
		sWts.remove(0)

	# if there are fewer than maxSteps requested, return the current steps
	if len(sWts) <= maxSteps:
		return pStep

	# otherwise iteratively merge nearby steps until there are only maxSteps left
	while len(sWts)-maxSteps > 0:
		# find the smallest weight from the list of all weights
		# zero weight was cleaned up in previous step
		x = min(sWts)

		# get the location if items corresponding to the lowest weights
		i = cntSmlr2Slf.index(x)

		# calculate current, left and right p values
		p, l, r = pStep[i], pStep[i-1], pStep[i+x+1]
		#print(x, sWts, i, p, l, r, pStep[i:i+x+1])

		# if near a zero values valley
		if l <= 0.0 or r <= 0.0:
			# if zero values at either end, this is a valid
			if l <= 0.0 and r <= 0.0:
				continue
			# else assign to value that is non-zero
			elif r <= 0.0:
				# convert all at peak to left leg
				pStep[i:i+x+1] = [l]*(x+1)
			elif l <= 0.0:
				# else convert all at peak to right leg
				pStep[i:i+x+1] = [r]*(x+1)
		# else if not near a zero valued valley, merge with nearest neighbor
		else:
			# if this is a genuine dip between peaks, skip it
			if p <= l and p <= r:
				continue
			# if distance between current peak and left leg is less that right leg
			elif abs(p-l) <= abs(p-r):
				# convert all at peak to left leg
				pStep[i:i+x+1] = [l]*(x+1)
			# else convert all at peak to right leg
			else:
				pStep[i:i+x+1] = [r]*(x+1)

		# recalculate the weights of the samples based on co-location
		cntSmlr2Slf = calcWeights(pStep)
		# get all unique values of weights, discard 0 value
		sWts = set(cntSmlr2Slf)
		while 0 in sWts: sWts.remove(0)

	return pStep


#@fn_timer
def pullDown2(pStep = [], maxSteps = 3):
	''' function: standardize result sets, compensating for valleys
	after the step has been mostly fitted, pulls down the count of distinct steps to
	the value expected by the user, with the default at 3
	Sudden and significant drops in phosphorescence values (referenced as valleys)
	should not be equalized and retained along with the steps

	!> calls{calcWeights, }
	'''

	# create a container for separating out genuine valleys
	lValley = [-1.0]*len(pStep)

	# calculate the weights of the samples based on co-location
	cntSmlr2Slf = calcWeights(pStep)
	# get all unique values of weights
	sWts = set(cntSmlr2Slf)
	# remove the edge values for start-at-zero or end-at-zero
	while 0.0 in sWts: sWts.remove(0.0)

	# iteratively merge nearby steps until there are only maxSteps left
	while len(sWts) - maxSteps > 0:
		# find the smallest weight from the list of all weights
		# zero weight was cleaned up in previous step
		minSpan = min(sWts)
		# get the location if items corresponding to the lowest weights
		i = cntSmlr2Slf.index(minSpan)

		# local variables to improve code readability
		stepSpan = minSpan + 1

		# if at the beginning of the list, then default to the value on right
		# do this for this entire span, i.e. the range from i:(i+minSpan+1)
		if i == 0:
			# if the initial set of values are 0.0, indicating start from zero,
			# retain those as valid valley values
			if pStep[i] == 0.0:
				lValley[i:(i+stepSpan)] = [0.0]*(stepSpan)
			# default pStep to the value on right
			pStep[i:(i+stepSpan)] = [pStep[i+minSpan+1]]*(stepSpan)
		# else if at the end of the list, then default to value on left
		# do this for this entire span, i.e. the range from i:(i+stepSpan)
		elif i+minSpan == len(pStep)-1:
			pStep[i:(i+stepSpan)] = [pStep[i-1]]*(stepSpan)
		# else if somewhere in the middle of the list, then
		else:
			# local variables to improve code readability
			pCurr, pLeft, pRight = pStep[i], pStep[i-1], pStep[i+minSpan+1]
			# check if a valley exists, when
			# a. the current value is lower than both of the neighbors
			if (pCurr < pLeft) and (pCurr < pRight):
				# b. and the vertical distance between the current steps and
				# nearer of the two neighbors is more than 50% of the overall
				# span of the steps in the entire range of phosphorescence values
				if (min(pLeft, pRight) - pCurr) > 0.5*(max(pStep)-min(pStep)):
					# retain the current step as valid valley
					lValley[i:(i+stepSpan)] = pStep[i:(i+stepSpan)]
			# merge the span with the smallest weight with the more prominent neighbor
			# do this for this entire span of the spike, i.e. the range from i:(i+stepSpan)
			if abs(pCurr - pLeft) <= abs(pCurr - pRight):
				pStep[i:(i+stepSpan)] = [pLeft]*(stepSpan)
			else:
				pStep[i:(i+stepSpan)] = [pRight]*(stepSpan)

		# re-calculate the weights of the samples based on co-location
		cntSmlr2Slf = calcWeights(pStep)
		# get all unique values of weights
		sWts = set(cntSmlr2Slf)
		# remove the edge values for start-at-zero or end-at-zero
		while 0.0 in sWts: sWts.remove(0.0)

	# a deep copy of the original pStep values, before including valleys
	pFinal = [p for p in pStep]
	# include valleys
	for i in range(0, len(lValley)):
		if lValley[i] >= 0.0:
			pStep[i] = lValley[i]

	return pStep, pFinal


#@fn_timer
def stepTransition(sDir, pStep):
	''' function: captures step transition data
	creates a list with start and end times corresponding to the state values
	representing transition between the steps generated on the original
	phosphorescence data, to be later exported to csv file

	!> calls {, }
	'''

	# initialize list with start and end times for the pValues
	dStep = [['pValue', 'start', 'end']]

	# iteratively compute the values at each step transition
	i, s = 0, 0
	while(i < len(pStep)-1):
		if pStep[i] != pStep[i+1]:
		# capture the start of step values
			dStep.append([pStep[i], s+1, i+1])
			s = i+1
		i += 1
	# capture the end of step values
	dStep.append([pStep[i], s+1, i+1])

	with open (sDir + "\\stepTransition.csv", 'w', newline='') as f:
		writer = csv.writer(f)
		for line in dStep:
			writer.writerow(line)

	return None


#@fn_timer
def saveData(sDir, tOrig, pOrig, pSmth, pFinal, lbl):
	''' function: saves result data
	creates a list with start and end times corresponding to the state values
	representing transition between the steps generated on the original
	phosphorescence data, to be later exported to csv file

	!> calls {, }
	'''

	# scale down phosphorescence data values if association data...
	if lbl == "asc":
		xFactor = 0.01
	else:
		xFactor = 1.0

	# and limit output data size to match time samples as extra security measure
	pOrigX = [float(i)*xFactor for i in pOrig[:len(tOrig)]]
	pSmthX = [float(i)*xFactor for i in pSmth[:len(tOrig)]]
	pFinalX = [float(i)*xFactor for i in pFinal[:len(tOrig)]]

	# pickle a list of lists with session data
	sData = [tOrig, pOrigX, pSmthX, pFinalX]
	with open (sDir + "\\sessionData.pickle", "wb") as f:
		pickle.dump(sData, f)

	# create list of list composing the session data to save as csv
	dOut = [['tOrig', 'pOrig', 'pSmth', 'pFinal']]

	# iteratively add the phosphorescence data values from different lists
	for i in range(0, len(tOrig)):
		dOut.append([tOrig[i], pOrigX[i], pSmthX[i], pFinalX[i]])

	# write to CSV file
	with open (sDir + "\\sessionData.csv", 'w', newline='') as f:
		writer = csv.writer(f)
		for line in dOut:
			writer.writerow(line)

	return None


def main():

	# USER: update the bulk csv file name / counter here
	bulkFile = "bulk_asc(7)"

	# USER: select appropriate scaling factor here default to 15.0
	# >> KEEP .0 at end to ensure decimal processing
	scalePhos = 15000.0		# phosphorescence value
	scaleTime = 3.5			# time sample distances

	# bulk processing on association data files
	lbl = "asc"

	# date value for folder structure
	toDay = time.strftime('%Y%m%d')

	# read csv into numpy Array with no header
	# assumes data is present in csv file named bulk_asc(N).csv
	# under the sub-directory named ascData
	npArray = np.loadtxt(".//ascData//" + bulkFile + ".csv", delimiter=',')

	# process for all data in the file
	# USER: If this breaks at n-th file, start from n+1
	#for fn in range(N, npArray.shape[1]):
	for fn in range(8, npArray.shape[1]):
		# notify user of the current operations
		print("Working on file # {} of {} files in the bulk csv file {}...".format(fn, npArray.shape[1], bulkFile))

		# create the directory structure to save the step transition plot
		sDir = os.getcwd() + "\\outputData\\" + bulkFile + "\\" + str(toDay) + "\\"
		if not os.path.exists(sDir):
			os.makedirs(sDir)

		# create a multi-page object for each input data file
		pdfFig = pltPdf(sDir + "file-" + str(fn) + ".pdf")

		# collect the base phosphorescence values in a list and scale by multiplication factor
		pOrig = npArray[:, fn] * scalePhos

		# default time values to frame counts in association data
		tOrig = [x for x in range(0, len(pOrig))]

		# prepare data and parameters for smoothening operations
		# recommended values for association file:
		# ------------------------------------------------------------------------------------------
		# maxSteps = 2 to 5, 4 preferred for 3 steps
		# wSize = 3 to 5, 3 preferred for association files - small samples and 10 for regular data
		# beta = -0.5 to 0.0
		# maxSmth = 2 to 5, depending on how jagged the data set is / many outliers or peaks
		# maxIter = 5 to 25, depending on how close or distant the different data samples are
		# ------------------------------------------------------------------------------------------
		maxSteps, beta = 4, -0.5
		for wSize in [5, 7, 11]:
			for maxSmth in [1, 2, 3, 5]:
				for maxIter in [3, 5, 10]:
					#print("Processing file-{} with maxSteps={}, beta={}, wSize={}, maxSmth={}, maxIter={}...".format(fn, maxSteps, beta, wSize, maxSmth, maxIter))

					# deep copy original phosphorescence values to a smoothening vector for reuse in loops without changing original pValues
					pSmth = [x for x in pOrig]

					# DO NOT OVERSMOOTHEN HERE, USE FEWER ITERATIONS
					# 1. smoothen using numpy window functions
					# iterate smoothening over the smoothening vector (default at 1)
					for i in range(0, maxSmth):
						pSmth = smooth(pSmth, wSize, 'kaiser', beta)

					# 2. smoothen using weighted equalizations based on relative delta between time series samples
					for i in range(0, maxIter):
						# calculate the diff or delta, sorted order of diff, and count of states with similar values
						diff, sortOrdrDiff = calcDiff(pSmth)
						cntSmlr2Slf = calcWeights(pSmth)
						# equalize the smoothened values
						pSmth = equalize(lbl, pSmth, diff, sortOrdrDiff, cntSmlr2Slf)

						# calculate the diff or delta, sorted order of diff, and count of states with similar values
						diff, sortOrdrDiff = calcDiff(pSmth)
						cntSmlr2Slf = calcWeights(pSmth)
						# 3. smoothen the spikes in increasing size of maximum width or span for the spike
						# ranging from 2 to 15 samples wide spikes
						for maxSpan in range(2, 5):
							pSmth = spikeAttenuate(pSmth, maxSpan)

					# 3. pull down peaks to the maximum allowed steps
					#pSmth = pullDown(pSmth, maxSteps)
					pSmth, pFinal = pullDown2(pSmth, maxSteps)

					# default first values for association plots to 0.0
					pSmth[0] = 0.0
					pFinal[0] = 0.0

					# USER: COMMENT FROM HERE if not wanting to store transition data

					# create the directory structure to save the session data
					sDirData = os.getcwd() + "\\outputData\\" + bulkFile + "\\" + str(toDay) + "\\file-" + str(fn) + "\\wSize-" + str(wSize) + "_maxSmth-" + str(maxSmth) + "_maxIter-" + str(maxIter)

					# create the directory structure to save the session data
					if not os.path.exists(sDirData):
						os.makedirs(sDirData)

					'''
					# scale down pFinal data by multiplication factor
					pFinal = [float(i)/scalePhos for i in pFinal]
					'''

					# compute state transitions as list of lists with pValue
					# and corresponding start and end times for the observations
					stepTransition(sDirData, pFinal)

					# save session data as pickled python list of list for re-ingestion and user friendly csv files
					saveData(sDirData, tOrig, pOrig, pSmth, pFinal, lbl)

					# USER: COMMENT TILL HERE if not wanting to store transition data

					# reduce noise in the phosphorescence data for charting
					pOrigPlot = [int(i) for i in pOrig][:len(tOrig)]
					pSmthPlot = [int(i) for i in pSmth][:len(tOrig)]
					# compensate for the sampling rate in association data
					tOrigPlot = [float(Decimal("%.1f" % (i * scaleTime))) for i in tOrig]

					# generate the step transition plot
					fig, ax = plt.subplots()
					fig.suptitle('Photon Intensity by Elapsed Time', fontsize=14, fontweight='bold')
					ax.set_xlabel('Time (sec)')
					ax.set_ylabel('Intensity (counts/sec)')
					ax.set_title('maxSteps={}, wSize={}, beta={}, maxSmth={}, maxIter={}'.format(maxSteps, wSize, beta, maxSmth, maxIter), fontsize=8)
					plt.plot(tOrigPlot, pOrigPlot, label='Input function', color='k', linewidth='0.65')
					plt.plot(tOrigPlot, pSmthPlot, label='Output function', color='k', linewidth='2.15')
					plt.margins(0)
					#legend = ax.legend(loc = 'upper right', fontsize=8, shadow = False)

					# save - append the plot to multi-page PDF
					pdfFig.savefig(fig)

					# close pyplot figure to release memory
					plt.close(fig)

		# close the multi-page object at the end of the iteration for each input data file
		pdfFig.close()

if __name__ == "__main__":
	main()
