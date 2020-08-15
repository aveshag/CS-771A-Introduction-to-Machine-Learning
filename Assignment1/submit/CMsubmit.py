import numpy as np
import random as rnd
import time as tm


def getObj( X, y, w, b ):
	hingeLoss = np.maximum( 1 - np.multiply( (X.dot( w ) + b), y ), 0 )
	return 0.5 * w.dot( w ) + C * hingeLoss.dot( hingeLoss )

def getCyclicCoord( currentCoord, n ):
    if currentCoord >= n-1 or currentCoord < 0:
        return 0
    else:
        return currentCoord + 1

def getRandCoord( currentCoord, n ):
    return rnd.randint( 0, n-1 )

def getRandpermCoord( currentCoord , n ):
    global randperm, randpermInner
    if randpermInner >= n-1 or randpermInner < 0 or currentCoord < 0:
        randpermInner = 0
        randperm = np.random.permutation( n )
        return randperm[randpermInner]
    else:
        randpermInner = randpermInner + 1
        return randperm[randpermInner]

C=1
randperm = []
randpermInner = -1

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def CMsolver( X, y, C, horizon ):

	primalObjValSeries = np.zeros( (horizon//50000,) )
	timeSeries = np.zeros( (horizon//50000,) )

	(n, d) = X.shape
	t = 0
	totTime = 0
	
	# w is the normal vector and b is the bias
	# These are the variables that will get returned once timeout happens
	w = np.zeros( (d,) )
	b = 0
	tic = tm.perf_counter()
################################
#  Non Editable Region Ending  #
################################
	
	# You may reinitialize w, b to your liking here
	# You may also define new variables here e.g. eta, B etc
	# Find the unconstrained new optimal value of alpha_i    
	# Initialize model as well as some bookkeeping variables
	alpha = C * np.ones( (y.size,) )
	alphay = np.multiply( alpha, y )
	# Initialize the model vector using the equations relating primal and dual variables
	w = X.T.dot( alphay )
	# Recall that we are imagining here that the data points have one extra dimension of ones
	# This extra dimension plays the role of the bias in this case
	b = alpha.dot( y )
	# Calculate squared norms taking care that we are appending an extra dimension of ones
	normSq = np.square( np.linalg.norm( X, axis = 1 ) ) + 1
	# We have not made any choice of coordinate yet
	
	randperm = np.random.permutation( y.size )
	randpermInner = -1
	 
	i=-1
	j=0
################################
# Non Editable Region Starting #
################################
	for t in range( horizon ):

		tic = tm.perf_counter()
		
################################
#  Non Editable Region Ending  #
################################
		i = getRandpermCoord( i , n)
		x = X[i,:]
		#print("A")

		newAlphai = (1 - y[i] * (x.dot(w) + b) + alpha[i] * normSq[i]) / (1/(2*C) + normSq[i])
		if newAlphai < 0:
			newAlphai = 0

		w = w + (newAlphai - alpha[i]) * y[i] * x
		b = b + (newAlphai - alpha[i]) * y[i]

		alpha[i] = newAlphai
		toc = tm.perf_counter()
		totTime = totTime + (toc - tic)

		if t%50000==0:
			primalObjValSeries[j] = getObj( X, y, w, b )
			timeSeries[j] = totTime
			j=j+1
		# Write all code to perform your method updates here within the infinite while loop
		# The infinite loop will terminate once timeout is reached
		# Do not try to bypass the timer check e.g. by using continue
		# It is very easy for us to detect such bypasses - severe penalties await
		
		# Please note that once timeout is reached, the code will simply return w, b
		# Thus, if you wish to return the average model (as we did for GD), you need to
		# make sure that w, b store the averages at all times
		# One way to do so is to define two new "running" variables w_run and b_run
		# Make all GD updates to w_run and b_run e.g. w_run = w_run - step * delw
		# Then use a running average formula to update w and b
		# w = (w * (t-1) + w_run)/t
		# b = (b * (t-1) + b_run)/t
		# This way, w and b will always store the average and can be returned at any time
		# w, b play the role of the "cumulative" variable in the lecture notebook
		# w_run, b_run play the role of the "theta" variable in the lecture notebook
		
	return (primalObjValSeries, timeSeries) # This return statement will never be reached
