# Kalman filter example demo in Python

# A Python implementation of the example given in pages 11-15 of "An
# Introduction to the Kalman Filter" by Greg Welch and Gary Bishop,
# University of North Carolina at Chapel Hill, Department of Computer
# Science, TR 95-041,
# http://www.cs.unc.edu/~welch/kalman/kalmanIntro.html

# by Andrew D. Straw | and Marius

import numpy as num


def construct_test_data(z):
    z *= num.sin(num.linspace(0, num.pi*5., n_iter))
    z *= num.sin(num.linspace(0, num.pi*3., n_iter))
    z *= num.sin(num.linspace(0, num.pi/2., n_iter))
    z[:len(z)/2] += 1.
    return z


class Kalman():
    ''' A simple Kalman filter which can be applied recusively to continuous
    data.'''

    def __init__(self, P, R, Q):
        self.P = P
        self.R = R
        self.Q = Q

    def evaluate(self, new_sample, previous_estimate, weight=1.):
        ''' Calculate the next estimate, based on the
        *new_sample* and the *previous_sample*'''

        # time update
        xhatminus = previous_estimate
        Pminus = self.P + self.Q

        # measurement update
        K = Pminus / (Pminus + self.R) * weight
        self.P = (1-K) * Pminus
        return xhatminus + K*(new_sample-xhatminus)

    def evaluate_array(self, array):
        xhat = num.zeros(array.shape)
        for k in range(1, len(array)):
            new_sample = array[k]       # grab a new sample from the data set

            # get a filtered new estimate:
            xhat[k] = self.evaluate(
                new_sample=new_sample,
                previous_estimate=xhat[k-1])

        return xhat


if __name__ == '__main__':

    import sys
    import matplotlib.pyplot as plt
    # intial parameters

    '''
    python kalman.py [inputfile.txt]

    where inputfile is a two column ascii file with x and y values
    '''

    try:
        # read data from file
        infile = sys.argv[1]

        f = num.loadtxt(infile)
        x, y = f.T

        # remove pitches, which are below this value:
        pitch_threshold = -2000
        i_filtered = num.where(y>pitch_threshold)

        x = x[i_filtered]
        y = y[i_filtered]
        n_iter = len(x)
        y_true = None

    except IndexError:

        # if data cannot be read from file, create test_data:
        n_iter = 2000
        y_shift = -0.37727 # truth value (typo in example at top of p. 13 calls this z)
        y = num.ones(n_iter, dtype=num.float)*y_shift # correct values
        x = num.arange(n_iter)
        y_true = construct_test_data(y)

        #z = construct_test_data(z)      # These are the "measured" values.
        y = y_true+num.random.normal(0, 0.2, size=n_iter) # observations (normal about x, sigma=0.1)
    sz = (n_iter,) # size of array
    xhat = num.zeros(sz)             # a posteri estimate of x

    Q = 1e-4 # process variance

    # R small: responsive
    # R large: more smooth
    R = 0.03**2 # estimate of measurement variance, change to see effect

    # intial guesses
    P = 0.

    # create a *Kalman* filter object
    kalman = Kalman(P, R, Q)

    for k in range(1, n_iter):
        new_sample = y[k]       # grab a new sample from the data set

        # get a filtered new estimate:
        xhat[k] = kalman.evaluate(
            new_sample=new_sample,
            previous_estimate=xhat[k-1])

    plt.figure()
    if y_true is not None:
        plt.plot(x, y_true, color='g',label='truth value')
    plt.plot(x, y,'k+',label='noisy measurements')
    plt.plot(x, xhat,'b-',label='a posteri estimate')
    fig = plt.gcf()
    plt.text(0.5, 0.01, 'Q: %s, R: %s, P:%s'% (Q, R, P),
             transform=fig.transFigure)
    plt.legend()
    plt.title('Estimate vs. iteration step', fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('Voltage')

    plt.show()
