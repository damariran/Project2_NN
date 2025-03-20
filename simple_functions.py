import numpy as np
import matplotlib.pyplot as plt
from scipy import fft, signal

def simple_fft(x, y):
    sampling_frequency = len(x) / (x[-1] - x[0])
    x_fft = fft.fftfreq(len(x), 1/sampling_frequency)
    y_fft = fft.fft(y)
    positive_x_fft = x_fft[x_fft >= 0]  # returns only the positive frequency bins.
    y_fft = np.abs(y_fft)
    positive_y_fft = y_fft[x_fft >= 0]  # returns the right side magnitudes only.
    return positive_x_fft, positive_y_fft

def simple_plot(x, y, my_xlimit=None, sub_plot=None, my_legend=None, my_title=None, my_x_label=None, my_y_label=None, my_text=None, close=True):

    if sub_plot is None:
        pass
    else:
        plt.subplot(sub_plot[0],sub_plot[1],sub_plot[2])

    if len(x) == len(y):
        plt.plot(x,y, label= my_legend,alpha=0.5)
    else:
        print('The x and y vectors are not in the same length!')
        return None

    if my_xlimit is None:
        pass
    else:
        plt.xlim(my_xlimit[0], my_xlimit[1])

    if my_title is None:
        pass
    else:
        plt.title(my_title)

    if my_x_label is None:
        pass
    else:
        plt.xlabel(my_x_label)

    if my_y_label is None:
        pass
    else:
        plt.ylabel(my_y_label)

    if my_legend is None:
        pass
    else:
        plt.legend()

    if my_text is None:
        pass
    else:
        plt.text(1,1,my_text)

    plt.grid(True)

    if close:
        plt.show()
    return

def simple_scatter(x, y, my_xlimit=None, sub_plot=None, my_legend=None, my_title=None, my_x_label=None, my_y_label=None, close=True):

    if sub_plot is None:
        pass
    else:
        plt.subplot(sub_plot[0],sub_plot[1],sub_plot[2])

    if len(x) == len(y):
        plt.scatter(x,y, label= my_legend,alpha=0.5)
    else:
        print('The x and y vectors are not in the same length!')
        return None

    if my_xlimit is None:
        pass
    else:
        plt.xlim(my_xlimit[0], my_xlimit[1])

    if my_title is None:
        pass
    else:
        plt.title(my_title)

    if my_x_label is None:
        pass
    else:
        plt.xlabel(my_x_label)

    if my_y_label is None:
        pass
    else:
        plt.ylabel(my_y_label)

    if my_legend is None:
        pass
    else:
        plt.legend()

    plt.grid(True)

    if close:
        plt.show()
    return

def simple_butter_filter(x, y, my_low_pass_cutoff_frequency=None, filter_order=3):
    if my_low_pass_cutoff_frequency is None:
        my_low_pass_cutoff_frequency = 5 #[Hz]
    else:
        pass

    sampling_frequency = len(x) / (x[-1] - x[0]) # [Hz] the t[-1] means we get the last element of array t

    sos = signal.butter(filter_order, my_low_pass_cutoff_frequency, 'low', fs=sampling_frequency, output='sos') # 5th order Butterworth filter the output is the second-order-sections coefficients (SOS)

    filtered_signal = signal.sosfilt(sos, y) #  this function employ the SOS coefficients to the signal filter.

    return filtered_signal, sampling_frequency

def damped_oscillator(state, t, m, b, k):
    # this function is to be used in conduction with "odeint" function
    # "state0 = [1.0, 0.0]  # Initial position=1, velocity=0"
    # "solution = odeint(def damped_oscillator(state, t, m, b, k):, state0, t, args=(m, b, k))"
    #
    # This function gets an initial state ( position x and velocity v)
    # The vector "t" is meant to work with "odeint"' not in this function
    # and the mass "m"
    # the damping coefficient "b"
    # and the spring constant "k"
    # the function returns the velocity dx_vt and the acceleration dv_dt
    # for every state x, v for the "odeint" to find a solution.
    # the solution "odient" returns is the 'amplitude' vector to match the 'time' vector 't'.
    x, v = state
    dx_dt = v
    dv_dt = -(b / m) * v -(k / m) * x
    return [dx_dt, dv_dt]