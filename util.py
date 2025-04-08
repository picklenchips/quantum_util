"""
Defines useful utility functions and constants. Run printModule(util) after importing to see dirs
- CVALS: object of physics constants
- printModule
- timeIt
- binarySearch
- linearInterpolate
- uFormat
- RSquared
- NRME

"""

import numpy as np
from glob import glob
from cycler import cycler
import matplotlib as mpl
from matplotlib.colors import to_rgba
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# import matplotlib.lines as mlines
# import plotly.graph_objects as go

import sys, time, os

from scipy import special  # for voigt function
from scipy.optimize import curve_fit
import functools
from itertools import zip_longest
from typing import Optional, Sequence, Iterable

# -- CONSTANTS -- #
DATADIR = "/Users/benkroul/Documents/Physics/Data/"
SAVEDIR = "/Users/benkroul/Documents/Physics/plots/"
SAVEEXT = ".png"
FIGSIZE = (10, 6)
TICKSPERTICK = 5
FUNCTYPE = type(sum)


class justADictionary:
    def __init__(self, my_name):
        self.name = my_name
        self.c = 2.99792458  # 1e8   m/s speed of lgiht
        self.h = 6.62607015  # 1e-34 J/s Plancks constant,
        self.kB = 1.380649  # 1e-23 J/K Boltzmanns constant,
        self.e = 1.60217663  # 1e-19 C electron charge in coulombs
        self.a = 6.02214076  # 1e23  /mol avogadros number
        self.Rinf = 10973731.56816  # /m rydberg constant
        self.G = 0.0  # m^3/kg/s^2 Gravitational constant
        self.neutron_proton_mass_ratio = 1.00137842  # m_n / m_p
        self.proton_electron_mass_ratio = 1836.15267343  # m_p / m_e
        self.wien = 2.89777  # 1e-3  m*K  peak_lambda = wien_const / temp

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        ", ".join([i for i in self.__dir__() if i != "name"])
        return self.name


CVALS = justADictionary("Useful Physics constants, indexed in class for easy access")

# IBM's colorblind-friendly colors
#           |   Red  |   Blue  |  Orange |  Purple | Yellow  |   Green |   Teal  | Grey
hexcolors = [
    "DC267F",
    "648FFF",
    "FE6100",
    "785EF0",
    "FFB000",
    "009E73",
    "3DDBD9",
    "808080",
]
mpl.rcParams["axes.prop_cycle"] = cycler("color", [to_rgba("#" + c) for c in hexcolors])

#: dictionary of metric prefixes for formatting numbers
METRIC_PREFIXES = {
    -24: "y",  # yotto
    -21: "z",  # zepto
    -18: "a",  # atto
    -15: "f",  # femto
    -12: "p",  # pico
    -9: "n",  # nano
    -6: "µ",  # micro
    -3: "m",  # milli
    0: "",  #
    3: "k",  # kilo
    6: "M",  # mega
    9: "G",  # giga
    12: "T",  # tera
    15: "P",  # peta
    18: "E",  # exa
    21: "Z",  # zetta
    24: "Y",  # yotta
}
# ====== TYPING CONSTANTS ====== #
#: suffix TYPES is used with `isinstance(x, TYPES)` to check if x is of a certain type
#: suffix TYPE is used with type-hinting python functions

INT_TYPES = (int, np.integer)
INT_TYPE = int | np.integer
FLOAT_TYPES = (float, np.floating)
FLOAT_TYPE = float | np.floating
REAL_TYPES = (*INT_TYPES, *FLOAT_TYPES)
REAL_TYPE = INT_TYPE | FLOAT_TYPE
COMPLEX_TYPES = (complex, np.complexfloating)
COMPLEX_TYPE = complex | np.complexfloating
NUMBER_TYPES = (*REAL_TYPES, *COMPLEX_TYPES)
NUMBER_TYPE = REAL_TYPE | COMPLEX_TYPE


def savefig(title):
    plt.savefig(SAVEDIR + title + SAVEEXT, bbox_inches="tight")


# -- GENERAL FUNCTIONS -- #
def printModule(module):
    """print a module AFTER IMPORTING IT"""
    print("all imports:")
    numListedPerLine = 3
    i = 0
    imported_stuff = dir(module)
    lst = []  # list of tuples of thing, type
    types = []
    for name in imported_stuff:
        if not name.startswith("__"):  # ignore the default namespace variables
            typ = str(type(eval(name))).split("'")[1]
            lst.append((name, typ))
            if typ not in types:
                types.append(typ)
    for typ in types:
        rowstart = "  " + typ + "(s): "
        i = 0
        row = rowstart
        for id in lst:
            if id[1] != typ:
                continue
            i += 1
            row += id[0] + ", "
            if not i % numListedPerLine:
                print(row[:-2])
                row = rowstart
        if len(row) > len(rowstart):
            print(row[:-2])
        i += numListedPerLine


def timeIt(_func=None, *, repeat=1, return_time=False, print_time=False):
    """Wrapper to time a function with various return options.

    Kwargs
    ------
    repeat: int
        will average the function runtime by repeating it `repeat` times
    return_time: bool
        returns (ret, time) if True, else just ret
    print_time: bool
        prints time to stdout if True

    .. note::
        Arguments MUST be given as keyword arguments to a wrapper function, the first arg is always assumed to be the function that is being wrapped.

    Examples
    --------
    .. code-block:: python

        repeat: int = 10
        return_time: bool = True
        @timeIt(repeat, return_time)
        def f(*args, **kwargs):
            ...
            return ret

    >>> print(f(*args, **kwargs))
    (str(ret), mean_time_in_seconds)

    .. code-block:: python

            repeat: int = 10
            return_time: bool = True
            @timeIt(repeat=repeat, return_time=return_time)
            def f(*args, **kwargs):
                ...
                return ret

    >>> ret = f(*args, **kwargs)
    f ran in 0.123s, averaged over 10 times

    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            total_time = 0
            for i in range(repeat):
                # changed from time.clock_gettime()
                # for windows compatibility
                start_time = time.time()
                ret = func(*args, **kwargs)
                end_time = time.time()
                total_time += end_time - start_time
            mean_time = total_time / repeat
            if print_time:
                if repeat > 1:
                    print(
                        func.__name__
                        + f"took {uFormat(mean_time, metric=True)}s, avg of {repeat}"
                    )
                else:
                    print(func.__name__ + f"took {uFormat(mean_time, metric=True)}s")
            if return_time:
                return ret, mean_time
            return ret

        return wrapper

    if _func is None:
        return decorator
    return decorator(_func)


def binarySearch(X_val, X, decreasing=False):
    """
    For sorted X, returns index i such that X[:i] < X_val, X[i:] >= X_val
     - if decreasing,returns i such that    X[:i] > X_val, X[i:] <= X_val
    """
    l = 0
    r = len(X) - 1
    # print(f"searching for {X_val}, negative={negative}")
    m = (l + r) // 2
    while r > l:  # common binary search
        # print(f"{l}:{r} is {X[l:r+1]}, middle {X[m]}")
        if X[m] == X_val:  # repeat elements of X_val in array
            break
        if decreasing:  # left is always larger than right
            if X[m] > X_val:
                l = m + 1
            else:
                r = m - 1
        else:  # right is always larger than left
            if X[m] < X_val:
                l = m + 1
            else:
                r = m - 1
        m = (l + r) // 2
    if r < l:
        return l
    if m + 1 < len(X):  # make sure we are always on right side of X_val
        if X[m] < X_val and not decreasing:
            return m + 1
        if X[m] > X_val and decreasing:
            return m + 1
    if X[m] == X_val:  # repeat elements of X_val in array
        if decreasing:
            while m > 0 and X[m - 1] == X_val:
                m -= 1
        elif not decreasing:
            while m + 1 < len(X) and X[m + 1] == X_val:
                m += 1
    return m


# linear interpolate 1D with sorted X
def linearInterpolate(x, X, Y):
    """example: 2D linear interpolate by adding interpolations from both
    -"""
    i = binarySearch(x, X)
    if i == 0:
        i += 1  # lowest ting, interpolate backwards
    m = (Y[i] - Y[i - 1]) / (X[i] - X[i - 1])
    b = Y[i] - m * X[i]
    return m * x + b


def intersect_2pts(X, Y1, Y2):
    """returns the intercept of a line given two points
    assuming there is some intersection there"""
    m1 = (Y1[1] - Y1[0]) / (X[1] - X[0])
    m2 = (Y2[1] - Y2[0]) / (X[1] - X[0])
    b1 = Y1[0] - m1 * X[0]
    b2 = Y2[0] - m2 * X[0]
    x = (b1 - b2) / (m2 - m1)
    y = m1 * x + b1
    return (x, y)


# - ---- -STATS FUNCTIONS


def RSquared(y, model_y):
    """R^2 correlation coefficient of data"""
    n = len(y)
    # get mean
    SSR = SST = sm = 0
    for i in range(n):
        sm += y[i]
    mean_y = sm / n
    for i in range(n):
        SSR += (y[i] - model_y[i]) ** 2
        SST += (y[i] - mean_y) ** 2
    return 1 - (SSR / SST)


def NRMSE(y, model_y, normalize=True):
    """Root mean squared error, can be normalized by range of data"""
    n = len(y)
    sm = 0
    min_y = y[0]
    max_y = y[0]
    for i in range(n):
        if y[i] < min_y:
            min_y = y[i]
        if y[i] > max_y:
            max_y = y[i]
        sm += (y[i] - model_y[i]) ** 2
    y_range = max_y - min_y
    val = np.sqrt(sm / n)
    if normalize:
        val = val / y_range
    return val


# ----- TEXT MANIPULATION ----- #

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# method to return the string form of an integer (0th, 1st, 2nd, 3rd, 4th...)
Ith = lambda i: str(i) + (
    "th"
    if (abs(i) % 100 in (11, 12, 13))
    else ["th", "st", "nd", "rd", "th", "th", "th", "th", "th", "th"][abs(i) % 10]
)


def arrFromString(data, col_separator="\t", row_separator="\n"):
    """Return numpy array from string
    - great for pasting Notion tables into np array"""
    x = []
    L = 0
    for row in data.split(row_separator):
        if len(row):  # ignore any empty rows
            entries = row.split(col_separator)
            newL = len(entries)
            if L != 0 and newL != L:
                print(f"Rows have different lengths {L} and {newL}:")
                print(x)
                print(entries)
            L = newL
            x.extend(entries)
    return np.reshape(np.array(x, dtype="float64"), (-1, L))


def align_numbers(
    numbers: np.ndarray | Sequence[str], inplace=False, figs=3
) -> list[str]:
    """
    Aligns numbers in a list of strings so that the decimal points are in the same column and all numbers have the same string length by adding spaces before and after the numbers.

    Args
    ----
    numbers: Iterable[str]
        list of strings to align
    inplace: bool
        if True, modifies the original list of strings in place
    """
    if isinstance(numbers, np.ndarray):
        numbers = numbers.tolist()
    if not isinstance(numbers, (list,)):
        raise TypeError("numbers must be a list or tuple of strings")
    if not all(isinstance(n, str) for n in numbers):
        raise TypeError("numbers must be a list or tuple of strings")
    # find the maximum length of the strings
    lengths = np.array([len(n) for n in numbers])
    # find the index of the decimal point in each string
    decimal_indices = np.empty(len(numbers), dtype=int)
    for i, n in enumerate(numbers):
        if decimal_indices[i] == -1:
            for char in [".", "(", "e", " "]:
                if (j := n.find(char)) != -1:
                    decimal_indices[i] = j
                    break
        if decimal_indices[i] == -1:
            decimal_indices[i] = min(figs, len(n))
    # find max chars before and after decimal point, and align
    befores = max(decimal_indices) - decimal_indices
    decimal_indices_r = lengths - decimal_indices
    afters = max(decimal_indices_r) - decimal_indices_r
    aligned_numbers = []
    # add spaces before and after the number to align it
    for i in range(len(numbers)):
        aligned_number = (
            " " * befores[i] + numbers[i] + " " * (afters[i] if afters[i] >= 0 else 0)
        )
        if inplace:
            numbers[i] = aligned_number
        else:
            aligned_numbers.append(aligned_number)
    return aligned_numbers if not inplace else numbers


def uFormat(
    number: REAL_TYPE | str | Iterable[REAL_TYPE | str],
    uncertainty: Optional[str | REAL_TYPE | Iterable[REAL_TYPE | str]] = 0.0,
    figs: INT_TYPE | Iterable[INT_TYPE] = 4,
    shift: INT_TYPE | Iterable[INT_TYPE] = 0,
    math: bool | Iterable[bool] = False,
    metric: bool | Iterable[bool] = False,
    percent: bool | Iterable[bool] = False,
    metric_space=True,
    debug=False,
    join_string=", ",
    align_all=False,
    align_function=align_numbers,
) -> str:
    r"""
    Formats a number with its uncertainty, according to `PDG §5.3 <https://pdg.lbl.gov/2011/reviews/rpp2011-rev-rpp-intro.pdf#page=13>`_ rules. Also handles other formatting options, see below.

    Args
    ----
    number:
        the value to format, converted to a float
    uncertainty:
        the absolute uncertainty (stddev) in the value
        * if zero, will format number according to `figs`
    figs:
        when `uncertainty == 0`, formats number to this # of significant figures. default 4.
    shift:
        shift the resultant number to a higher/lower digit expression
        * i.e. if number is in Hz and you want a string in GHz, specify `shift = 9`
        * likewise for going from MHz to Hz, specify `shift = -6`
    math:
        LaTeX format: output "X.XX\text{e-D}" since "-" will render as a minus sign otherwise
        .. note:: to print correctly, you must make sure the string is RAW r''
    metric:
        will correctly label number using metric prefixes (see :py:dict:`METRIC_PREFIXES`) based on scale of number
        * ex: `1.23e6 -> 1.23 M`, or `0.03 -> 30 m`
        .. note:: make sure to put a unit at the end of the string for real quantities!
    percent:
        will treat number as a percentage and add a "%" at the end, automatically
        shifting the number by 2 to the left
        * ex: `0.03 -> 3%` and `3 -> 300%`
    join_string:
        if ANY of the arguments are Iterable, will map uFormat to the arguments as appropriate and
        join them with join_string
        * e.g. number = (1.0, 2.0, 3.0), uncertainty = (0.1, 0.2, 0.3), join_string = "." will return "1(1).2(2).3(3)"
    align_all:
        if True, will align all numbers (iterable arguments given) so that decimal places are in the same column by adding spaces before and after the number
        * this is useful for printing tables of numbers

    Returns
    -------
    str: the formatted number as a string

    Examples
    --------
    >>> uFormat(0.01264, 0.0023)
    '0.013(2)'
    >>> uFormat(0.001234, 0.00067)
    '1.234(67)e-3'
    >>> uFormat(0.0006789, 0.000023, metric=True)
    '679(23) µ'
    >>> uFormat(0.306, 0.02, percent=True, math=True)
    '31(2)\%'
    >>> uFormat(32849, 5000, metric=True, math=True)
    '33(5)\text{ k}'
    >>> uFormat(0.00048388, figs=3)
    '4.84e-4'
    >>> uFormat((0.001, 0.002), metric=((True, False),False), percent=(False, True))
    '1 µ, 1e-3, 0.2%'

    Notes
    ------
    * if both `metric` and `percent` are specified, will raise ValueError, as these formatting options conflict!
    * able to handle any arguments as (nested) iterables, if given.

        * this will copy the last value of shorter-length arguments to match the longest-length argument, so be careful!
        * for example, `metric=((True, False),)` and `percent=(False, True)` will raise an error because this will map to `metric=((True, False),(True, False))` and `percent=(False, True)`, which contradicts on the 3rd call.

    * best way to specify different metric/percent formatting is to use same-length iterables.

        * e.g. metric=(False, True, True), percent=(False, False, True)

    """
    # if any of the arguments are iterable, apply uFormat to each element
    # and join them with join_string
    # if join_string is not specified and arguments are iterable, will raise a value error
    # this is actually also able to handle nested iterables, if given.

    kwargs_per_iter = []
    kwargs = {
        "number": number,
        "uncertainty": uncertainty,
        "figs": figs,
        "shift": shift,
        "math": math,
        "metric": metric,
        "percent": percent,
    }
    kwargs["debug"] = debug
    for arg, argval in kwargs.items():
        if hasattr(argval, "__iter__") and not isinstance(argval, str):
            for i, v in enumerate(argval):
                if i >= len(kwargs_per_iter):
                    kwargs_per_iter.append({})
                kwargs_per_iter[i][arg] = v
    # if ANY of the arguments are iterable, apply uFormat over each argument

    if len(kwargs_per_iter) > 0:
        if debug:
            print(kwargs_per_iter)
        # place single args into the first dictionary
        ret = []
        for i in range(len(kwargs_per_iter)):
            kwargs.update(kwargs_per_iter[i])
            ret.append(uFormat(**kwargs))
        # format the output strings if align is true
        if align_all:
            if isinstance(figs, Iterable):
                figs = max(figs)
            align_function(ret, inplace=True, figs=int(figs))
        return join_string.join(ret)
        # else, just apply uFormat to the single number
    assert isinstance(figs, int), "figs must be an integer!"
    assert isinstance(shift, int), "shift must be an integer!"
    if metric and percent:
        raise ValueError(
            "Cannot have both metric and percent formatting! See docstring for formatting info."
        )
    num = str(number)
    err = str(uncertainty)

    if figs < 1:
        figs = 1
    ignore_uncertainty = not uncertainty  # UNCERTAINTY ZERO: IN SIG FIGS MODE

    is_negative = False  # add back negative later
    if num[0] == "-":
        num = num[1:]
        is_negative = True
    if err[0] == "-":
        err = err[1:]

    # ni = NUM DIGITS to the RIGHT of DECIMAL
    # 0.00001234=1.234e-4 has ni = 8, 4 digs after decimal and 4 sig figs
    # 1234 w/ ni=5 corresponds to 0.01234
    # 1234 w/ ni=-4 corresponds to 12340000 = 1234e7, n - ni - 1 = 7
    ni = ei = 0

    def get_raw_number(num: str) -> tuple[str, int]:
        """returns raw_num, idx where raw_num contains all significant figures of number and idx is the magnitude of the rightmost digit of the number"""
        found_sigfig = False
        found_decimal = False
        index_right_of_decimal = 0
        raw_num = ""
        # scientific notation
        if "e" in num:
            ff = num.split("e")
            num = ff[0]
            index_right_of_decimal = -int(ff[1])
        for ch in num:
            if found_decimal:
                index_right_of_decimal += 1
            if not found_sigfig and ch == "0":  # dont care ab leading zeroes
                # TODO: any scenario in which we want to conserve leading zeros?
                continue
            if ch == ".":
                found_decimal = True
                continue
            if not ch.isdigit():
                return "?", 0
            found_sigfig = True
            raw_num += ch
        return raw_num, index_right_of_decimal

    def round_to_idx(string: str, idx: int) -> str:
        """rounds string to idx significant figures"""
        if idx >= len(string):
            return string
        if int(string[idx]) >= 5:
            return str(int(string[:idx]) + 1)
        return string[:idx]

    # get raw numbers
    raw_num, ni = get_raw_number(num)
    if raw_num == "?":
        return str(num)
        # raise ValueError(f"input number {number} is not a valid number!")
    n = len(raw_num)
    if n == 0:  # our number contains only zeros!
        return "0"
    raw_err, ei = get_raw_number(err)
    if raw_err == "?":
        print(f"input error {uncertainty} is not a valid number, continuing anyways...")
    m = len(raw_err)
    if m == 0:  # our error contains only zeros!
        ignore_uncertainty = True
    # 0.01234 -> '1234', (4, 5)
    if debug and ignore_uncertainty:
        print("ignoring uncertainty!")
    #
    # round error according to PDG rules
    # consider only three significant figures of error
    #
    if m > 3:
        ei += 3 - m
        raw_err = raw_err[:3]
    if m > 1:
        # have 3 digits in error, round correctly according to PDG
        if m == 2:
            raw_err += "0"
            ei += 1
        # round error correctly according to PDG
        err_three = int(raw_err)
        # 123 -> (12.)
        if err_three < 355:
            raw_err = round_to_idx(raw_err, 2)
            ei -= 1
        # 950 -> (10..)
        elif err_three > 949:
            raw_err = "10"
            ei -= 2
        # 355 -> (4..)
        else:
            raw_err = round_to_idx(raw_err, 1)
            ei -= 2
        m = len(raw_err)
    if ignore_uncertainty:
        # raw_err = ""
        # m = 0
        # round to sig figs!!
        assert m == 0
        assert not raw_err
        ei = min(ni, ni - n + figs)
    # shift numbers, if specified
    if percent:
        shift += 2
    ni -= shift
    ei -= shift
    #
    # round number according to error
    # n = number of significant digits in number
    # ni = magnitude of rightmost digit in number
    # mag_num = magnitude of leftmost digit in number
    # eg: 0.0023 -> 2.3e-3, n=2, ni=4, mag_num=-3
    # place of 1st digit in number (scientific notation of number)
    mag_num = n - ni - 1
    mag_err = m - ei - 1
    d = ni - ei
    # format number according to metric prefixes
    end = ""
    if debug:
        print("pre-metric:")
        print(f"'{raw_num}' {n}_{ni}({mag_num}) '{raw_err}' {m}_{ei}({mag_err})")
        print("post metric:")
    if metric:
        b = int(np.floor(mag_num / 3))
        # equivalent to c = mag_num % 3
        c = mag_num - b * 3  # either of 0,1,2
        prefix = METRIC_PREFIXES[b * 3]
        # 0.0003 -> 0.000300, so real ni is now c - mag_num = 2 - (-4) = 6 instead of 4
        # c - mag_num is the digit to the left of the position of the metric decimal
        real_ni = ni
        ni = max(ni, c - mag_num)
        added_zeros = ni - real_ni
        raw_num = raw_num + "0" * added_zeros
        final_ni = (c - mag_num) - ni
        # add "ghost zeros" to error if necessary
        if c - mag_num > ei:
            if not ignore_uncertainty:
                raw_err = raw_err + "~" * (c - mag_num - ei)
            ei += c - mag_num - ei
        # 0.003 -> 3 m, b = -1, c = 0
        # 0.0003333 -> 333.3 µ, b = -2, c = 2
        # 0.00003 -> 30 µ, b = -2, c = 1
        # change formatting to metric formatting
        end = prefix
        if metric_space:
            end = " " + end
        if debug:
            print(
                f"c={c}, b={b}, prefix={prefix}, real_ni={real_ni}, final_ni={final_ni}"
            )
    if debug:
        print(f"'{raw_num}' {n}_{ni}({mag_num}) '{raw_err}' {m}_{ei}({mag_err})")
    # this is position of LEFTmost digit
    # ni, ei are positions of RIGHTmost digit
    # now round NUMBER to ERROR
    if mag_err > mag_num:
        # num = 0.0012345 -- n=5, ni = 3, mag = -3 = n - ni - 1
        # err = 0.019
        # uncertainty is a magnitude larger than number, still format number
        if not ignore_uncertainty:
            print(f"Uncrtnty: {uncertainty} IS MAGNITUDE(S) > THAN Numba: {number}")
        raw_err = "?"
        m = len(raw_err)
    elif ni > ei:
        # num = 0.00012345 --> 1235(23)  (note the rounding of 12345->12350)
        # err = 0.00023
        raw_num = round_to_idx(raw_num, n - (ni - ei))
        n = len(raw_num)
        ni = ei
    elif ni < ei:
        if ni > ei - m:
            # there is some overlap...
            # num = 0.000300  --> 1.2345(2)e-3
            # err = 0.000238
            raw_err = round_to_idx(raw_err, m + d)
            m = len(raw_err)
            ei = ni
        else:
            # num = 0.000100  --> 1.2345e-3
            # err = 0.000000234
            raw_err = ""
    elif ni == ei and not metric:
        # raw_err = ""
        pass
    if metric:
        ni = ni - (c - mag_num)
    if debug:
        print("post rounding:")
        print(f"'{raw_num}' {n}_{ni}({mag_num}) '{raw_err}' {m}_{ei}({mag_err})")
    extra_ni = 0
    # final form saves space by converting to scientific notation 0.0023 -> 2.3e-3
    if not shift and not percent and (ni - n) >= 2:
        extra_ni = ni - n + 1
        ni = n - 1
    if debug:
        print("final conversion:")
        print(f"'{raw_num}' {n}_{ni}({mag_num}) '{raw_err}' {m}_{ei}({mag_err})")
    # FINAL number formatting according to n and ni
    if ni >= n:  # place decimal before any digits
        raw_num = "0." + "0" * (ni - n) + raw_num
    elif ni > 0:  # place decimal in-between digits
        raw_num = raw_num[: n - ni] + "." + raw_num[n - ni :]
    elif ni < 0 and not metric:  # add non-significant zeroes after number (POSITIVE e)
        # if e1, want to just add 2 zeros
        if ni > -2:
            raw_num += "0" * (-ni)
            if ei > -2 and raw_err:
                raw_err += "0" * (-ei)
        else:
            end = "e" + str(-ni)
    if extra_ni and not metric:  # format removed decimal zeroes  (NEGATIVE e)
        end = "e" + str(-extra_ni)
    if end and math:  # format for LaTeX
        end = r"\text{" + end + "}"
    if percent:
        if math:
            end += "\\"
        end += "%"
    if is_negative:  # add back negative
        raw_num = "-" + raw_num
    if raw_err and not ignore_uncertainty:
        end = "(" + raw_err + ")" + end
    return raw_num + end


def format_dict_table(
    d: dict | list,
    maxperline=3,
    rowsep="\n\t",
    colsep=" | ",
    keyvalsep=" : ",
    val_from_key=None,
):
    """
    Formats a dictionary into a table with aligned columns for keys and values.

    :param d: Dictionary to format.
    :param maxperline: Number of dictionary items per line.
    :return: A formatted string representing the table.
    """
    if isinstance(d, dict):
        keys = list(d.keys())
    else:
        keys = d
    if val_from_key is None:
        val_from_key = lambda x: d[x]
    values = [val_from_key(key) for key in keys]

    # Determine max width per column before formatting
    key_widths = [
        max(len(str(keys[i])) for i in range(j, len(keys), maxperline))
        for j in range(min(maxperline, len(keys)))
    ]
    val_widths = [
        max(len(str(values[i])) for i in range(j, len(values), maxperline))
        for j in range(min(maxperline, len(values)))
    ]

    lines = []
    for i in range(0, len(keys), maxperline):
        chunk_keys = keys[i : i + maxperline]
        chunk_values = values[i : i + maxperline]

        line = colsep.join(
            f"{k:<{key_widths[j]}}{keyvalsep}{v:<{val_widths[j]}}"
            for j, (k, v) in enumerate(
                zip_longest(chunk_keys, chunk_values, fillvalue="")
            )
        )
        lines.append(line)

    return rowsep.join(lines)


# ----- SIGNAL MODELS ------ #
def ScaledGaussian(X, m, s, a):  # statistical/thermodynamic normal distribution
    return abs(a) * np.exp(-((X - m) ** 2) / (2 * s * s))


def ScaledLorentzian(X, m, s, a):  # physics signal function
    return abs(a) / ((2 * (X - m) / s) ** 2 + 1)


def ScaledVoigt(X, m, sg, sl, a):  # most robust signal modeler
    sg = abs(sg)
    sl = abs(sl)
    z = (X - m + sl * 1j) / (sg * np.sqrt(2))
    peak = special.wofz(sl * 1j / (sg * np.sqrt(2))).real
    if peak == 0 or peak == np.NaN:
        return np.zeros(len(X))
    if peak == np.inf:
        return np.ones(len(X)) * np.inf
    return a * special.wofz(z).real / peak


def planckFreqSpectrum(X, T, A):  # in frequency
    return (
        A
        * (X**3)
        * CVALS.h
        * 2e-32
        / (CVALS.c * CVALS.c * (np.exp(X * CVALS.h * 1e-5 / (CVALS.kB * T)) - 1))
    )


def planckWavelengthSpectrum(X, T, A):
    A0 = 1e6 / (4.09567405227 * T**5)  # normalize spectrum, set peak to 1
    return (
        A
        * A0
        * CVALS.h
        * CVALS.c
        * CVALS.c
        * 2e27
        / (X**5 * (np.exp(CVALS.h * CVALS.c * 1e6 / (X * CVALS.kB * T)) - 1))
    )


def Polynomial(X, *coeffs):  # general corrective polynomial? usually just constant
    Y = np.ones(len(X)) * coeffs[0]
    for i in range(1, len(coeffs)):
        Y += coeffs[i] * X**i
    return Y


def scaledPowerFunc(X, p, w):
    return w * (X**p)


def scaledExponentialFunc(X, exp, w):
    return w * (exp**X)


def linearStep(X, m1, b1, m2, b2):
    cutoff = (b2 - b1) / (m1 - m2)
    cut_idx = binarySearch(X, cutoff)
    return np.concatenate((X[:cut_idx] * m1 + b1, X[cut_idx:] * m2 + b2))


# ---- WRAP MODELS W/ PARAM NAMES/LABELS -----
class FuncWLabels:
    """Wrap a function with coefficient labels, excluding the first arg"""

    def __init__(self, func, lbls):
        if func.__code__.co_argcount - 1 != len(lbls):
            print(
                f"function {func.__name__} has {func.__code__.co_argcount - 1} coeffs \
                  and you provided {len(lbls)} labels."
            )
        self.func = func
        self.lbls = lbls

    def __str__(self):
        return self.func.__name__

    def __len__(self):
        return len(self.lbls)


Gaussian = FuncWLabels(ScaledGaussian, [r"\mu_{", r"\sigma_{", "a_{"])
Lorentzian = FuncWLabels(ScaledLorentzian, [r"\mu_{", r"\sigma_{", "a_{"])
Voigt = FuncWLabels(ScaledVoigt, [r"\mu_{", r"\sigma_{g", r"\sigma_{\ell", "a_{"])
WavelengthBBR = FuncWLabels(planckWavelengthSpectrum, [r"T_{", r"A_{"])
FrequencyBBR = FuncWLabels(planckFreqSpectrum, [r"T_{", r"A_{"])
ScaledPower = FuncWLabels(scaledPowerFunc, ["p_{", "w_{"])
ScaledExp = FuncWLabels(scaledExponentialFunc, ["exp_{", "w_{"])
LinearStep = FuncWLabels(linearStep, ["m1_{", "b1_{", "m2_{", "b2_{"])


# func can be single component funcs or a list of component funcs
# will make ncopies of each func or list of component funcs
# FUNC = sum of funcs and their components
class FuncAdder:
    """
    Makes a composite function from a list of FuncWLabels
     - FuncWLabels or list of FuncWLabels, functions stored w their coeff labels
       - self.funcs: list of functions
       - self.lbls:  list of all labels, each ending with an open "{" for labeling
       - self.nargs: number of arguments for each function (not including X)
     - ncopies of all functions to add together, or list of ncopies per function
       - self.ncopies:  int OR list of ints
       - if 0, function is "frozen" and simply added without being fit
       - only works if "frozen" functions are all placed before "unfrozen" ones
     - initialize coefficients - FULL coeff list given to composite function
       - self.coeffs: list of all coeffs
     - name given to the funcAdder (composite function name)
       - self.name
     - addPoly to add a polynomial function up to the nth polynomial offset
       - addPoly = 1 will add y-intercept, 2 will add parabolic func, etc.
    """

    def __init__(
        self, funcWLabels, ncopies=1, coeffs=[], name="funcAdder", addPoly=0
    ) -> None:
        self.name = self.__name__ = name
        if not isinstance(ncopies, list):
            ncopies = [ncopies]
        self.ncopies = ncopies
        self.coeffs = coeffs
        self.covar = []
        self.set_funcs(funcWLabels, self.ncopies, addPoly)

    def __str__(self) -> str:
        return self.printCoeffs()

    def set_funcs(
        self, funcWLabels: FuncWLabels | list[FuncWLabels], ncopies, addPoly=0
    ) -> None:
        """Compile self.funcs, self.nargs, and self.lbls from funcWLabels and ncopies
        - funcWLabels = list of funcWLabels
        - ncopies = list of ncopies to be made for each function, can be shorter than funcWLabels
        - addPoly = k: add polynomial of degree k
        """
        self.funcs = []
        self.nargs = []
        self.lbls = []
        self.frozen_idx = 0  # every coeff before this will not be fit
        if not isinstance(funcWLabels, list):  # list of functions
            if isinstance(funcWLabels, FuncWLabels):
                funcWLabels = [funcWLabels]
            elif isinstance(funcWLabels, str):  # u can name your function as a string?
                funcWLabels = [exec(funcWLabels)]
                if not isinstance(funcWLabels[0], FuncWLabels):
                    sys.exit(
                        "this should be a FuncWLabels type",
                        funcWLabels,
                        type(funcWLabels),
                    )
            elif addPoly:
                funcWLabels = []
                print("assuming just polynomial of degree", addPoly)
        elif not isinstance(funcWLabels[0], FuncWLabels):
            if isinstance(funcWLabels[0], str):  # of function strs?
                funcWLabels = [exec(func_str) for func_str in funcWLabels]
            else:
                sys.exit(
                    "what have u done... what is this--", funcWLabels, type(funcWLabels)
                )
        # iterate thru functions, ncopies, and populate relevant fields
        ncopy_idx = 0
        ncopy = ncopies[ncopy_idx]  # iterate thru ncopies
        frozen = ncopy == 0
        for funcWLbl in funcWLabels:  # for all functions
            f = funcWLbl.func
            lbls = funcWLbl.lbls
            if not frozen and ncopy == 0:
                print(
                    f"Please place frozen functions at the beginning. \
                        proceeding with ncopy=1 for {f.__name__}"
                )
            if frozen and ncopy:
                frozen = False
            if not ncopy:
                ncopy = 1
            self.funcs.extend([f] * ncopy)  # duplicate ncopies times
            nargs = f.__code__.co_argcount - 1
            if nargs != len(lbls):
                print("NOT OK", self.name, f, lbls)
            self.nargs.extend([nargs] * ncopy)
            if frozen:
                self.frozen_idx += nargs
            if ncopy == 1:
                self.lbls.extend([lbl + "}" for lbl in lbls])
            else:
                for i in range(1, ncopy + 1):
                    self.lbls.extend([lbl + str(i) + "}" for lbl in lbls])
            if ncopy_idx + 1 < len(ncopies):  # iterate ncopy
                ncopy_idx += 1
                ncopy = ncopies[ncopy_idx]
        # add an intercept for addPoly = 1, up to any arbitrary polynomial
        if addPoly > 0:
            self.funcs.append(Polynomial)
            self.nargs.append(addPoly)
            if addPoly == 1:
                self.lbls.append("C")
            elif addPoly == 2:
                self.lbls.extend(["b", "m"])
            else:
                self.lbls.extend(ALPHABET[:addPoly][::-1])

    # return the individual Ys of all funcs
    def indiv_Ys(self, X, coeffs=None, alsoFuncName=False):
        n = len(self.funcs)
        outputs = []
        coeff_index = 0
        if not coeffs:  # no coeffs specified, go with the saved ones
            coeffs = self.coeffs
        if len(coeffs) < len(self.coeffs):  # self.fit w/ frozen coeffs
            coeffs = self.coeffs[: self.frozen_idx] + list(coeffs)
        for i in range(n):
            f = self.funcs[i]
            nargs = self.nargs[i]
            cs = coeffs[coeff_index : coeff_index + nargs]
            coeff_index += nargs
            if alsoFuncName:
                outputs.append((f(X, *cs), f.__name__))
            else:
                outputs.append(f(X, *cs))
        return outputs

    def predict(self, X, *coeffs):
        """Use model to return Y_predict from X"""
        return np.sum(self.indiv_Ys(X, coeffs), axis=0)

    def fit(self, X, Y, initial_coeffs=None):
        """fits model over X, Y, returning Y_predict"""
        if not initial_coeffs:
            initial_coeffs = self.coeffs
        if len(initial_coeffs) + self.frozen_idx > len(self.coeffs):
            initial_coeffs = initial_coeffs[self.frozen_idx :]
        try:
            self.coeffs[self.frozen_idx :], self.covar = curve_fit(
                self.predict, X, Y, p0=initial_coeffs
            )
        except RuntimeError:
            print("ERROR: curve_fit could not predict given the model")
            print(
                "\t",
                ", ".join(
                    [
                        "%d:%s" % (self.nargs[i], self.funcs[i].__name__)
                        for i in range(len(self.funcs))
                    ]
                ),
            )
            print("\tcoeffs:", self.formatCoeffs(range(0, len(self.coeffs)), 1))
        return self.predict(X, *self.coeffs)

    def formatCoeffs(self, indices, printType=2):
        """uses .lbls and uFormat to return list of correctly formated coefficients for single or list of indices
        - printType =
          - 0: return copy-able list
          - 1: format coeffs to sig figs w/ labels
          - 2: format coeffs to uncertainty w/ labels"""
        if type(indices) == int:  # single coefficient index given
            indices = [indices]
        if printType == 2:
            lst = []
            for j in indices:
                if (
                    j < self.frozen_idx
                ):  # frozen idxs aren't fit, and thus have no covariance
                    cvar = 0
                else:
                    cvar = np.sqrt(self.covar[j - self.frozen_idx, j - self.frozen_idx])
                lst.append(
                    "$" + self.lbls[j] + "=" + uFormat(self.coeffs[j], cvar) + "$"
                )
            return lst
        elif printType == 1:
            return [
                "$" + self.lbls[j] + "=" + uFormat(self.coeffs[j], 0, figs=4) + "$"
                for j in indices
            ]
        else:
            return [str(self.coeffs[j]) for j in indices]

    def printCoeffs(self, printType=2, nCoeffsPerLine=2, onlyMus=False) -> str:
        """Returns string of model coefficients formatted for printing or plotting
        - printType:
          - 0: print coeffs in copy-able format as list of coeffs, w/ lines separated by function
          - 1: format coeffs for plotting with their labels
          - 2: format coeffs with labels AND uncertainties for plotting
        - nCoeffsPerLine = max num coefficients for plotting formatting
        - onlyMus: only return 'mean' coefficients (for gaussian, voigt, lorentzian)
        """
        if not len(self.coeffs):
            return self.name + " has no coeffs to print"
        if not len(self.covar) and printType == 2:  # no covars to format
            printType = 1
        n = len(self.coeffs)
        i = 0
        allCoeffs = []
        func = (
            allCoeffs.extend if printType else lambda x: allCoeffs.append(", ".join(x))
        )
        for fi in range(len(self.funcs)):
            nargs = self.nargs[fi]
            fname = self.funcs[fi].__name__
            thing = ""
            if onlyMus:
                if fname == "Gaussian" or fname == "Lorentzian" or fname == "Voigt":
                    thing = self.formatCoeffs(i, printType)
            else:
                thing = self.formatCoeffs(range(i, i + nargs), printType)
            func(thing)
            i += nargs
        # join extra non-function labels, if they exist
        if i + 1 < n:
            func(self.formatCoeffs(range(i, n), printType))
        if printType:
            i = 0
            n = len(allCoeffs)
            lines = []
            while i < n:
                lines.append(", ".join(allCoeffs[i : i + nCoeffsPerLine]))
                i += nCoeffsPerLine
            return "\n".join(lines)
        return "[" + "\n".join(allCoeffs) + "]"

    def plotComposite(
        self,
        X,
        plot_name="",
        coeffs=[],
        opacity=1.0,
        labelPeaks=False,
        Axis=None,
        plot_size=FIGSIZE,
        showFig=False,
    ):
        """
        Plot all the individual component functions that add up to composite model
        - plot_name is self-explanatory
        - coeffs: optionally specify coeffs for composite func, else will use self.coeffs
        - opacity: of the component lines
        - labelPeaks: will add peaks of any signal functions to the legend
        - Axis: specify existing pyplot axis to plot onto. if None, will make a new plot
        - plot_size: size of plot for making a new plot
        - showFig: plt.show() to show figure after making new plot
        """
        if not len(coeffs):
            coeffs = self.coeffs
        Y_add = self.predict(X, *coeffs)

        if not Axis:
            plt.figure("COMPOSITION_OF_FUNCADD", figsize=plot_size)
            ax = plt.gca()
        else:
            ax = Axis
        ax.plot(X, Y_add, label="Composite Fit", alpha=1)
        coeff_index = 0
        for i in range(len(self.funcs)):  # plot components individually
            f = self.funcs[i]
            nargs = self.nargs[i]
            cs = coeffs[coeff_index : coeff_index + nargs]
            if labelPeaks and f.__name__ != "Polynomial":  # add peaks into plot
                # print(f.__name__)
                if "planck" == f.__name__[:6]:  # black body spectra
                    mu = 1e6 * CVALS.wien / cs[0]  # wien displacement law for peak
                else:
                    mu = cs[0]
                y_mu = f(mu, *cs)
                ax.plot(
                    X,
                    f(X, *cs),
                    label="(%.1f,%.4f" % (mu, y_mu) + ")",
                    linestyle="dashed",
                    color=f"C{i}",
                    alpha=opacity,
                )
                ax.scatter(mu, y_mu, marker="o", color=f"C{i}", alpha=opacity, s=5)
            else:
                ax.plot(X, f(X, *cs), linestyle="dashed", color=f"C{i}", alpha=opacity)
            coeff_index += nargs

        if not Axis:
            ticksPerTick = 5
            ax.xaxis.set_minor_locator(mtick.AutoMinorLocator(ticksPerTick))
            ax.yaxis.set_minor_locator(mtick.AutoMinorLocator(ticksPerTick))
            plt.grid(which="both", color="#E5E5E5")
            plt.title("Composite Plot of " + plot_name + ", " + self.name)
            plt.legend(loc="best")
            print(
                "made",
                plot_name.replace(" ", "_")
                + "_"
                + self.name.replace(" ", "_")
                + "_composite_plot.png",
            )
            plt.savefig(
                plot_name.replace(" ", "_")
                + "_"
                + self.name.replace(" ", "_")
                + "_composite_plot.png",
                bbox_inches="tight",
            )
            if showFig:
                plt.show()
            plt.clf()


# EXAMPLE data array generation
make_example_data = False
if make_example_data:
    maxchars = max([len(fname) for fname in glob(dir + "/*")])
    data_entry = np.dtype([("name", f"U{maxchars}"), ("data", "O"), ("numpts", "i4")])
    data = np.array([], dtype=data_entry)
    # get data from file
    for fname in glob(dir + "/*"):
        arr = np.loadtxt(fname)
        entry = np.array([(fname, arr, len(arr))], dtype=data_entry)
        data = np.append(data, entry)


# ---- PLOTTING SCRIPTS ----- #
def plotRaw(
    arr: np.ndarray | tuple[np.ndarray],
    title: str,
    axes_titles: str | tuple[str],
    saveName=None,
    Axis=None,
    lines=[],
    axes_ranges=[],
):
    """Plots 2D array or tuple of arrays. Specify plt.ax with Axis.
    - axes_ranges: set (xlim, ylim) for tuple axis limitations
    - lines: list of X-positions for drawing vertical lines"""
    X = arr[0] if isinstance(arr, tuple) else arr[:, 0]
    Y = arr[1] if isinstance(arr, tuple) else arr[:, 1]
    if not isinstance(axes_titles, tuple):  # default to x-label
        axes_titles = (axes_titles, r"Wavelength $\lambda$, nm")
    fig, ax = (Axis, plt.gcf()) if Axis else plt.subplots(figsize=FIGSIZE)
    if len(lines):
        for line_x in lines:
            ax.plot(
                [line_x, line_x], [min(Y), max(Y) + 0.002], linewidth=1, color="blue"
            )
    ax.plot(X, Y, label="data", color="C0", zorder=2)
    ax.xaxis.set_minor_locator(mtick.AutoMinorLocator(TICKSPERTICK))
    ax.yaxis.set_minor_locator(mtick.AutoMinorLocator(TICKSPERTICK))
    ax.set(xlabel=axes_titles[0])
    if axes_titles[0][0] == "C":
        ax.set(ylabel=axes_titles[1])
    if axes_ranges:  # set ranges on axes from list of two tuples
        if not axes_ranges[0]:
            axes_ranges[0] = ax.get_xlim()
        if not axes_ranges[1]:
            axes_ranges[1] = ax.get_ylim()
        ax.set(xlim=axes_ranges[0], ylim=axes_ranges[1])
    if Axis:
        ax.set(title=title)
        return
    if saveName:
        fig.savefig(SAVEDIR + saveName + ".pdf", bbox_inches="tight")
        print("Saved figure to " + SAVEDIR + saveName + ".pdf")
    plt.legend()
    plt.show()
