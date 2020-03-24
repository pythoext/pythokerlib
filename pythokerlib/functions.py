# -*- coding: UTF-8 -*-

import sys
import decimal
import logging
import math
import string
import six
import unicodedata
import numpy as np
import pandas as P
import scipy
import scipy.stats as sps
import statsmodels

from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd
from unidecode import unidecode

from .gsf import to_date as _to_date, is_named_index_df, ExitNow
from .ufsa import ufsa_opener as _get_ufsa_opener, ufsa_listdir as _get_ufsa_listdir

def logger_to_stderr(level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stderr)
    # Create formatter and add it to the handler
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.handlers = [handler]


df = None
working_df = None
resource = None
global_safe_env = dict()


def pad(var, width, pos='left', filler='0'):
    """
    Pad a string to a new length.

    Parameters
    ----------
    var : specifies the variable to pad (variable as char, string or number
          type).

    width : specifies the final string length.

    pos : specifies the position of the variable var as left, center or
          right.

    filler : specifies the single char used for padding. Default is '0'.

    Returns
    -------
    padded : new formatted string

    Examples
    --------

    Pad to the right side of the string, to a new length of 3 characters:

    >>> pad(5,3,'left','0')
    '500'

    Pad to the right and left side of the string, to a new length of 5
    characters:

    >>> pad('E',5,'center','0')
    '00E00'

    Pad to the right side of the string, to a new length of 8 characters:

    >>> pad('client',8,'left','_')
    'client__'


    """
    def padder(var, width=width, pos=pos, filler=filler):
        if pos == 'left':
            f = 'ljust'
        elif pos == 'right':
            f = 'rjust'
        else:
            f = 'center'
        var = '%s' % var
        return getattr(var, f)(width, filler)

    if isinstance(var, P.Series):
        return var.map(padder)
    return padder(var)



def lpad(var, width, filler='0'):
    """
    Pad a string to a new length, leaving the original variable to the left
    side of the final string.

    Parameters
    ----------
    var : specifies the variable to pad (variable as char, string or number
          type)

    width : specifies the final string length.

    filler : specifies the single char used for padding. Default is '0'.

    Returns
    -------
    padded : new formatted string

    Examples
    --------

    Pad to the right side of the string with '0' character, to a new length
    of 3 characters:

    >>> lpad(5,3,'0')
    '500'

    """
    return pad(var, width, 'left', filler)


def rpad(var, width, filler='0'):
    """
    Pad a string to a new length, leaving the original variable to the
    right side of the final string.

    Parameters
    ----------
    var : specifies the variable to pad (variable as char, string or number
          type)

    width : specifies the final string length.

    filler : specifies the single char used for padding. Default is '0'.

    Returns
    -------
    padded : new formatted string

    Examples
    --------

    Pad to the left side of the string with '0' character, to a new length
    of 3 characters:

    >>> rpad(5,3,'0')
    '005'
    """
    return pad(var, width, 'right', filler)


def cpad(var, width, filler='0'):
    """
    Pad a string to a new length, leaving the original variable in the
    middle of the final string.

    Parameters
    ----------
    var : specifies the variable to pad (variable as char, string or number
          type)

    width : specifies the final string length.

    filler : specifies the single char used for padding. Default is '0'.

    Returns
    -------
    padded : new formatted string

    Examples
    --------

    Pad to the left and right side of the string with '_' character, to a
    new length of 3 characters:

    >>> cpad(5,3,'_')
    '_5_'
    """
    return pad(var, width, 'center', filler)


def reldelta(dt1, dt2, sorted=False):
    """
    Compute the relative delta time between two Series of the same length.
    At the moment the function can handle only two parameters of the same
    type.

    Parameters
    ----------
    dt1 : Series

    dt2 : Series

    sorted: True | False, if True reorder values to have always positive
            difference.

    Returns
    -------
    ser : Series with relative deltas between two Series of the same length


    Examples
    --------
    Compute the relative delta between 01/05/2012 and 14/05/2012

    >>> from pandas import datetime, Series

    >>> dt1 = Series(datetime(2012, 5, 1))
    >>> dt2 = Series(datetime(2012, 5, 14))
    >>> reldelta(dt1, dt2)

    [relativedelta(days=-13)]

    """
    if isinstance(dt1, P.Series) and isinstance(dt2, P.Series):
        return P.Series(
            [relativedelta(a, b)
             if not sorted else relativedelta(a, b)
             if a > b else relativedelta(b, a)
             if P.notnull([a, b]).all()
             else None for a, b in zip(dt1, dt2)])


def months_diff(dt1, dt2):
    """
    Compute the relative difference in months between two Series of the
    same length.
    If the Series is od datatime type it apply the algorithm for each row:
         diff = (dt1.years-dt2.years)*12 + dt1.months - dt2.months

    This is identical to SAS and excel implementation:
                 =(YEAR(A2)-YEAR(B2))*12+MONTH(A2)-MONTH(B2)


    Parameters
    ----------
    dt1 : Series

    dt2 : Series

    Returns
    -------
    ser : Series of integer with the differences in months.

    """
    if isinstance(dt1, P.Series) and isinstance(dt2, P.Series):
        return P.Series(
            [(a.year * 12 + a.month) - (b.year * 12 + b.month)
             if P.notnull([a, b]).all()
             else None for a, b in zip(dt1, dt2)])


def months_between(dt1, dt2):
    """
    Compute the relative difference in months between two Series like
    Oracle's MONTHS_BETWEEN.
    cfr: https://docs.oracle.com/cd/B19306_01/server.102/b14200/functions089.htm

    Parameters
    ----------
    dt1 : Series

    dt2 : Series

    Returns
    -------
    ser : Series of integer with the differences in months.

    """
    if (not isinstance(dt1, P.Series) or not isinstance(dt2, P.Series)):
        return
    # differenza in mesi
    dt3 = []
    for a, b in zip(dt1, dt2):
        if P.isnull(a) or P.isnull(b):
            dt3.append(P.np.nan)
            continue
        sign = 1
        if a < b:
            a, b = b, a
            sign = -1
        dm = (a.year - b.year) * 12 + (a.month - b.month)
        a_end = a + MonthEnd(0)
        b_end = b + MonthEnd(0)
        if (a == a_end and b == b_end) or (a.day == b.day):
            diff = dm
        else:
            delta = (a.day - b.day) / 31.0
            diff = dm + delta
        dt3.append(sign * diff)
    return P.Series(dt3)


def ser_to_dt(x):
    """
    Return a Series of date objects

    Parameters
    ----------
    x : Series of string or int64

    Returns
    -------
    y : Series of date objects

    Examples
    --------

    Convert the Q1 of 2005 to a datetime object:

    >>> dt1 = '2005q1'
    >>> to_dt(dt1)
    datetime.date(2005, 3, 31)

    Convert the date 15/01/1980 to a datetime object:

    >>> dt1 = (1980, 1, 15)
    >>> to_dt(dt1)
    datetime.date(1980, 1, 15)

    """
    # nan verrà convertito in NaT
    return x.map(lambda x: P.np.nan if P.isnull(x) else to_date(x))


def ser_to_timestamp(x):
    """
    Return a pandas Timestamp Series

    Parameters
    ----------
    x : Series of string or int64

    Returns
    -------
    y : Series of date objects

    Examples
    --------
    see to_timestamp for help on handled formats.
    """
    return x.apply(P.Timestamp)


def ser_to_istr(ser, missing=''):
    """
    Return a pandas series (dtype=object)
    trying to cast to string removing decimals.

    :param ser: a numeric Series.
    :return:


    'A' -> 'A'
    '3' -> '3'
    3   -> '3'
    3.0 -> '3'   *** Integer casting special case ***
    2.5 -> '2.5'
    2000-05-05 -> 2000-05-05
    NaN -> '' [by input parameter]

    corner case:
    9.9999999999999999999999999999999999999 --> 10

    """
    if ser.dtype.kind == 'O':
        return ser

    if ser.dtype.kind == 'f':
        return ser.apply(
            lambda x, missing=missing: missing if P.isnull(x)
            else str(int(x)) if P.np.mod(x, 1) == 0 else str(x))

    return ser.astype(str)


def to_dt(x):
    """
    Return a Series of date objects

    Parameters
    ----------
    x : Series of string or int64

    Returns
    -------
    y : Series of date objects

    Examples
    --------

    Convert the Q1 of 2005 to a datetime object:

    >>> dt1 = '2005q1'
    >>> to_dt(dt1)
    datetime.date(2005, 3, 31)

    Convert the date 15/01/1980 to a datetime object:

    >>> dt1 = (1980, 1, 15)
    >>> to_dt(dt1)
    datetime.date(1980, 1, 15)

    """
    # nan verrà convertito in NaT
    return x.map(lambda x: P.np.nan if P.isnull(x) else to_date(x))


def drop(*series):
    """
    Remove one or more series from a DataSource.

    Parameters
    ----------
    x :  Series of datatime as string or int64

    Examples
    --------

    Drop one Series X1`, `X2` and `X3` from current data-frame:

    >>> drop(X1, X2, X3)

    """

    # WORKAROUND: python attualmente non consente di passare più di 255
    # parametri ad una chiamata di funzione
    # (a meno che non vengano passati sotto forma di lista preceduta da *)
    # In questi casi si passa un solo parametro, che contiene la lista
    # completa.
    if len(series) == 1 and isinstance(series[0], (list, tuple)):
        series = series[0]

    todrop = []
    for x in series:
        if isinstance(x, P.Series):
            todrop.append(x.name)
        elif isinstance(x, six.string_types):
            todrop.append(x)

    logging.debug("_DROPing %s", todrop)
    global df
    df = df.drop(todrop, axis=1)


def keep_rows():
    """
    Remove rows in the data-frame subject to the condition specify in the
    filter box.
    Empty filter cause drop of all rows of DataSource

    Parameters
    ----------
    none

    Examples
    --------

    Keep the rows of a DataSource only if Series X has values.

    In the filter box

    >>> isnan(X)

    In the equation box

    >>> keep_rows()

    """
    global df
    df = working_df


def keep(*series):
    """
    Keep one or more list of Series from a DataSource. All other Series
    will be deleted. This function can be seen as the complementary
    function of drop() Besides the name we add a casting type as one of:
        *case insensitive*
        int, i,
        float, double, f, d
        object, string, str
        date, dt
        datetime, time, timestamp
    Parameters
    ----------
    x: series of datatime as string or int64
    Examples
    --------
    >>> keep([col_A])
    >>> keep(["col_A as float", col_B, "col_F as int", "col_G"])
    """

    # WORKAROUND: python attualmente non consente di passare più di 255
    # parametri ad una chiamata di funzione (a meno che non vengano passati
    # sotto forma di lista preceduta da *) In questi casi si passa un solo
    # parametro, che contiene la lista completa.
    if len(series) == 1 and isinstance(series[0], (list, tuple)):
        series = series[0]

    keepus = [y.name if isinstance(y, P.Series) else y for y in series]
    typed = {}

    def splitter(var_type):
        # split var / type
        x = var_type.split(' as ')
        if len(x) > 1:
            col = x[0].strip()
            cast = x[1].strip().lower()
            typed[col] = cast
        return x[0]

    keepus = map(splitter, keepus)
    logging.debug("keeping %s", keepus)
    global df
    df = df.drop(df.columns.difference(keepus), axis=1)

    logging.debug("Casting %s", typed.keys())
    for k, cast in six.iteritems(typed):
        try:
            if cast in ('dt', 'date'):
                df[k] = ser_to_dt(df[k])
            elif cast in ('datetime', 'time', 'timestamp'):
                df[k] = ser_to_timestamp(df[k])
            else:
                df[k] = df[k].astype(cast)
        except Exception as err:
            # Series name injection into error
            ety = type(err)
            raise ety("{} casting {} into {}".format(repr(err), k, cast))
    if keepus:
        df = df[keepus]


def dropnanrows():
    """
    Delete rows of missing values within a DataSource

    Parameters
    ----------
    x : Series as string or int64

    Examples
    --------

    Drop rows of missing values:

    >>> dropnanrows()

    """
    global df
    df = df.dropna(axis=0, how='all')


def dropnancols():
    """
    Delete columns of missing values.

    Parameters
    ----------
    x : Series

    Examples
    --------
    Drop columns of missing values:

    >>> dropnancols()
    """
    global df
    df = df.dropna(axis=1, how='all')


def roll_mean(ser, window):
    """
    Compute the mean over axis=1 (of the window size) and
    broadcast result over data-frame shape.

    Parameters
    ----------
    ser : Series
    window : size of the window to compute the mean

    Returns
    -------
    y : Series

    Examples
    --------

    Compute the mean of a rolling window of size 2 to the array x,
    along .........

    >>> x=np.arange(10).reshape((2,5))
    >>> x
    array([[0, 1, 2, 3, 4],
          [5, 6, 7, 8, 9]])
    >>> x.shape
    (2, 5)
    >>> roll_mean(x, 2)
    array([[ 1.5,  2.5],
           [ 6.5,  7.5]])

    """
    return P.rolling_mean(ser, window)


def roll_apply(window, func, stepper=1):
    global df
    ii = [int(x) for x in
          np.arange(0, df.shape[0] - window + 1, stepper)]
    out = P.Series([func(df.iloc[i:(i + window)]) for i in ii])
    out.index = df.index[window - 1::stepper]
    return out


def sanify_name(name):
    """
    Sanify the given name removing all invalid characters;
    if unicode is found, labels will be encoded in utf8.

    :param name: the string to sanify
    :return: the sanified
    """
    GOOD_CHOICE = set(string.ascii_letters + string.digits + '_')

    def remove_accents(input_str):
        try:
            input_str = six.text_type(input_str)
        except UnicodeDecodeError:
            input_str = six.text_type(input_str, encoding="utf8")
        nkfd_form = unicodedata.normalize('NFKD', input_str)
        return "".join([c for c in nkfd_form if not unicodedata.combining(c)])

    # transliterate
    # +
    # fixing unicode accent.
    new = unidecode(remove_accents(name))

    # removing bad characters
    new = ''.join([(c if c in GOOD_CHOICE else '_') for c in new])

    # check if starts with a digit
    if new[0] in string.digits:
        new = '_' + new

    # # nel caso in cui nel df ci sia una colonna chiamata "index" la
    # # traslittero
    # if new == "index":
    #     new = "index_{0}".format(idx)

    return new


def sanify_labels():
    """
    Remove all invalid characters from labels in a DataSource;
    if unicode is found, labels will be encoded in utf8.

    * convert all bad characters  into _ (underscore)
    * all labels starting with a digit will be prefixed with _ (underscore)
    * turn all accented unicode character into normalized form

    Parameters
    ----------
    none

    Examples
    --------
    Sanify lables of all Series in a data-frame:

    >>> sanify_labels()

    """
    mapping = {}
    index = []
    global df
    for x in df.columns:
        new = sanify_name(x)

        if x != new:
            mapping[x] = new

    if is_named_index_df(df):
        for x in df.index.names:
            new = sanify_name(x)
            index.append(new)
        df.index.names = index

    if mapping:
        df.rename(columns=mapping, inplace=True)
        logging.debug(
            "sanify_labels for %s labels", len(mapping) + len(index))
    else:
        logging.debug("sanify_labels: nothing to do")


def grp_shift(grp, trg, shift=1):
    """
    Apply a shift function to each group and
    broadcast the result to the original data frame.

    Parameters
    ----------
    grp : group key
       columns to use for grouping
    trg : target column
       target data
    shift : lag or lead, default value = 1

    Returns
    -------
    df : data-frame

    Examples
    --------

    Shift a time Series by n.2 observations:
    >>> grp_shift(ColumnID, ColumnIDtarget, -2)

    """
    _all = []
    for x in (grp, trg):
        if isinstance(x, six.string_types):
            _all.append(x)
        else:
            _all += x
    return df[_all].groupby(grp).apply(lambda x: x[trg].shift(shift))


def lookup(key):
    """
    Map values of a Series using input correspondence that must be
    described as parameters.

    Parameters
    ----------
    key : Series or scalar for the lookup

    Returns
    -------
    y : Series
        Same index as caller

    Examples
    --------
    >>> lookup(ColumnID)

    """
    global global_safe_env
    get = global_safe_env.get
    if isinstance(key, P.Series):
        return key.map(lambda x: get(x, np.nan))
    return get(key, np.nan)

def invalues(ser, choices):
    """
    Check if values of a Series are contained in a list of values.

    Parameters
    ----------
    ser : Series to be checked
    choices : List of values

    Returns
    -------
    ser : Boolean Series showing whether each element is contained in
          choices.

    Examples
    --------
    Find if the values 1, 2 or 3 are contained in Series X:

    >>> invalues(X, [1, 2, 3])

    """
    return ser.isin(choices)

def notinvalues(ser, choices):
    """
    Check if values of a Series are not in a list of values.

    Parameters
    ----------
    ser : Series to be checked
    choices : List of values

    Returns
    -------
    ser : Return boolean Series showing whether each element is not
          contained in choices.

    Examples
    --------
    Find if the values 1 and 2 are not contained in Series X:

    >>> notinvalues(X, [1, 2])

    """
    return ser.map(lambda x: x not in choices)

def isnotnan(x):
    """
    Check for non empty values (NaN) inside a Series.

    Parameters
    ----------
    x : Series to be checked

    Returns
    -------
    ser : Return boolean Series, values equal to True if not empty values

    Examples
    --------
    >>> isnotnan(X)

    """
    return np.invert(P.isnull(x))

def asstr(ser):
    """
    Convert a Series into a Series of string objects.

    Parameters
    ----------
    ser : Series

    Returns
    -------
    ser : Series of string objects

    Examples
    --------
    >>> asstr(ser)
    """
    return ser.map(str)


def set_pk(*keys):
    """
    Set the DataSource index (the row labels) to the values of one or
    more existing columns. The columns with the index value will be
    automatically erased from the DataSource.

    Parameters
    ----------
    keys : column label or list of column labels

    Examples
    --------

    Set the data-frame index equal to the values contained in Series X1
    and X2:

    >>> set_pk(X1, X2)

    """
    keys = [y.name if isinstance(y, P.Series) else y for y in keys]
    global df
    df.set_index(keys, inplace=True, drop=True)


def reset_pk():
    """
    XXX
    """
    global df
    df.reset_index(inplace=True)


def ascending(ser):
    """
    Sort a Series in ascending order.

    Parameters
    ----------
    ser : Series to be sorted.

    Returns
    -------
    ser_sorted : Series of the same type and shape as `ser`.

    Examples
    --------

    Sort a Series in ascending order:

    >>> ascending(ser)

    =====  =====
    Input  Output
    -----  -----
      ser  ser_sorted
    =====  =====
    3       1
    9       3
    4       4
    1       9
    =====  =====

    """
    return np.sort(ser)

def descending(ser):
    """
    Sort a Series in descending order.

    Parameters
    ----------
    ser : Series to be sorted.

     Returns
    -------
    ser_sorted : Series of the same type and shape as `ser`.

    Examples
    --------
    Sort a Series in descending order:

    >>> descending(ser)

    =====  =====
    Input  Output
    -----  -----
      ser  ser_sorted
    =====  =====
    3       9
    9       4
    4       3
    1       1
    =====  =====

    """
    return np.sort(ser)[::-1]

def lag(ser, lag=1):
    """
    Delay a Series by a specified time step (i.e. number of observations)
    [one can access the preceding row, ]
    and returns the result in a Series object. Default lag time is n.1
    observation.

    Parameters
    ----------
    ser : Series
    lag : Number of shifts (backward in time, default value is '1')

    Returns
    -------
    ser_shifted : same type as caller

    Examples
    --------
    Delay a Series by one observation:

    >>> lag(ser)

    ======  ========  ===========
    pk     Input       Output
    ------  --------  -----------
             ser      ser_shifted
    ======  ========  ===========
    2012Q1   1.5
    2012Q2   1.6       1.5
    2012Q3   1.5       1.6
    2012Q4   1.7       1.5
    ======  ========  ===========


    Delays a Series by two observations:

    >>> lag(ser,2)

    """
    # "lead(ser,2)" is the same as "ser.shift(-2)"
    return ser.shift(abs(lag))

def lead(ser, lead=1):
    """
    Advance a Series by a specified time step (i.e. number of observations)
    and returns the result in a Series object. Default lead time is n.1
    observation.

    Parameters
    ----------
    ser : Series
    lead : Number of shifts (forward in time, default value is '1')

    Returns
    -------
    ser_shifted : same type as caller

    Examples
    --------
    Advances a Series by one observation:

    >>> lead(ser)

    ======  ========  ===========
      pk     Input       Output
    ------  --------  -----------
             ser      ser_shifted
    ======  ========  ===========
    2012Q1   1.5       1.6
    2012Q2   1.6       1.5
    2012Q3   1.5       1.7
    2012Q4   1.7
    ======  ========  ===========


    Advances a Series by two observations:

    >>> lead(ser,2)
    """
    # "lead(ser,2)" is the same as "ser.shift(-2)"
    return ser.shift(-abs(lead))

def today():
    """
    Return today date as a datetime object.

    Parameters
    ----------
    none

    Returns
    -------
    date : datetime object

    Examples
    --------

    >>> today()
    2014-06-11
    """
    return P.datetime.today().date()


def last(n=1):
    """
    Return the last `n` items of a sequence.

    Parameters
    ----------
    n : Number of items (integer type) from the end of a sequence

    Returns
    -------
    out : same type as caller

    Examples
    --------
    Take the last 2 elements of series `A`:

    >>> A[last(2)]

    """
    return slice(-n, None)


def first(n=1):
    """
    Return the first `n` items of a sequence.

    Parameters
    ----------
    n : Number of items from the beginning of a sequence. N must be an int
        type.

    Returns
    -------
    out : same type as caller

    Examples
    --------
    Take the first 3 elements of an array `A`:

   >>> A[first(3)]


    """
    return slice(None, n)


def rename(src, dst):
    """
    Rename a Series within a DataSource.

    Parameters
    ----------
    src : Name of the Series to be renamed
    dst : New name of the Series

    Examples
    --------
    >>> rename("old_name", "new_name")

    """
    global df
    return df.rename(columns={src: dst}, inplace=True)


def pop(ser):
    """
    Return a new Series equal to Series in a DataSource, and drop that
    Series from the DataSource.

    Parameters
    ----------
    ser : Series to be copied and to be dropped

    Returns
    -------
    out : Series with the same values as ser

    Examples
    --------
    Assign Series X to a new Series Y and drop X from the DataSource:

    >>> pop(X)

    The previous operation can be obtained also in two steps with:

    >>> Y = X
    >>> drop(X)

    """
    global df
    if isinstance(ser, P.Series):
        df.pop(ser.name)
    else:
        df.pop(ser)


# R.M.: qui tutte le funzioni matematiche etc di numpy.
# utili per autogenerare la guida con le nostre descrizioni.
def nanmin(a, **kw):
    """
    Return minimum of a Series or array_like, ignoring any NaNs.
    When all-NaN slices are encountered a ``RuntimeWarning`` is raised and
    Nan is returned for that slice.

    Parameters
    ----------
    ser : Series or array containing numbers whose minimum is desired.

    Returns
    -------
    out : Minimum value

    Examples
    --------
    Return minimun of a Series X:

    >>> nanmin(X)

    """
    return np.nanmin(a, **kw)

def nanmax(ser, axis=None, out=None, keepdims=None):
    """
    Return the maximum of a Series or array_like, ignoring any
    NaNs.  When all-NaN slices are encountered a ``RuntimeWarning`` is
    raised and NaN is returned for that slice.

    Parameters
    ----------
    ser : Series or array containing numbers whose minimum is desired.
          If `a` is not an array, a conversion is attempted.

    Returns
    -------
    out : Maximum value

    Examples
    --------
    Return maximum of a Series X:

    >>> nanmax(X)

    """
    # il parametro keepdims viene passato al metodo mean della classe
    # dell'oggetto a se valorizzato, nel caso però di series pandas delle
    # versione 0.18 questo parametro non esiste; nel caso quindi che mean
    # venga usata come funzione di aggregazione in una groupby va in errore
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html
    keepdims = keepdims if keepdims is not None else np._NoValue
    return np.nanmax(ser, axis, out, keepdims)


def nansum(*a):
    """
    Return sum row wise of Series or array_likes, ignoring any NaNs.
    Nan is considered Zero.

    Parameters
    ----------
    ser : Series or array or scalars.

    Returns
    -------
    out : Series with sum row wise.

    Examples
    --------
    >>> nansum(X,Y)

    """
    return np.nansum(a, axis=0)


def mean(a, axis=None, dtype=None, out=None, keepdims=None):
    """
    Compute the arithmetic mean.

    Parameters
    ----------
    a : Series or array_like containing numbers whose mean is desired.
        If `a` is not an array, a conversion is attempted.

    Returns
    -------
    m : ndarray, see dtype parameter above
        If `out=None`, returns a new array containing the mean values,
        otherwise a reference to the output array is returned.

    """
    # il parametro keepdims viene passato al metodo mean della classe
    # dell'oggetto a se valorizzato, nel caso però di series pandas delle
    # versione 0.18 questo parametro non esiste; nel caso quindi che mean
    # venga usata come funzione di aggregazione in una groupby va in errore
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html
    keepdims = keepdims if keepdims is not None else np._NoValue
    return np.mean(a, axis, dtype, out, keepdims)

def exp(x):
    """
    Calculate the exponential of all elements in the input.

    Parameters
    ----------
    x : Series or input values.

    Returns
    -------
    out : Output array, element-wise exponential of `x`.

    Examples
    --------
    Compute the exponential of a series

    >>> exp(ser)

    """
    return np.exp(x)


def sqrt(x):
    """
    Calculate the positive square-root of all elements in the input.

    Parameters
    ----------
    x : Series or input values.

    Returns
    -------
    out : Output array, element-wise sqrt of `x`.

    Examples
    --------
    Compute the sqrt of a series

    >>> sqrt(ser)
    """
    return np.sqrt(x)


def round(a, decimals=0, out=None, rule=None):
    """
    Round to the given number of decimals.

    Parameters
    ----------
    a : Number or Series of numbers to be rounded
    decimals : Number of decimals (int type). Default is 0.
    rule : more sophisticated kind of rounding. Rounding rule can be:
      'ROUND_CEILING',
      'ROUND_DOWN',
      'ROUND_FLOOR',
      'ROUND_HALF_DOWN',
      'ROUND_HALF_EVEN',
      'ROUND_HALF_UP',
      'ROUND_UP',
      'ROUND_05UP'
      for more details visit: https://docs.python.org/2/library/decimal.html

    Returns
    -------
    out : same type as caller

    Examples
    --------
    Round to 2 decimals.

    >>> round([10.1342, 9.4519],2)
    [10.13, 9.45]


    """
    if not isinstance(a, P.Series):
        rule = None

    if rule is None:
        if isinstance(a, P.Series):
            return np.round(a.values, decimals, out)
        return np.round(a, decimals, out)
    else:
        if decimals > 1:
            quant_str = '.' + '0' * (decimals - 1) + '1'
        else:
            quant_str = '.1'

        return a.map(
            lambda x: decimal.Decimal(str(x)).quantize(
                decimal.Decimal(quant_str),
                rounding=getattr(decimal, rule.upper())))

def floor(x):
    """
    Return the floor of the input, element-wise.
    The floor of the scalar x is the largest integer i, such that i <= x.
    It is often denoted as lfloor x \rfloor.

    Parameters
    ----------
    a : Series or array_like.

    Returns
    -------
    out : same type as caller

    Notes
    -----
    Some spreadsheet programs calculate the “floor-towards-zero”, in other
    words floor(-2.5) == -2. NumPy instead uses the definition of floor
    where floor(-2.5) == -3.

    Examples
    --------

    >>> floor(X)

    """
    return np.floor(x)


def log(x):
    """
    Compute the natural logarithm, element-wise.

    The natural logarithm log is the inverse of the exponential function,
    so that log(exp(x)) = x. The natural logarithm is logarithm in base e.

    Parameters
    ----------
    x : Series or array_like

    Returns
    -------
    out : same type as caller

    Examples
    --------
    Compute the exponential of X:

    >>> exp(X)
    """
    return np.log(x)


def std(a):
    """
    Compute the standard deviation.

    Returns the standard deviation, a measure of the spread of a
    distribution, of the array elements.

    Parameters
    ----------
    a : Series or array_like
        Calculate the standard deviation of these values.

    Returns
    -------
    out : ndarray.

    Examples
    --------
    Compute the std of Series X:

    >>> std(X)

    """
    return np.std(a)


def sum(a):
    """
    Sum of elements of array or series.

    Parameters
    ----------
    a : Series or array_like
        Elements to sum.

    Returns
    -------
    sum_along_axis : ndarray
        An array with the same shape as `a`, with the specified
        axis removed.   If `a` is a 0-d array, or if `axis` is None, a
        scalar is returned.

    Examples
    --------
    Compute the sum of Series X:

    >>> sum(X)
    """
    return np.sum(a)


def to_date(d):
    """
    Convert a string or numbers into a date object
    d can be of the following type, where the notation used is
    D= Day, W=Week, M=Month, Q=Quarter, Y=Year:

    ================  =================================
     Date format       Example
    ================  =================================
    YYYY-MM-DD         2006-10-05  (recommended)
    YYYYMMDD           20061005
    DD-MM-Y(YYY)	   05-10-06 or 05-10-2006
    DD MM Y(YYY)       05 10 06 or 05 10 2006
    DD/MM/Y(YYY)       05/10/06 or 05/10/2006
    DD.MM.Y(YYY)	   05.10.06 or 05.10.2006
    D-M-Y(YYY)	       5-10-6 or 5-10-2006
    D mon Y(YYY)	   5 oct 06 o 5 october 2006
    NNNNNNNN	       732313
    YYYY-Qn	           2006-Q4
    YYYY:n             2001:3 (n number of trimester)
    YYYYQn	           2006Q4
    YYYY-Mn	           2006-M10
    YYYYMn	           2006M10
    ================  =================================

    Note: The format 2006.4 is not valid, whereas 20106-4 is valid, but
    incorrect since it is equivalent to 28/4/2011.

    Parameters
    ----------
    d : date written in different formats.

    Returns
    -------
    out : date converted to data object.

    Examples
    --------

    >>> to_date('set 1985')
    2014-09-27

    >>> to_date('1985,dec,1')
    1985-12-01

    >>> to_date('2 jan 1998')
    1998-01-02

    >>> to_date((2012, 1, 15))
    2012-01-15

    >>> to_date( 722820 )
    1980-01-06

    >>> to_date('2005q1')
    2005-03-31

    >>> to_date('2000 Q4')
    2000-12-31

    >>> to_date('2001:3')
    2001-09-30

    >>> to_date('2001M3')
    2001-03-31

    >>> to_date('2001 M12')
    2001-12-31

    """
    return _to_date(d)


def to_timestamp(d):
    """
    Convert a string or numbers into a pandas Timestamp object.

    Is unsafe to use not univoque formats as:
       P.Timestamp('03/04/2015')
       Timestamp('2015-03-04 00:00:00')  >> wrong value

    d can be of the following type, where the notation used is
    D= Day, W=Week, M=Month, Q=Quarter, Y=Year:

    ================  =================================
     Date format       Example
    ================  =================================
    YYYY-MM-DD         2006-10-05  (recommended)
    YYYYMMDD           20061005
    MM-DD-YYYY
    19-Apr-2016
    ================  =================================

    Parameters
    ----------
    d : date written in different formats (usually a string)

    Returns
    -------
    out : date converted to Timestamp object.

    Examples
    --------

    >>> to_timestamp('2014-09-27')
    Timestamp('2014-09-27 00:00:00')
    """
    return P.Timestamp(d)

# ho aggiunto questa esplicita, invece che avere solo pk nel dizionario
# sotto
def get_pk():
    """
    Get the current DataSource index.

    Parameters
    ----------
    none

    Returns
    -------
    out : index values

    Examples
    --------

    >>> get_pk()

    """
    global df
    return df.index


def isnan(x):
    """
    Check for empty values (NaN) inside a Series.

    Parameters
    ----------
    x : Series to be checked

    Returns
    -------
    ser : Return boolean Series, values equal to True if NaN values

    Examples
    --------

    >>> isnan(X)
    """
    return P.isnull(x)


def isinf(x):
    """
    Check for Inf inside a Series.

    Parameters
    ----------
    x : Series to be checked

    Returns
    -------
    ser : Return boolean Series, values equal to True if Inf values

    Examples
    --------

    >>> isinf(X)

    """
    return np.isinf(x)


# ho aggiunto queste due per ordinare un dataframe
def sort_asc(*keys):
    """
    Sort a DataSource by the values in one or more columns in ascending
    order. NaNs are placed at the end.

    Parameters
    ----------
    ser : Name of the Series. Accepts a column name or a list for a nested
          sort.

    Examples
    --------
    Sort by only the value of a Series

    >>> sort_asc(CustomerID)

    Sort by only the value of more Series

    >>> sort_asc(CustomerID, Name)

    """
    keys = list(keys)
    global df
    df.sort(columns=keys, ascending=True, inplace=True)


def sort_desc(*keys):
    """
    Sort a DataSource by the values in one or more columns in descending
    order. NaNs are placed at the end.

    Parameters
    ----------
    ser : Name of the Series. Accepts a column name or a list for a nested
          sort.

    Examples
    --------
    Sort by only the value of a Series

    >>> sort_desc(CustomerID)

    Sort by only the value of more Series

    >>> sort_desc(CustomerID, Name)

    """
    keys = list(keys)
    global df
    df.sort(columns=keys, ascending=False, inplace=True)


def index_values(level):
    """
    Return vector of label values for requested level, equal to the length
    of the index.

    Parameters
    ----------
    level : int or level name

    Returns
    -------
    values : ndarray
    """
    global df
    return df.index.get_level_values(level)


def random_db(cols=None, rows=100):
    """
    Return a random db.

    Parameters
    ----------
    cols: tuple of columns labels.
    rows: number of rows as scalar.

    Returns
    -------
    dataframe of shape (rows,cols) random.

    """
    if cols is None:
        cols = ['A', 'B', 'C']

    return P.DataFrame(
        data=np.random.rand(rows, len(cols)),
        index=np.arange(rows),
        columns=cols)


def head(n=1000):
    """
    Set the current db with the first rows.

    Parameters
    ----------
    n: Number of rows. default=1000
    """
    if not isinstance(n, int):
        raise ValueError("Number of row must be an integer value.")
    global df
    if n < len(df):
        df = df.head(n)


def sample(samplesize=1000):
    """
    Set the current db with a sample of data.

    Parameters
    ----------
    samplesize: Number of rows. Sample size.
                default=1000
    """
    if not isinstance(samplesize, int):
        raise ValueError("Sample size must be an integer value.")

    global df
    if samplesize < len(df):
        idxs = [True] * samplesize + [False] * (len(df) - samplesize)
        P.np.random.shuffle(idxs)
        df = df[idxs]

def distinct(*cols):
    """
    XXX
    """
    global df
    if cols:
        col_names = []
        for x in cols:
            if isinstance(x, P.Series):
                col_names.append(x.name)
            else:
                col_names.append(x)
        df.drop_duplicates(col_names, inplace=True)
    else:
        df.drop_duplicates(inplace=True)


def pivot(index=None, columns=None, values=None, **kw):
    """
    Reshape data (produce a pivot table) based on column values.
    Uses unique values from index / columns to form axes

    Parameters
    ----------
    index : string or object
            Column name to use to make new frame’s index
    columns : string or object
              Column name to use to make new frame’s columns
    values : string or object, optional
             Column name to use for populating new frame’s values

    """
    if isinstance(index, P.Series):
        index = index.name
    if isinstance(columns, P.Series):
        columns = columns.name
    if isinstance(values, P.Series):
        values = values.name

    global df
    df = P.pivot_table(
        df, index=index, columns=columns, values=values, **kw)


def unpivot(keys=None, columns=None, value_name='value', var_name='variable'):
    global df
    df = P.melt(
        df,
        id_vars=keys,
        value_vars=columns,
        var_name=var_name,
        value_name=value_name)


def full_transpose():
    """
    Transpose the current data-frame
    index and columns.
    """
    global df
    df = df.transpose()


def set_db(newdf, sanify_labels=False):
    """
    Substitute current data-frame with df.

    Parameters
    ----------
    df: new data-frame
    sanify_labels: (default: False) Remove all invalid characters from
        labels in a DataSource

    Examples
    --------
    set_db( db.transpose() )
    """
    if not isinstance(newdf, P.DataFrame):
        raise ValueError("set_db must be used with a valid data-frame")

    global df
    df = newdf
    if sanify_labels:
        sanify_labels()


def stack(level=-1):
    """
    Pivot a level of the column labels.
    new data-frame have a hierarchical index with a new
    inner-most level of row labels.

    Parameters
    ----------
    level: int or column name (or list of them)
    """
    global df
    ret = df.stack(level=level)

    if isinstance(ret, P.Series):
        ret = P.DataFrame(ret)

    df = ret


def unstack(level=-1):
    """
    Pivot a level of the index labels.
    new data-frame have a new level of column
    labels whose inner-most level consists
    of the pivoted index labels.

    Parameters
    ----------
    level: int or column name (or list of them)
    """
    # FIXME: verificare il funzionamento quando torna una serie
    global df
    df = df.unstack(level=level)


def log_growth_rate(start_value, ser):
    """
    calculate logaritmic growth rate of a serie
    by default it will skipp nan values.
    """
    return ser.cumsum() + start_value


def exit(msg=None):
    """
    exit macro's execution

    Parameters
    ----------
    msg: exit's message
    """
    if msg:
        logging.info("Exit function: %s", msg)
        raise ExitNow()


def lprint(msg):
    """
    log message in execution.log file; you have to set 'enabled' in
    'Log execution data' option of macro's metadata

    Parameters
    ----------
    msg: message to log
    """
    logging.info(msg)


def opener_ufsa(filename, mode='r'):
    """
    :param filename: logic name in the ufsa path
    :param mode: opener mode, all python modes are supported: r, rb, w, wb, r+, w+, a
    :return: a filehandler

    caveats.. user MUST close the file
    """
    return _get_ufsa_opener(filename, mode)


def lister_ufsa():
    """
    :param filename: logic name in the ufsa path
    :param mode: opener mode, all python modes are supported: r, rb, w, wb, r+, w+, a
    :return: a filehandler

    caveats.. user MUST close the file
    """
    return _get_ufsa_listdir(absname=False)


def get_attachment(attachment_name):
    """
    get an attachment of the running macro

    Parameters
    ----------
    attachment_name: name of the attachment
    """
    if resource is None:
        raise RuntimeError("Resource not defined!")
    if not hasattr(resource, "get_attachment"):
        raise RuntimeError("Invalid resource!")
    return getattr(resource, "get_attachment")(attachment_name)


def add_attachment(attachment_name, filestream, description="",
                   overwrite=True):
    """
    add an attachment to the running macro

    Parameters
    ----------
    attachment_name: name of the attachment
    filestream: file stream to write
    description: a description of the attachment [default: empty string]
    overwrite: if overwrite existing attachment with same name
        [default: True]
    """
    if resource is None:
        raise RuntimeError("Resource not defined!")
    if not hasattr(resource, "add_attachment"):
        raise RuntimeError("Invalid resource!")
    getattr(resource, "add_attachment")(attachment_name, filestream, description=description,
            overwrite=overwrite)
    getattr(resource, 'save')()


def functions_env():
    global df

    functions = {
        # moduli / obj
        "numpy": np,
        "pandas": P,
        "sps": sps,
        "scipy": scipy,
        "statsmodels": statsmodels,


        # facilities

        "linalg": np.linalg,
        "db": df,
        "dblen": len(df) if df is not None else None,
        "set_db": set_db,
        "math": math,
        "random_db": random_db,
        "exit": exit,
        "lprint": lprint,
        "open_file": opener_ufsa,

        # funzioni STR
        "pad": pad,
        "lpad": lpad,
        "rpad": rpad,
        "cpad": cpad,

        # funzioni statistiche
        "nanmin": nanmin,  # np.nanmin,
        "nanmax": nanmax,  # np.nanmax,
        "nansum": nansum,  # np.nansum,
        "mean": mean,  # np.mean,
        "std": std,  # np.std,

        # funzioni matematiche
        "sum": sum,  # np.sum,   #buildin override
        "log": log,  # np.log,   #buildin override
        "exp": exp,  # np.exp,
        "sqrt": sqrt,
        # diff" :diff,   #np.diff,
        "round": round,  # np.round,
        "floor": floor,  # np.floor,
        "rand": np.random,

        # tipi / tipizzatori
        "isnan": isnan,  # P.isnull,
        "isinf": isinf,  # np.isinf,
        "isnotnan": isnotnan,
        "asstr": asstr,
        "ser_to_istr": ser_to_istr,

        # operatori logici
        "land": np.logical_and,
        "lor": np.logical_or,
        "lxor": np.logical_xor,

        # datetime / Timestamp
        "ser_to_dt": ser_to_dt,
        "ser_to_timestamp": ser_to_timestamp,
        "to_date": to_date,
        "reldelta": reldelta,
        "today": today,
        "months_diff": months_diff,
        "months_between": months_between,

        # parametri costanti
        "delta30d": np.timedelta64(1, '30D'),
        "delta1d": np.timedelta64(1, 'D'),
        "nan": np.nan,

        # intra-groupby-functions
        "grp_shift": grp_shift,

        # index/join/append/reshaping/sorting
        "set_pk": set_pk,
        "get_pk": get_pk,
        "reset_pk": reset_pk,
        "drop": drop,
        "keep_rows": keep_rows,
        "keep": keep,
        "dropnanrows": dropnanrows,
        "dropnancols": dropnancols,
        "distinct": distinct,
        "sanify_labels": sanify_labels,
        "rename": rename,
        "pop": pop,
        "sort_asc": sort_asc,
        "sort_desc": sort_desc,
        "index_values": index_values,

        # ordering
        "ascending": ascending,
        "descending": descending,

        # rolling windows
        "roll_mean": roll_mean,
        "roll_apply": roll_apply,
        "log_growth_rate": log_growth_rate,

        # trasposizioni/pivot
        "invalues": invalues,
        "notinvalues": notinvalues,
        "lookup": lookup,
        "lag": lag,
        "lead": lead,
        "pivot": pivot,
        "unpivot": unpivot,
        "stack": stack,
        "unstack": unstack,
        "full_transpose": full_transpose,

        # slicing
        "last": last,
        "first": first,
        "head": head,
        "sample": sample,

        # macro specific
        "get_attachment": get_attachment,
        "add_attachment": add_attachment,
    }
    return functions
