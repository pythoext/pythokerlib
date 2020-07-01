import numbers
import re
import six
from calendar import monthrange
from datetime import date, datetime

import numpy as np
from dateutil.parser import parse
from pandas import Timestamp


class ExitNow(Exception):
    """
    This exception only break the macro execution
    Don't go further executing equations
    """
    pass


def is_named_index_df(df):
    try:
        return bool(
            set(df.index.names) - {None})
    except AttributeError:
        raise RuntimeError("Empty or corrupted dataset!")


def to_date(d):
    """ Convert d into datetime.date

        datetime.datetime.now() => datetime.date(2009, 4, 1)
        (1980, 1, 15) => datetime.date(1980, 1, 15)
        datetime.date.today() => datetime.date(2009, 4, 1)
        722820 => datetime.date(1980, 1, 6)
        '2005q1' => datetime.date(2005, 1, 1)
        '2000 Q4' => datetime.date(2000, 10, 1)
        '2001:3' => datetime.date(2001, 7, 1)
        '2001M3' => datetime.date(2001, 3, 1)
        '2001  M12' => datetime.date(2001, 1, 1)
        '2 jan 1998' => datetime.date(1998, 1, 2)
        'oct 1985' => datetime.date(1985, 10, 22)
        '1985,dec,1' => datetime.date(1985, 12, 1)
    """

    if not d:
        return

    if isinstance(d, Timestamp):
        return d.to_datetime().date()

    if isinstance(d, np.datetime64):
        d = d.astype(datetime)

    if isinstance(d, date):
        return d

    if isinstance(d, datetime):
        return d.date()

    if isinstance(d, (tuple, list)):
        return date(d[0], d[1], d[2])

    if isinstance(d, six.string_types):
        qre = r'(\d\d\d\d)[ ]*[qQ:]([1-4]$)'
        m = re.match(qre, d)
        if m:
            # Convenzione fine periodo
            _y, _m = int(m.groups()[0]), (int(m.groups()[1]) - 1) * 3 + 3
            return date(_y, _m, monthrange(_y, _m)[1])

        mre = r'(\d\d\d\d)[ ]*[mM-](0?[1-9]$|1[0-2]$)'
        m = re.match(mre, d)
        if m:
            # Convenzione fine periodo
            _y, _m = int(m.groups()[0]), int(m.groups()[1])
            return date(_y, _m, monthrange(_y, _m)[1])
        # la parse interpreta la stringa '' come la data odierna, in questa
        # maniera evitiamo lo faccia
        if not (len(d) == 2 and d.startswith("'") and d.endswith("'")):
            try:
                return parse(d, dayfirst=True).date()
            except OverflowError:
                pass

    if isinstance(d, numbers.Integral):
        if 1677 < d < 2262:
            # 1677 e 2262 sono la data min e max per l'oggetto timestamp di
            # pandas
            return date(d, 1, 1)
        elif 612148 < d < 825814:
            # come sopra ma per il formato ordinale
            return date.fromordinal(d)

    if hasattr(d, 'to_timestamp'):
        return d.to_timestamp().date()

    raise NotImplementedError("{} Not yet implemented".format(type(d)))