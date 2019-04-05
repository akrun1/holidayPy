from datetime import date
import holidays
import pandas as pd
import numpy as np
from fbprophet import Prophet
import inspect


def getHolidayListCountries(country_list, year_list, lower_window_list, upper_window_list):
    """
    Parameters
    ----------
    country_list: list of country name
    year_list: list of years
    lower_window_list: list of lower range of days around the date to be included as holidays
    upper_window_list: list of upper range of days around the date to be included as holidays

    Returns
    -------
    dataframe sorted by country_name, date with or without lower_window, upper_window
    """

    country_name = []
    ds = []
    holiday = []
    for yr in year_list:
        for country in country_list:
            for date, name in sorted(eval("holidays." + country)(years=yr).items()):
                country_name.append(country)
                ds.append(date)
                holiday.append(name)

    dat1 = pd.DataFrame({"country_name": country_name, "ds": ds, "holiday": holiday})
    dat2 = dat1.sort_values(['country_name', 'ds']).reset_index(drop=True)
    return createHolidayWindows(dat2, country_list, lower_window_list, upper_window_list)


def createHolidayWindows(data, country_list, lower_window_list, upper_window_list):
    """
    Parameters
    ----------
    country_list: list of country name
    lower_window_list: list of lower range of days around the date to be included as holidays
    upper_window_list: list of upper range of days around the date to be included as holidays
    Returns
    -------
    merge data with holiday data

    """

    tmpdat = pd.DataFrame({"country_name": country_list, "lower_window": lower_window_list, \
                           "upper_window": upper_window_list})

    return pd.merge(data, tmpdat, how="left", on=["country_name"])


class HolidayFeature(object):
    """
    Parameters
    ----------

    holidays: pd.DataFrame with columns holiday (string) and ds (date type)
        and optionally columns lower_window and upper_window which specify a
        range of days around the date to be included as holidays.
        lower_window=-2 will include 2 days prior to the date as holidays. Also
        optionally can have a column prior_scale specifying the prior scale for
        that holiday.

    holidays_prior_scale: Parameter modulating the strength of the holiday
        components model, unless overridden in the holidays input.


    """

    def __init__(self, holidays=None, holidays_prior_scale=10.0):
        self.holidays = holidays
        self.holidays_prior_scale = float(holidays_prior_scale)
        self.validate_inputs()

    def validate_inputs(self):

        """Validates the inputs to HolidayFeature"""

        if self.holidays is not None:

            if not (
                    isinstance(self.holidays, pd.DataFrame)
                    and 'ds' in self.holidays
                    and 'holiday' in self.holidays
            ):
                raise ValueError("holidays must be a DataFrame with 'ds' and "
                                 "'holiday' columns.")
            self.holidays['ds'] = pd.to_datetime(self.holidays['ds'])
            has_lower = 'lower_window' in self.holidays
            has_upper = 'upper_window' in self.holidays
            if has_lower + has_upper == 1:
                raise ValueError('Holidays must have both lower_window and ' +
                                 'upper_window, or neither')
            if has_lower:

                if self.holidays['lower_window'].max() > 0:
                    raise ValueError('Holiday lower_window should be <= 0')
                if self.holidays['upper_window'].min() < 0:
                    raise ValueError('Holiday upper_window should be >= 0')
            for h in self.holidays['holiday'].unique():
                self.validate_column_name(h, check_holidays=False)

    def validate_column_name(self, name, check_holidays=True):

        """Validates the name of a holiday
        Parameters
        ----------
        name: string
        check_holidays: bool check if name already used for holiday
        """
        if '_delim_' in name:
            raise ValueError('Name cannot contain "_delim_"')
        reserved_names = [
            'holidays', 'zeros'
        ]
        rn_l = [n + '_lower' for n in reserved_names]
        rn_u = [n + '_upper' for n in reserved_names]
        reserved_names.extend(rn_l)
        reserved_names.extend(rn_u)
        reserved_names.extend([
            'ds', 'y'])
        if name in reserved_names:
            raise ValueError('Name "{}" is reserved.'.format(name))
        if (check_holidays and self.holidays is not None and
                name in self.holidays['holiday'].unique()):
            raise ValueError(
                'Name "{}" already used for a holiday.'.format(name))
        if (check_holidays and self.country_holidays is not None and
                name in get_holiday_names(self.country_holidays)):
            raise ValueError(
                'Name "{}" is a holiday name in {}.'.format(name, self.country_holidays))
        return None

    def construct_holiday_dataframe(self, dates):
        """Construct a dataframe of holiday dates.

        Will combine self.holidays with the built-in country holidays
        corresponding to input dates, if self.country_holidays is set.

        Parameters
        ----------
        dates: pd.Series containing timestamps used for computing seasonality.

        Returns
        -------
        dataframe of holiday dates, in holiday dataframe format used in
        initialization.
        """
        all_holidays = pd.DataFrame()
        if self.holidays is not None:
            all_holidays = self.holidays.copy()
        if self.country_holidays is not None:
            year_list = list({x.year for x in dates})
            country_holidays_df = make_holidays_df(
                year_list=year_list, country=self.country_holidays
            )
            all_holidays = pd.concat((all_holidays, country_holidays_df), sort=False)
            all_holidays.reset_index(drop=True, inplace=True)
        # If the model has already been fit with a certain set of holidays,
        # make sure we are using those same ones.
        if self.train_holiday_names is not None:
            # Remove holiday names didn't show up in fit
            index_to_drop = all_holidays.index[
                np.logical_not(
                    all_holidays.holiday.isin(self.train_holiday_names)
                )
            ]
            all_holidays = all_holidays.drop(index_to_drop)
            # Add holiday names in fit but not in predict with ds as NA
            holidays_to_add = pd.DataFrame({
                'holiday': self.train_holiday_names[
                    np.logical_not(self.train_holiday_names.isin(all_holidays.holiday))
                ]
            })
            all_holidays = pd.concat((all_holidays, holidays_to_add), sort=False)
            all_holidays.reset_index(drop=True, inplace=True)
        return all_holidays

    def make_holiday_features(self, dates, holidays):

        """Construct a dataframe of holiday features.

        Parameters
        ----------
        dates: pd.Series containing timestamps used for computing seasonality.
        holidays: pd.Dataframe containing holidays, as returned by
            construct_holiday_dataframe.

        Returns
        -------
        holiday_features: pd.DataFrame with a column for each holiday.
        prior_scale_list: List of prior scales for each holiday column.
        holiday_names: List of names of holidays
        """
        # Holds columns of our future matrix.
        expanded_holidays = defaultdict(lambda: np.zeros(dates.shape[0]))
        prior_scales = {}
        # Makes an index so we can perform `get_loc` below.
        # Strip to just dates.
        row_index = pd.DatetimeIndex(dates.apply(lambda x: x.date()))

        for _ix, row in holidays.iterrows():
            dt = row.ds.date()
            try:
                lw = int(row.get('lower_window', 0))
                uw = int(row.get('upper_window', 0))
            except ValueError:
                lw = 0
                uw = 0
            ps = float(row.get('prior_scale', self.holidays_prior_scale))
            if np.isnan(ps):
                ps = float(self.holidays_prior_scale)
            if (
                    row.holiday in prior_scales and prior_scales[row.holiday] != ps
            ):
                raise ValueError(
                    'Holiday {} does not have consistent prior scale '
                    'specification.'.format(row.holiday))
            if ps <= 0:
                raise ValueError('Prior scale must be > 0')
            prior_scales[row.holiday] = ps

            for offset in range(lw, uw + 1):
                occurrence = dt + timedelta(days=offset)
                try:
                    loc = row_index.get_loc(occurrence)
                except KeyError:
                    loc = None

                key = '{}_delim_{}{}'.format(
                    row.holiday,
                    '+' if offset >= 0 else '-',
                    abs(offset)
                )
                if loc is not None:
                    expanded_holidays[key][loc] = 1.
                else:
                    # Access key to generate value
                    expanded_holidays[key]
        holiday_features = pd.DataFrame(expanded_holidays)
        # Make sure column order is consistent
        holiday_features = holiday_features[sorted(holiday_features.columns.tolist())]
        prior_scale_list = [
            prior_scales[h.split('_delim_')[0]]
            for h in holiday_features.columns
        ]
        holiday_names = list(prior_scales.keys())
        # Store holiday names used in fit
        if self.train_holiday_names is None:
            self.train_holiday_names = pd.Series(holiday_names)
        return holiday_features, prior_scale_list, holiday_names

    def add_country_holidays(self, country_name):

        """Add in built-in holidays for the specified country.

        These holidays will be included in addition to any specified on model
        initialization.

        Holidays will be calculated for arbitrary date ranges in the history
        and future. See the online documentation for the list of countries with
        built-in holidays.

        Built-in country holidays can only be set for a single country.

        Parameters
        ----------
        country_name: Name of the country, like 'UnitedStates' or 'US'

        Returns
        -------
        HolidayFeature object
        """
        if self.history is not None:
            raise Exception(
                "Country holidays must be added prior to model fitting."
            )
        # Validate names.
        for name in get_holiday_names(country_name):
            # Allow merging with existing holidays
            self.validate_column_name(name, check_holidays=False)
        # Set the holidays.
        if self.country_holidays is not None:
            logger.warning(
                'Changing country holidays from {} to {}'.format(
                    self.country_holidays, country_name
                )
            )
        self.country_holidays = country_name
        return self