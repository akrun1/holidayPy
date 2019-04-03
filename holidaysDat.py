from datetime import date
import holidays
import pandas as pd
import numpy as np


countries = ["UnitedStates", "Canada", "UnitedKingdom", "Germany", "France", "Switzerland"]
years = range(2014, 2020, 1)


def getHolidayListCountries(countryList, yearList):
    Country = []
    Date = []
    Holiday_Name = []
    for yr in yearList:
        for country in countryList:
            for date, name in sorted(eval("holidays." + country)(years=yr).items()):
                Country.append(country)
                Date.append(date)
                Holiday_Name.append(name)

    return pd.DataFrame({"Country": Country, "Date": Date, "Holiday_Name": Holiday_Name})


dat = getHolidayListCountries(countries, years)
