import pandas as pd
import os
import datetime as dt
import numpy as np
import matplotlib.pyplot as pt
import math as math
from statistics import mean
import copy


class data_collector:

    def __init__(self):
        self.equities  =  self.get_equities()
        self.fixd_incme = self.get_fixd_incme()


    def get_equities(self):
        """
        Seed data tables with equity data
        :param date_time start_date: initial date
        :param date_time end_date: ending date date
        :param float interval: day count for desired interval.
        :return:dict equities: a dictonary of data frames for each s&p 500 sector
        """
        equities = {"communications":pd.read_csv(os.path.join("Data","Equities","S&P 500 Communications Services Sector Index.csv")),
                "cnsmr_discr":pd.read_csv(os.path.join("Data","Equities","S&P 500 Consumer Discretionary Sector Index.csv")),
                "cnsmr_stap":pd.read_csv(os.path.join("Data","Equities","S&P 500 Consumer Staples Sector Index.csv")),
                "engry":pd.read_csv(os.path.join("Data","Equities","S&P 500 Energy Sector Index.csv")),
                "fincals":pd.read_csv(os.path.join("Data","Equities","S&P 500 Financials Sector GICS Level 1 Index .csv")),
                "health":pd.read_csv(os.path.join("Data","Equities","S&P 500 Health Care Sector Index.csv")),
                "indtsry" :pd.read_csv(os.path.join("Data", "Equities","S&P 500 Industrials Sector Index.csv")),
                "information":pd.read_csv(os.path.join("Data","Equities","S&P 500 Information Technology Sector Index.csv")),
                "materials":pd.read_csv(os.path.join("Data","Equities","S&P 500 Materials Sector Index.csv")),
                "real_estate":pd.read_csv(os.path.join("Data","Equities","S&P 500 Real Estate Sector Index.csv")),
                "utilities":pd.read_csv(os.path.join("Data","Equities","S&P 500 Utilities Sector Index.csv"))}
        for k,value in equities.items():
            value = value.rename(columns = lambda x: x.strip())
            value = value.iloc[::-1]
            value['daily_return_percent'] = (value["Close"] - value["Close"].shift(1))/value["Close"].shift(1)*100
            equities[k] = value
        for value in equities.values():
            value["Date"] = pd.to_datetime(value["Date"])


        return equities

    def get_fixd_incme(self):
        """
        Seed data tables with fixed income data
        :return:dict equities: a dictonary of data frames for each fixed income asset
        """
        fixed_income = {"short_corps_hy":pd.read_csv(os.path.join("Data","Fixed Income",
                                                                  "Vanguard Short-Term Corporate "
                                                                  "Bond Index Fund ETF Shares (VCSH).csv")),
                "mid_corps_ig":pd.read_csv(os.path.join("Data","Fixed Income",
                                                        "iShares 5-10 Year Investment Grade Corporate Bond ETF.csv")),
                "short_corps_ig":pd.read_csv(os.path.join("Data","Fixed Income",
                                                          "iShares 5-10 Year Investment Grade Corporate Bond ETF.csv")),
                "10-20_t-nts":pd.read_csv(os.path.join("Data","Fixed Income","iShares 10-20 Year Treasury Bond ETF.csv")),
                "20+_t-nts":pd.read_csv(os.path.join("Data","Fixed Income","iShares 20+ Year Treasury Bond ETF.csv")),
                "1-3_year_t_bond":pd.read_csv(os.path.join("Data","Fixed Income","iShares 1-3 Year Treasury Bond ETF.csv")),
                "lng_term_corps":pd.read_csv(os.path.join("Data","Fixed Income","iShares Long-Term Corporate Bond ETF (IGLB).csv")),
               }
        for k,value in fixed_income.items():
            value = value.rename(columns = lambda x: x.strip())
            value = value.iloc[::-1]
            value['daily_return_percent'] = value['Close'] - value["Close"].shift(1)
            fixed_income[k] = value
        for value in fixed_income.values():
            value["Date"] = pd.to_datetime(value["Date"])
        return fixed_income

    def date_filter(self,start_date, end_date, asset):
        """
        Method to standerdize tables by date
        :param date_time start_date:
        :param date_time end_date:
        :param str equities:
        :param str fixed_income:
        :param pd.dataframe single_asset:
        :return: dataframe of assets
        """
        frame = {'equities':self.equities,'bonds':self.fixd_incme}
        data = copy.deepcopy(frame[asset])
        start = start_date
        end = end_date
        for k,val in data.items():
            val.drop(val[val['Date'] < start].index, inplace=True)
            val.drop(val[val['Date'] > end].index, inplace=True)
        return data


    def get_weights(self, asset):
        """
        This method finds the inverse standard deviation for each asset
        :param asset:
        :return: weight for portfolio
        """
        window = asset.head(200)
        returns_as_array = window['daily_return_percent']
        stnd = np.std(returns_as_array)
        return 1/stnd

    def portfolio(self,asset_class, type ):
        """
        Method to allocate over an asset class.
        :param dict asset_class:
        :param str type: equities or bonds
        :return: list weight vector
        """
        weight_vector = [type]
        for k,v in asset_class.items():
            weight_vector.append([k,self.get_weights(v)])
        sum = 0
        for i in weight_vector[1:]:
            sum+= i[1]

        for i in weight_vector[1:]:
            i[1] = i[1]/sum
        return weight_vector

    def book_init(self, cash, alpha):
        equities = self.equities
        bonds    = self. fixd_incme
        eq_weight = self.portfolio(asset_class = equities, type = 'equities')
        bd_weight = self.portfolio(asset_class = bonds, type = 'bonds')
        eq_book = ['equities']
        bd_book = ['bonds']
        eq_allocation = alpha * cash
        bd_allocation = (1-alpha)*cash
        for i in eq_weight[1:]:
            i[1] = eq_allocation*i[1]
            eq_book.append(i)
        for i in bd_weight[1:]:
            i[1] = bd_allocation * i[1]
            bd_book.append(i)
        return eq_book, bd_book

    def update_book(self, book, date_start, date_end):
        start_value = 0
        for i in book[1:]:
            start_value += i[1]
        stds = []
        data = self.date_filter(start_date = date_start, end_date = date_end, asset = book[0])
        for i,(k,v) in enumerate(data.items()):
            v['daily_return_percent'] = v['daily_return_percent'].fillna(0)
            percent_change = v['daily_return_percent'].to_numpy(dtype = float)/100
            stds.append(np.std(percent_change)*100)
            for j in percent_change:
                book[i+1][1] += book[i+1][1]*j
        end_value = 0
        for i in book[1:]:

            end_value += i[1]
        ret_perct = (end_value - start_value)/start_value*100
        return book,ret_perct,start_value,end_value, mean(stds)

    def reallocate(self, alpha, eq_book,bd_book,cash = 10000000):
        return self.book_init(alpha = alpha, cash = cash)

    def total_return(self, start, end, eq_book, bd_book):
        book_eq,per_eq,start_val_eq,end_val_eq, std_eq = self.update_book(eq_book, start, end)
        book_bd, per_bd,start_val_bd, end_val_bd,std_bd = self.update_book(bd_book,start,end)
        total_return = ((end_val_bd+end_val_eq)-(start_val_bd+start_val_eq))/(start_val_bd+start_val_eq)
        return total_return,end_val_eq,end_val_bd,eq_book,bd_book





