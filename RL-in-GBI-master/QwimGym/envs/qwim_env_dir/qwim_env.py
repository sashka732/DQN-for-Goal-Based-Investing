import gym
from Data import data_collector
from gym import spaces
import numpy as np
import datetime as dt



class QwimEnv(gym.Env):
    action_space = spaces.Discrete(10)
    observation_space = spaces.Box(low = 0, high = np.infty, shape = (1,3) )

    def __init__(self, cash, alpha_init):
        super(QwimEnv,self).__init__()
        self.book_worker  = data_collector.data_collector()
        self.equities = self.book_worker.equities
        self.bonds = self.book_worker.fixd_incme
        self.eq_book,self.bd_book = self.book_worker.book_init(cash = cash, alpha = alpha_init)
        self.alpha_init = alpha_init
        self.cash = cash
        self.start_date_init = dt.datetime(year = 2010, month = 1, day = 3)
        self.start_date = self.start_date_init
        self.end_date_init   = self.start_date_init + dt.timedelta(days = 7)


    def reset(self):
        eq_book, bd_book = self.book_worker.book_init(cash = 10000000, alpha = self.alpha_init)
        self.state = [10000000, .5, .5]




    def step(self,action):
        """
        TODO: action should mamash be the integer (index), and the environment should pass that to the book worker for
        it to interpret
        :param action:
        :return:
        """
        start_date = self.start_date
        end_date  = start_date + dt.timedelta(days = 7)
        action_dict = {i: x for i, x in enumerate(np.arange(0., 1.0, 0.1) + 0.1)}
        reward, eq_val, bd_val ,eq_book,bd_book = self.book_worker.total_return(start = start_date, end = end_date,
                                                                                eq_book = self.eq_book, bd_book = self.bd_book)
        cash = eq_val+bd_val
        eq,bd = self.book_worker.reallocate(alpha = action_dict[action],cash = cash,eq_book = eq_book,bd_book = bd_book)
        self.state = [cash, eq_val/cash, bd_val/cash]
        self.start_date = end_date
        self.eq_book = eq
        self.bd_book = bd
        done = False
        if end_date > dt.datetime(year = 2020, month = 9, day = 3):
            done = True  # figure out how to tell when the sim is over

        # return state, reward, done, info
        return self.state, reward, done, {}

    def render(self, mode='human'):
        pass