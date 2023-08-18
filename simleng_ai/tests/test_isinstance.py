#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 20:34:00 2023

@author: agagora
"""


class pw:
    def __init__(self, shape=None):
        self.GenLogit_shape = shape

    def get_c_value(self):
        self.c = self.GenLogit_shape
        print(self.c)
        return self.c


print(pw(2).get_c_value())
