
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 18:17:39 2023

@author: delta
"""
from abc import ABC, abstractmethod, abstractclassmethod, abstractstaticmethod, abstractproperty
#,update_abstractmethods

class DATA(ABC):
    @abstractmethod
    def data_generation_from_table(self):
        pass
    """
    @abstractmethod
    def data_generation_from_cloud(self, arg1):
        pass
    
    @abstractmethod
    def data_generation_from_web(self, arg1):
        pass

    @abstractmethod
    def data_generation_from_multiple_sources(self, arg1):
        pass

    @abstractmethod
    def data_generation_from_streaming_data(self, arg1):
        pass

    @abstractmethod
    def data_generation_from_geophysics(self, arg1):
        pass

    @abstractmethod
    def data_generation_from_time_series(self, arg1):
        pass

    @abstractmethod
    def data_generation_from_synthetic_data(self, arg1):
        pass
    
    @abstractmethod
    def data_generation_to_supervised_methods(self,arg1):
        pass
    """
