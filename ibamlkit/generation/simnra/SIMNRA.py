# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 12:49:58 2012

@author: 
"""

import win32com.client as wc
import pythoncom

class SIMNRA:
    def __init__(self):
        ''' initializes the SIMNRA object and connects with a SIMNRA instance '''
        pythoncom.CoInitialize()  # Initialize COM library
        self.App = wc.Dispatch('SIMNRA.app')
        self.Setup = wc.Dispatch('SIMNRA.setup')
        self.Target = wc.Dispatch('SIMNRA.target')
        self.Calc = wc.Dispatch('SIMNRA.calc')
        self.Fit = wc.Dispatch('SIMNRA.fit')
        self.Projectile = wc.Dispatch('SIMNRA.projectile')
        self.Spectrum = wc.Dispatch('SIMNRA.spectrum')
        self.Stopping = wc.Dispatch('SIMNRA.stopping')
        self.PIGE =  wc.Dispatch('SIMNRA.pige')
        self.CrossSec = wc.Dispatch('SIMNRA.crosssec')

    def __del__(self):
        pythoncom.CoUninitialize()  # Uninitialize COM library


