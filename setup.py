# -*- coding: utf-8 -*-
"""
Created on Mon May  9 15:24:21 2016

@author: marcobarsacchi

This file is part of MixtureDP.

MixtureDP is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

MixtureDP is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with MixtureDP.  If not, see <http://www.gnu.org/licenses/>.
"""

from setuptools import setup

__version__='0.1'

setup(
    name='mixtureDP',
    version=__version__,
    packages=['mixtureDP',],
    license='GPL',
    long_description=open('README.md').read(),
)