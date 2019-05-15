#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 10:49:19 2019

@author: matteo
"""

import argparse
import screenutils as su

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--name', help='First characters of screens to kill', type=str, default='hello')
args = parser.parse_args()

screens = su.list_screens()
for screen in screens:
    if screen.name.startswith(args.name):
        screen.kill()