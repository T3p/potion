#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 23:18:56 2019

@author: matteo
"""

import argparse
import screenutils as su

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--command', help='Command', type=str, default='echo hello')
parser.add_argument('--name', help='Name', type=str, default='hello')
parser.add_argument('--nseeds', help='Number of seeds', type=int, default=10)
args = parser.parse_args()

seeds = [2030, 4473, 5759, 5756, 4146, 1428, 9723, 3212, 8589, 1971]

for seed in seeds[:args.nseeds]:    
    screen = su.Screen(args.name + '_' + str(seed), create=True)
    commands = [args.command + ' --seed %d' % seed + ' --name %s' % args.name]
    screen.send_commands(commands)