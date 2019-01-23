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
parser.add_argument('--nseeds', help='Number of seeds', type=int, default=5)
args = parser.parse_args()

#seeds = [334, 837, 358,	191,	 683]
seeds = [385, 971, 482, 932, 508]

for seed in seeds[:args.nseeds]:    
    screen = su.Screen(args.name + '_' + str(seed), create=True)
    commands = [args.command + ' --seed %d' % seed + ' --name %s' % args.name]
    screen.send_commands(commands)