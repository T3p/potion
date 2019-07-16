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

screen = su.Screen(args.name, create=True)

seeds = [507, 160, 649, 144, 233]

commands = [args.command + ' --seed %d' % seed + ' --name %s' % args.name for seed in seeds[:args.nseeds]]

screen.send_commands(commands)