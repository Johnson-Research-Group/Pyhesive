#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 17:45:21 2020

@author: jacobfaibussowitsch
"""
import sys, atexit, argparse, gc

class OptionsDataBase:
    MeshFileIn = None
    MeshFormatIn = None
    verbosity = 1
    vlevel = 50
    stream = sys.stdout.name
    registered_exit = False

    def __init__(self):
        self.vlevel = 50
        if not self.registered_exit:
            atexit.register(self.Finalize)
            self.registered_exit = True

    def Finalize(self):
        self.stream.flush()
        atexit.unregister(self.Finalize)
        self.registered_exit = False

def OptionsInitialize():
    gc.set_debug(gc.DEBUG_UNCOLLECTABLE | gc.DEBUG_SAVEALL)
    args = OptionsDataBase()
    path = "<path>"
    string = "<string>"
    parser = argparse.ArgumentParser(description='Insert Cohesive Elements Into Mesh', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', required=True, metavar=path, help='Specify the input mesh file', dest='MeshFileIn')
    parser.add_argument('-f', '--format', required=False, metavar=string, help='Specify the input mesh file format', dest='MeshFormatIn')
    parser.add_argument('-l', '--log-file', nargs=1, default=sys.stdout.name, metavar=path, help='Log output to file instead of STDOUT', dest='stream')
    parser.add_argument('-v', '--verbose', default=0, action='count', help='Increase verbosity of logging statements, default no logging', dest='verbosity')
    parser.parse_args(namespace=args)
    if args.stream is not sys.stdout.name:
        args.stream = open(args.stream[0], 'w')
    else:
        args.stream = sys.stdout
    for verb in range(args.verbosity):
        if args.vlevel >= 20:
            args.vlevel -= 10
    return args