#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 17:45:21 2020

@author: jacobfaibussowitsch
"""
import sys, atexit, argparse, gc, io

class OptionsDataBase:
    meshFileIn = None
    vLevel = 50
    registered_exit = False

    def __init__(self):
        self.vLevel = 50
        if not self.registered_exit:
            atexit.register(self.Finalize)
            self.registered_exit = True

    def Finalize(self):
        if isinstance(self.stream, io.IOBase):
            self.stream.flush()
        atexit.unregister(self.Finalize)
        self.registered_exit = False

def OptionsInitialize():
    gc.set_debug(gc.DEBUG_UNCOLLECTABLE | gc.DEBUG_SAVEALL)
    args = OptionsDataBase()
    path = "<path>"
    string = "<string>"
    integer = "<int>"
    defaultFormat = "gmsh"
    defaultName = "cgenOutputMesh"
    parser = argparse.ArgumentParser(description="Insert Cohesive Elements Into Mesh", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', required=True, metavar=path, help="Specify the input mesh file", dest='meshFileIn')
    parser.add_argument('-b', '--num-partitions', metavar=integer, default=3, type=int, help="Specify the number of partitions to make of the mesh", dest='numPart')
    parser.add_argument('--output', nargs='?', const=defaultName+"."+defaultFormat, default=defaultName+"."+defaultFormat, metavar=path, help='Specify the output mesh file', dest='meshFileOut')
    parser.add_argument('-fi', '--input-format', metavar=string, help='Specify the input mesh file format', dest='meshFormatIn')
    parser.add_argument('-fo', '--output-format', nargs='?', const=defaultFormat,  default=defaultFormat, metavar=string, help='Specify the output mesh file format', dest='meshFormatOut')
    parser.add_argument('-l', '--log-file', nargs='?', default=sys.stdout.name, metavar=path, help='Log output to file instead of STDOUT', dest='stream')
    parser.add_argument('-v', '--verbose', default=1, action='count', help='Increase verbosity of logging statements, default no logging', dest='verbosity')
    parser.parse_args(namespace=args)
    if args.stream is not sys.stdout.name:
        args.stream = open(args.stream[0], 'w')
    else:
        args.stream = sys.stdout
    for verb in range(args.verbosity):
        if args.vLevel >= 20:
            args.vLevel -= 10
    return args