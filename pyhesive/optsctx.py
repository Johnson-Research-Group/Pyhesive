#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 17:45:21 2020

@author: jacobfaibussowitsch
"""
import sys, atexit, argparse, io, os

class Optsctx:
    stream = sys.stdout.name
    vLevel = 50
    registered_exit = False

    def __init__(self, cliRequire=True):
        if not self.registered_exit:
            atexit.register(self.Finalize)
            self.registered_exit = True
        path = "<path>"
        string = "<string>"
        integer = "<int>"
        boolean = "<bool>"
        defaultFormat = "abaqus"
        defaultName = "pyhesiveOutput_{meshname}"
        parser = argparse.ArgumentParser(description="Insert Cohesive Elements Into Mesh", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('-i', '--input', required=cliRequire, metavar=path, help="Specify the input mesh file", dest='meshFileIn')
        parser.add_argument('-b', '--num-partitions', metavar=integer, default=3, type=int, help="Specify the number of partitions to make of the mesh", dest='numPart')
        parser.add_argument('-o', '--output', nargs='?', const=defaultName+"."+defaultFormat, default=defaultName+"."+defaultFormat, metavar=path, help='Specify the output mesh file', dest='meshFileOut')
        parser.add_argument('-fi', '--input-format', metavar=string, help='Specify the input mesh file format', dest='meshFormatIn')
        parser.add_argument('-fo', '--output-format', nargs='?', const=defaultFormat,  default=defaultFormat, metavar=string, help='Specify the output mesh file format', dest='meshFormatOut')
        parser.add_argument('-l', '--log-file', nargs='?', default=sys.stdout.name, metavar=path, help='Log output to file instead of STDOUT', dest='stream')
        parser.add_argument('-v', '--verbose', default=1, action='count', help='Increase verbosity of logging statements, default no logging', dest='verbosity')
        parser.add_argument('-p', '--performance', default=False, help='Profile the code during creation', dest='perf')
        parser.parse_args(namespace=self)
        if self.meshFileIn is not None:
            _, filen = os.path.split(self.meshFileIn)
            filename = filen.split('.')[0]
        else:
            fiename = "mesh"
        self.meshFileOut = self.meshFileOut.format(meshname=filename)
        if self.stream is not sys.stdout.name:
            self.stream = open(self.stream[0], 'w')
        else:
            self.stream = sys.stdout
        for verb in range(self.verbosity):
            if self.vLevel >= 20:
                self.vLevel -= 10

    def Finalize(self):
        if isinstance(self.stream, io.IOBase):
            self.stream.flush()
        atexit.unregister(self.Finalize)
        self.registered_exit = False
