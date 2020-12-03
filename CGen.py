#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 17:46:31 2020

@author: jacobfaibussowitsch
"""
import logging, atexit, meshio
from CGenOptionsDB import OptionsDataBase

class CGen:
    log = None
    OptCtx = None
    Mesh = None
    registered_exit = False

    def __init__(self, Opts):
        if not self.registered_exit:
            atexit.register(self.Finalize)
            self.registered_exit = True
        if not isinstance(Opts, OptionsDataBase):
            raise TypeError("Opts must be of type" + type(OptionsDataBase))
        slog = logging.getLogger(self.__class__.__name__)
        slog.setLevel(Opts.vLevel)
        slog.propagate = False
        ch = logging.StreamHandler(Opts.stream)
        ch.setLevel(Opts.vLevel)
        ch.propagate = False
        formatter = logging.Formatter('%(message)s')
        ch.setFormatter(formatter)
        slog.addHandler(ch)
        self.log = slog
        self.OptCtx = Opts

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            import traceback
            traceback.print_exception(exc_type, exc_value, tb)
        self.Finalize()
        return True

    def Finalize(self):
        self.Mesh.Finalize()
        self.OptCtx.Finalize()
        self.Mesh = None
        self.OptCtx = None
        handlers = self.log.handlers[:]
        for handler in handlers:
            handler.flush()
            handler.close()
            self.log.removeHandler(handler)
        self.log = None
        atexit.unregister(self.Finalize)
        self.registered_exit = False

    def Setup(self):
        from CGenMesh import Mesh
        msh = meshio.read(self.OptCtx.meshFileIn,
                          self.OptCtx.meshFormatIn)
        self.Mesh = Mesh(self.OptCtx, msh)
        self.Mesh.Setup()

    def PartitionMesh(self):
        self.Mesh.Partition()

    def GenerateElements(self):
        self.Mesh.ExtractLocalBoundaryElements()

    def OutputMesh(self):
        meshPoints, meshCells = self.Mesh.PrepareOutputMesh()
        meshOut = meshio.Mesh(meshPoints, meshCells)
        meshio.write(self.OptCtx.meshFileOut, meshOut, file_format=self.OptCtx.meshFormatOut)