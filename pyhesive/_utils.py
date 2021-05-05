#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 15:14:00 2021

@author: jacobfaibussowitsch
"""
import sys

PYHESIVE_LOG_LEVEL  = 50
PYHESIVE_LOG_STREAM = sys.stdout

def setLogLevel(logLevel):
  global PYHESIVE_LOG_LEVEL
  assert isinstance(logLevel,int)
  PYHESIVE_LOG_LEVEL = logLevel
  return

def getLogLevel():
  return PYHESIVE_LOG_LEVEL

def setLogStream(stream):
  global PYHESIVE_LOG_STREAM
  PYHESIVE_LOG_STREAM.flush()
  PYHESIVE_LOG_STREAM = stream
  return

def getLogStream():
  return PYHESIVE_LOG_STREAM

def flatten(inlist):
  return [item for sublist in inlist for item in sublist]
