#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 15:14:00 2021

@author: jacobfaibussowitsch
"""
import sys

PYHESIVE_LOG_LEVEL  = 50
PYHESIVE_LOG_STREAM = sys.stdout

def set_log_level(log_level):
  assert isinstance(log_level,int)
  global PYHESIVE_LOG_LEVEL
  PYHESIVE_LOG_LEVEL = log_level
  return PYHESIVE_LOG_LEVEL

def get_log_level():
  return PYHESIVE_LOG_LEVEL

def set_log_stream(stream):
  global PYHESIVE_LOG_STREAM
  PYHESIVE_LOG_STREAM.flush()
  PYHESIVE_LOG_STREAM = stream
  return PYHESIVE_LOG_STREAM

def get_log_stream():
  return PYHESIVE_LOG_STREAM

def flatten(in_list):
  return [item for sub_list in in_list for item in sub_list]
