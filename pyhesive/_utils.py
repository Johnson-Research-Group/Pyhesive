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
  """
  Set the logging verbosity

  Parameter
  ---------
  log_level : int
    The logging verbosity

  Returns
  -------
  old_level : int
    The previous logging verbosity
  """
  global PYHESIVE_LOG_LEVEL
  old_level = PYHESIVE_LOG_LEVEL
  PYHESIVE_LOG_LEVEL = int(log_level)
  return old_level

def get_log_level():
  """
  Get the logging verbosity

  Returns
  -------
  level : int
    The logging verbosity
  """
  return PYHESIVE_LOG_LEVEL

def set_log_stream(stream):
  """
  Set the logging stream

  Parameter
  ---------
  stream : iostreamlike
    The new logging stream

  Returns
  -------
  old_stream : iostreamlike
    The previous logging stream

  Raises
  ------
  ValueError
    If stream is not flushable
  """
  try:
    stream.flush()
  except Exception as e:
    raise ValueError("Could not flush stream") from e

  global PYHESIVE_LOG_STREAM
  PYHESIVE_LOG_STREAM.flush()
  old_stream = PYHESIVE_LOG_STREAM
  PYHESIVE_LOG_STREAM = stream
  return old_stream

def get_log_stream():
  """
  Get the logging stream

  Returns
  -------
  stream : iostreamlike
    The logging stream
  """
  return PYHESIVE_LOG_STREAM

def flatten(in_list):
  return [item for sub_list in in_list for item in sub_list]
