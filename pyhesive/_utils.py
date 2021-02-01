#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 15:14:00 2021

@author: jacobfaibussowitsch
"""
import timeit, pstats, cProfile, functools, io, atexit

strm = io.StringIO()
perfFile = None

def pause():
    programPause = input("Press the <ENTER> key to continue...")
    print("Moving on...")

def timeFunc(reps):
    def decorator(func):
        @functools.wraps(func)
        def wraps(*args, **kwargs):
            t = timeit.Timer(lambda: func(*args, **kwargs))
            repeat = 10
            r = t.repeat(repeat, reps)
            best, worst = min(r), max(r)
            avg = sum(r)/repeat
            print("{repeat} sets of {reps} loops. AVG: {avg:.3g} usec/loop BEST: {best:.3g} usec/loop WORST: {worst:.3g} usec/loop".format(**vars()))
        return wraps
    return decorator


def finalizeProfile(fname):
    with open(fname, "w+") as perfFile:
        perfFile.write(strm.getvalue())

def profileFunc(func):
    @functools.wraps(func)
    def wraps(*args, **kwargs):
        if not args[0].perf: return func(*args, **kwargs)
        strm.write("-------- FUNC "+func.__name__+" -----------------------\n")
        profiler = cProfile.Profile()
        retval = profiler.runcall(func, *args, **kwargs)
        stats = pstats.Stats(profiler, stream=strm)
        stats.strip_dirs().sort_stats('cumtime').print_stats(50)
        return retval
    return wraps

def l2s(inlist):
    return ', '.join(map(str, inlist))

def flattenList(inlist):
        return [item for sublist in inlist for item in sublist]
