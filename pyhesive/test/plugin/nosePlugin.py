from nose2.events import Plugin

class overWrite(Plugin):
    '''
    This plugin exists solely to make nose let the following command line arguements through
    '''
    commandLineSwitch = (None, 'REPLACE', 'RRR')