'''
Simple vehicle detection implementation that determines if there is a vehicle
ahead, calculates the distances to which and determines if we're encroaching on
the vehicle ahead.
'''
import cv2
import numpy as np


def object_ahead(lane):
    '''
    Determine whether or not there is an object or vehicle ahead on the projected path
    @params:
        lane: image of the projected path for the vehicle
    @returns:
        block: true if there is object ahead or false otherwise
    '''
    print('Determining if there is object ahead...')
    #TODO Implement


def calc_distance():
    '''
    Calculate the distance to the object ahead and cache it (or return? - TBD)
    @params:
        TBD
    @returns:
        None
    '''
    print('Calculating distance to object ahead...')
    #TODO Implement


def collate_velocity():
    '''
    Decipher whether or not we are encroaching on the vehicle ahead.
    @params:
        TBD
    @returns:
        None
    '''
    print('Analyzing velocity to determine encroachment...')
    #TODO Implement


def detect_vehicles(lane):
    '''
    Detect vehicles
    @params:
        lane: segmented lane image of projected path
    @returns:
        TBD
    '''
    print('Initiating vehicle detection...')
    if object_ahead(lane):
        calc_distance()
