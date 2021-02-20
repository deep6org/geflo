#blueberry pod pipe to fft

#connect to selected pod
#stream data into buffer
#perform fft on window 

#MH notes: 
#provide output x e.g. 10 bins, some distribution across 10 buckets
#time for fft - 256 - 512 - 10 - 128, 128 samples
#3 - 4 freq. bins - qnode into penny lane
#displacement in penny lane 

#sequential learning 
#chemistry -
#CLI interface of some format 

#connection code
#https://github.com/blueberryxtech/BlueberryPython

#log power bins
#https://github.com/skylarkwireless/sklk-demos/blob/7f1cfd0974c88ff9633e5dd077b69951b653edb5/python/sklk_widgets/LogPowerFFT.py

# -*- coding: utf-8 -*-
"""
cayden, Blueberry
hbldh <henrik.blidh@gmail.com>, BLEAK
"""

import sys
import logging
import asyncio
import platform
import bitstring
import argparse
import time
import asyncio
import math
from numpy import fft
import numpy as np
import scipy.signal
import pandas as pd
import array as A

from bleak import BleakClient 
from bleak import _logger as loggers
from bleak import discover

#Blueberry glasses GATT server characteristics information
bbxService={"name": 'fnirs service',
            "uuid": '0f0e0d0c-0b0a-0908-0706-050403020100' }
bbxchars={
          "commandCharacteristic": {
              "name": 'write characteristic',
                  "uuid": '1f1e1d1c-1b1a-1918-1716-151413121110',
                  "handles": [None],
                    },
            "shortFnirsCharacteristic": {
                    "name": 'short_path',
                        "uuid": '2f2e2d2c-2b2a-2928-2726-252423222120',
                        "handles": [19, 20, 27],
                          },
            "longFnirsCharacteristic": {
                    "name": 'long_path',
                        "uuid": '3f3e3d3c-3b3a-3938-3736-353433323130',
                        "handles": [22, 23, 31],
                          }

            }
SHORT_PATH_CHAR_UUID = bbxchars["shortFnirsCharacteristic"]["uuid"]
LONG_PATH_CHAR_UUID = bbxchars["longFnirsCharacteristic"]["uuid"]

stream = True
save = False
debug = False
save_file = None

#buffers for computation
#use mag of 3 for first pass
temp = []
short_mag_buff = np.array(list)
long_mag_buff = np.array(list)
buff_size = 128

df_fft_short = pd.DataFrame()
df_fft_long = pd.DataFrame()

#refer to select_pod.py to find address/mac
pod_address = "56FFAB79-ACF1-4E4B-85B7-ED0C0A199973"

#unpack fNIRS byte string
def unpack_fnirs(sender, packet):
    global bbxchars
    data = dict()
    data["path"] = None
    #figure out which characteristic sent it (using the handle, why do we have UUID AND handle?)
    for char in bbxchars:
        if sender in bbxchars[char]["handles"]:
            data["path"] = bbxchars[char]["name"]
            break
        elif type(sender) == str and sender.lower() == bbxchars[char]["uuid"]:
            data["path"] = bbxchars[char]["name"]
            break
    if data["path"] == None:
        # print("Error unknown handle number: {}. See: https://github.com/blueberryxtech/BlueberryPython/issues/1 or reach out to cayden@blueberryx.com".format(sender))
        return None
    #unpack packet
    aa = bitstring.Bits(bytes=packet)
    if data["path"] == "long_path" and len(packet) >= 21:
        pattern = "uintbe:8,uintbe:8,intbe:32,intbe:32,intbe:32,uintbe:8,uintbe:8,uintbe:8,uintbe:8,uintbe:8,intbe:16"
        res = aa.unpack(pattern)
        data["packet_index"] = res[1]
        data["channel1"] = res[2] #740/940
        data["channel2"] = res[3] #880
        data["channel3"] = res[4] #850
        data["sp"] = res[5]
        data["dp"] = res[6]
        data["hr"] = res[7]
        data["hrv"] = res[8]
        data["ml"] = res[9]
        data["temperature"] = res[10]
        data["big"] = True #big: whether or not the extra metrics were packed in
    else:
        pattern = "uintbe:8,uintbe:8,intbe:32,intbe:32,intbe:32,uintbe:8,uintbe:8"
        res = aa.unpack(pattern)
        data["packet_index"] = res[1]
        data["channel1"] = res[2] #740/940
        data["channel2"] = res[3] #880
        data["channel3"] = res[4] #850
        data["big"] = False #big: whether or not the extra metrics were packed in
    return data

def notification_handler(sender, data):
    global save, debug, short_mag_buff, long_mag_buff
    """Simple notification handler which prints the data received."""
    data = unpack_fnirs(sender, data)
    idx = data["packet_index"]
    path = data["path"]

    c1 = data["channel1"]
    c2 = data["channel2"]
    c3 = data["channel3"]

    if data["path"] == "long_path" and data["big"] == True:
        sp = data["sp"]
        dp = data["dp"]
        hr = data["hr"]
        hrv = data["hrv"]
        ml = data["ml"]
        temperature = data["temperature"]

    if data["path"] == "long_path" and data["big"] == True:
            save_file.write("{},{},{},{},{},{}\n".format(time.time(), idx, path, c1, c2, c3))
    else:
            save_file.write("{},{},{},{},{},{}\n".format(time.time(), idx, path, c1, c2, c3))

    if data["path"] == "long_path" and data["big"] == True:
    	mag = math.sqrt( c1*c1 + c2*c2 + c3*c3 )
    	np.append(long_mag_buff, mag, axis=None)
    	# long_mag_buff = long_mag_buff.squeeze()
    	if long_mag_buff.size == 128:
    		compute_fft(long_mag_buff, "long")
        # print("Blueberry: {}, path: {}, index: {}, C1: {}, C2: {}, C3: {}, SP : {}, DP : {}, HR : {}, HRV : {}, ML : {}, temperature : {},".format(sender, path, idx, c1, c2, c3, sp, dp, hr, hrv, ml, temperature))
    else:
    	mag = math.sqrt( c1*c1 + c2*c2 + c3*c3 )
    	np.append(short_mag_buff, mag, axis=None)
    	if short_mag_buff.size == 128:
    		compute_fft(long_mag_buff, "short")
        # print("Blueberry: {}, path: {}, index: {}, C1: {}, C2: {}, C3: {}".format(sender, path, idx, c1, c2, c3))

def LogPowerFFT(samps, tag, peak=1.0, reorder=True, window=None):
    """
    Calculate the log power FFT bins of the complex samples.
    @param samps numpy array of complex samples
    @param peak maximum value of a sample (floats are usually 1.0, shorts are 32767)
    @param reorder True to reorder so the DC bin is in the center
    @param window function or None for default flattop window
    @return an array of real values FFT power bins
    """
    size = len(samps)
    numBins = size

    #scale by dividing out the peak, full scale is 0dB
    scaledSamps = samps/peak

    #calculate window
    if not window: window = scipy.signal.hann
    windowBins = window(size)
    windowPower = math.sqrt(sum(windowBins**2)/size)

    #apply window
    windowedSamps = np.multiply(windowBins, scaledSamps)

    #window and fft gain adjustment
    gaindB = 20*math.log10(size) + 20*math.log10(windowPower)

    #take fft
    fftBins = np.abs(np.fft.fft(windowedSamps))
    fftBins = np.maximum(fftBins, 1e-20) #clip
    powerBins = 20*np.log10(fftBins) - gaindB

    #bin reorder
    if reorder:
        idx = np.argsort(np.fft.fftfreq(len(powerBins)))
        powerBins = powerBins[idx]
    else:
    	""

    print("bins: " + powerBins)

    if tag == "short":
    	short_mag_buff = np.array([])
    else:
    	long_mag_buff = np.array([])
    # if tag == "short":
    # 	print("short bins: " + powerBins)
    # 	df_fft_short.append(powerBins, ignore_index=False)
   	# else:
   	# 	print("long bins: " + powerBins)
   	# 	df_fft_long.append(powerBins, ignore_index=False)

    return powerBins

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-a","--address", help="MAC address of the blueberry")
    # parser.add_argument("-s","--save", help="If present, save", action='store_true')
    # parser.add_argument("-f","--filename", help="Name of file to save to", type=str)
    # parser.add_argument("-d", "--debug", help="debug", action='store_true')
    # args = parser.parse_args()

    #get address
    # mac = args.address

    #should we debug?
    # if args.debug:
    #     debug = True

    #if we should save, and make the save file
    #for build should save all data!
    # if args.save:
    # save = True
    # if not args.filename or args.filename == "":
    save_file = open("./data/{}.csv".format(time.time()), "w+")
    # else:
        # save_file = open(args.filename, "w+")
    save_file.write("timestamp,idx,path,c1,c2,c3\n")

    #translate address to be multi-platform
    # address = (
    #     mac # <--- Change to your device's address here if you are using Windows or Linux
    #     if platform.system() != "Darwin"
    #     else mac # <--- Change to your device's address here if you are using macOS
    # )

    async def run(address, debug=False):
	    global stream

	    if debug:
	        l = logging.getLogger("asyncio")
	        l.setLevel(logging.DEBUG)
	        # h = logging.StreamHandler(sys.stdout)
	        # h.setLevel(logging.DEBUG)
	        # l.addHandler(h)
	        # logger.addHandler(h)

	    print("Trying to connect...")
	    async with BleakClient(address) as client:
	        x = await client.is_connected()
	        print("connected!")

	        await client.start_notify(SHORT_PATH_CHAR_UUID, notification_handler)
	        await client.start_notify(LONG_PATH_CHAR_UUID, notification_handler)
	        while stream:
	            await asyncio.sleep(0.1)
	        await client.stop_notify(CHARACTERISTIC_UUID)

    #start main loop
    loop = asyncio.get_event_loop()
    # loop.set_debug(True)
    loop.run_until_complete(run(pod_address, debug=True))
