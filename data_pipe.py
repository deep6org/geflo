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
import numpy as np
import scipy.signal
import pandas as pd
import array as A

from scipy.fftpack import fft,ifft
from scipy.io.wavfile import read,write
from scipy.signal import get_window
from pdb import set_trace
from os.path import join,split
from glob import glob

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
buff_count_short = 0
buff_count_long = 0

#buffers for computation
#use mag of 3 for first pass
short_mag_buff = np.zeros(128)
long_mag_buff = np.zeros(128)
buff_size = 128

energyB1_s = 0.0
energyB2_s = 0.0
energyB3_s = 0.0

energyB1_l = 0.0
energyB2_l = 0.0
energyB3_l = 0.0

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
	global save, debug, short_mag_buff, long_mag_buff, buff_count_long, buff_count_short, energyB1_s, energyB2_s, energyB3_s, energyB1_l, energyB2_l, energyB3_l
	"""Simple notification handler which prints the data received."""
	data = unpack_fnirs(sender, data)
	idx = data["packet_index"]
	path = data["path"]

	c1 = data["channel1"]
	c2 = data["channel2"]
	c3 = data["channel3"]

	if data["path"] == "long_path" and data["big"] == True:
		buff_count_long += 1
		sp = data["sp"]
		dp = data["dp"]
		hr = data["hr"]
		hrv = data["hrv"]
		ml = data["ml"]
		temperature = data["temperature"]
	else:
		buff_count_short += 1

	if data["path"] == "long_path" and data["big"] == True:
		save_file.write("{},{},{},{},{},{},{},{},{}\n".format(time.time(), idx, path, c1, c2, c3, energyB1_l, energyB2_l, energyB3_l))
	else:
		save_file.write("{},{},{},{},{},{},{},{},{}\n".format(time.time(), idx, path, c1, c2, c3, energyB1_s, energyB2_s, energyB3_s))

	mag = math.sqrt( c1*c1 + c2*c2 + c3*c3 )
	if data["path"] == "long_path" and data["big"] == True:
		long_mag_buff = np.roll(long_mag_buff, -1)
		np.put(long_mag_buff, -1, mag)
	else:
		short_mag_buff = np.roll(short_mag_buff, -1)
		np.put(short_mag_buff, -1, mag)
			
	if (buff_count_long == 128):
		getFFTEnergyBins(long_mag_buff, "long")
		buff_count_long = 0

	if (buff_count_short == 128):
		getFFTEnergyBins(short_mag_buff, "short")
		buff_count_short = 0

def triang_win(width,center=0.5):
    win = []
    cpos = center * width
    for i in range(width + 1):
        if i <= cpos:
            win.append(1.0 / cpos * i)
        else:
            win.append(float(width - i) / (width - cpos))
    return np.array(win)[0:width]

def getFFTEnergyBins(data, tag):
	global energyB1_s, energyB2_s, energyB3_s, energyB1_l, energyB2_l, energyB3_l

	num_bands = 3
	rate = 10
	
	if len(data)%2==0:
		band_width = (len(data)+2)/(num_bands+1)
	else:
		band_width = (len(data)+1)/(num_bands+1)

	if len(data)%2==0:
		spectrum = fft(data)[0:len(data)//2+1]
	else:
		spectrum = fft(data)[0:(len(data)-1)//2+1]
		
	linear_step = rate/2/num_bands
	linear_center = [0.0]+list(map(lambda i:(i+0.5)*linear_step,range(num_bands)))+[rate/2]
	banks = []
	if len(data)%2==0:
		freq_unit = rate/(len(data)+2)
	else:
		freq_unit = rate/(len(data)+1)
	for i in range(num_bands):
		length = linear_center[i+2]-linear_center[i]
		center = (linear_center[i+1]-linear_center[i])/length
		win_size = int(length/freq_unit)
		banks.append(triang_win(win_size,center))
	energy = []
	for i in range(num_bands):
		start = int(linear_center[i]/freq_unit)
		energy.append(sum(list(map(lambda x:np.power(np.abs(x),2),spectrum[start:start + len(banks[i])] * banks[i]))))

	print(str(energy[0]) + " " + str(energy[1]) + " " + str(energy[2]))

	if (tag == "short"):
		energyB1_s = energy[0]
		energyB2_s = energy[1]
		energyB3_s = energy[2]
	else:
		energyB1_l = energy[0]
		energyB2_l = energy[1]
		energyB3_l = energy[2]

	return

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
	save_file.write("timestamp,idx,path,c1,c2,c3,p1,p2,p3\n")

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
	main()

