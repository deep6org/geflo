#select pod
import asyncio
from bleak import discover

#run this to find unique mac address for pod
#e.g. 56FFAB79-ACF1-4E4B-85B7-ED0C0A199973
#pod name structure: blueberry-XX || blueberry-XXXX

async def run():
    devices = await discover()
    for d in devices:
        print(d)

loop = asyncio.get_event_loop()
loop.run_until_complete(run())