from arm import Arm
from serial.tools import list_ports
from pydobot import Dobot
print('running')
p = list_ports.comports()[0].device
print(p)
d = Dobot(port=p, verbose=True)
d.move_to(140, 4, 6, 0.0, wait = True)


d.close()
print('success')