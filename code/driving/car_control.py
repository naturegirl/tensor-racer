# simple car control script modified from Basic Robot Control

from gopigo import *
import sys

class _Getch:
    """Gets a single character from standard input.  Does not echo to the
screen."""
    def __init__(self):
        self.impl = _GetchUnix()

    def __call__(self): return self.impl()


class _GetchUnix:
    def __init__(self):
        import tty, sys

    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

getch = _Getch()
c = getch()
print(c)
print(c)

print "This is a basic example for the GoPiGo Robot control"
print "Press:\n\tw: Move GoPiGo Robot forward\n\ta: Turn GoPiGo Robot left\n\td: Turn GoPiGo Robot right\n\ts: Move GoPiGo Robot backward\n\tt: Increase speed\n\tg: Decrease speed\n\tx: Stop GoPiGo Robot\n\tz: Exit\n"

while True:
    c = getch()
    print(c)
    if c=='w':
            fwd()	# Move forward
    elif c=='a':
            left()	# Turn left
    elif c=='d':
            right()	# Turn Right
    elif c=='s':
            bwd()	# Move back
    elif c=='x':
            stop()	# Stop
    elif c=='t':
            increase_speed()	# Increase speed
    elif c=='g':
            decrease_speed()	# Decrease speed
    elif c=='z':
            print "Exiting"		# Exit
            sys.exit()
    else:
            print "Wrong Command, Please Enter Again"
    time.sleep(.1)   
