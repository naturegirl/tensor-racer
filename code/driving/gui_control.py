#!/usr/bin/env python
#############################################################################################################                                                                  
# Basic example for controlling the GoPiGo using the Keyboard
# Contributed by casten on Gitub https://github.com/DexterInd/GoPiGo/pull/112
#
# This code lets you control the GoPiGo from the VNC or Pi Desktop. Also, these are non-blocking calls so it is much more easier to use too.
#
# Controls:
#   w:  Move forward
#   a:  Turn left
#   d:  Turn right
#   s:  Move back
#   x:  Stop
#   t:  Increase speed
#   g:  Decrease speed
#   z:  Exit
# http://www.dexterindustries.com/GoPiGo/                                                                
# History
# ------------------------------------------------
# Author        Date            Comments  
# Karan     27 June 14          Code cleanup                                                    
# Casten    31 Dec  15          Added async io, action until keyup
# Karan     04 Jan  16          Cleaned up the GUI

'''
## License
 GoPiGo for the Raspberry Pi: an open source robotics platform for the Raspberry Pi.
 Copyright (C) 2015  Dexter Industries

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/gpl-3.0.txt>.
'''     
##############################################################################################################

from gopigo import *    #Has the basic functions for controlling the GoPiGo Robot
import sys  #Used for closing the running program
import os
import pygame #Gives access to KEYUP/KEYDOWN events
import picamera
import shutil

#Initialization for pygame
pygame.init()
screen = pygame.display.set_mode((700, 400))
pygame.display.set_caption('Remote Control Window')


# Camera
cam = picamera.PiCamera()
cam.resolution = (100, 100)
DATA_DIR = "./data/"
# adjust image_idx
image_idx = 304
# Labels
LEFT = 0
RIGHT = 1
STRAIGHT = 2

# Fill background
background = pygame.Surface(screen.get_size())
background = background.convert()
background.fill((251, 250, 250))

# Display some text
instructions = '''
                      BASIC GOPIGO CONTROL GUI

This is a basic example for the GoPiGo Robot control 

(Be sure to put focus on thi window to control the gopigo!)

Press:
      ->w: Move GoPiGo Robot forward
      ->a: Turn GoPiGo Robot left
      ->d: Turn GoPiGo Robot right
      ->x: Stop GoPiGo
      ->z: Exit
''';
size_inc=22
index=0
for i in instructions.split('\n'):
    font = pygame.font.Font(None, 36)
    text = font.render(i, 1, (10, 10, 10))
    background.blit(text, (10,10+size_inc*index))
    index+=1

# Blit everything to the screen
screen.blit(background, (0, 0))
pygame.display.flip()

REPEAT_DELAY = 100      # delay in ms for repeat key
DEFAULT_SPEED = 50
DELTA_INC = 20               # change in speed when turning
DELTA_DEC = 5
speed_l = DEFAULT_SPEED
speed_r = DEFAULT_SPEED

def create_data_dir():
    """creates the folders to save the training data"""
    if os.path.exists(DATA_DIR):
        print("delete existing dir", DATA_DIR)
        shutil.rmtree(DATA_DIR)
    os.mkdir(DATA_DIR)
    os.mkdir(os.path.join(DATA_DIR, "left"))
    os.mkdir(os.path.join(DATA_DIR, "right"))
    os.mkdir(os.path.join(DATA_DIR, "straight"))

def save_img(label, image_idx):
    """capture and save image"""
    if label == LEFT:
        img_path = os.path.join(DATA_DIR, "left", str(image_idx) + ".jpg")
    elif label == RIGHT:
        img_path = os.path.join(DATA_DIR, "right", str(image_idx) + ".jpg")
    elif label == STRAIGHT:
        img_path = os.path.join(DATA_DIR, "straight", str(image_idx) + ".jpg")
    cam.capture(img_path)
    print("saved image", img_path)

# a hold down key will be seen as repeated KEYDOWN events
pygame.key.set_repeat(100, 100)

# create_data_dir()

while True:
    event = pygame.event.wait();

    if event.type == pygame.KEYUP:
        # released turning keys, go straight
        print("set back to default speed")
        set_speed(DEFAULT_SPEED)
        speed_l = speed_r = DEFAULT_SPEED
    elif event.type == pygame.KEYDOWN:
        char = event.unicode;
        print("keydown event char:", char)
        if char == 'w':   # Forward
            save_img(STRAIGHT, image_idx)
            image_idx += 1
            fwd()
            set_speed(DEFAULT_SPEED)
            speed_l = speed_r = DEFAULT_SPEED
        elif char == 'a': # Left
            save_img(LEFT, image_idx)
            image_idx += 1
            speed_l = 0
            set_left_speed(speed_l)
        elif char == 'd': # Right
            save_img(RIGHT, image_idx)
            image_idx += 1
            speed_r = 0
            set_right_speed(speed_r)
        elif char == 'x':
            stop()
        elif char == 'z':
            print "\nExiting";      # Exit
            stop()
            sys.exit();


