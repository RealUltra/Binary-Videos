# Binary-Videos

This project allows you to convert any file into a video that I call a binary video. This video can then be uploaded to YouTube as a way of infinite free cloud. Other cloud storage services such as Dropbox and Google Drive charge you money to store large files. However, YouTube allows an infinite number of videos to be uploaded to a channel for free and each video can be upto 256 GB. This makes YouTube the cheapest way to store your data. 

My project was heavily inspired by [this project](https://github.com/DvorakDwarf/Infinite-Storage-Glitch)

## Table of Contents

- [Example](#example)
- [How To Set It Up](#how-to-set-it-up)
- [How To Use It](#how-to-use-it)
- [How It Works](#how-it-works)
- [Problems and Solutions](#problems-and-solutions)

## Example
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/pT8qMJZUI0Y/1.jpg)](https://www.youtube.com/watch?v=pT8qMJZUI0Y)

## How To Set It Up

To use this program, you'll need a python interpreter installed. When you run main.py, it should automatically install all the necessary packages. In case that doesn't work, use the following command: 
pip install opencv-python numpy scikit-image scipy beautifulsoup4 pillow qrcode pyzbar colorama

After that when you run main.py again it should work. I haven't tested this app on any other operating system besides Windows as that's what I'm familiar with so its possible that it doesn't work on Mac or Linux.

## How To Use It

After running the main.py file, it will let you choose whether you want to create a binary video or read one. 
If you choose to create one, you'll have to select what level of compression resistance you want to choose. 
For all the cases I've tested, Optimal works perfectly. *Why this is important is explained later* 
How long the creation or reading of the binary videos takes depends on the size of the file.

## How It Works

We know that at its base, every file is made of binary which is just 0s and 1s. 
We also know that images are created from pixels where each each pixel has a color. 
My program creates binary videos by creating images where each pixel is either black for 0 or white for 1. 
The binary video is just a series of these images as frames. 

## Problems and Solutions

The main issue with the concept is that youtube's compression algorithm is absolutely brutal. In a perfect world, we wouldn't just be making binary videos using white and black pixels, we'd be using different unique colors as each color is made using 3 bytes which would mean each pixel would store 3 whole bytes. Instead each pixel is just storing a single bit. When youtube compresses a video, the colors of the pixels are slightly shifted. This shift is enough to corrupt entire files that have been turned into binary videos.

The way I'm combatting this is that instead of using just one pixel for one bit, I'm using a block of 4x4 pixels (this is the optimal compression resistance, paranoid is 6x6). This way, even if some of the pixels in the block are changed, the majority are usually retained so the program is able to still fully understand the file. For now, all the tests I've done have shown no hint of corruption with the optimal compression resistance being used.
