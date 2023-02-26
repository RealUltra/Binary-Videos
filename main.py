from binary_video import *
import tkinter
from tkinter import filedialog
import os
import sys

try:
    from colorama import init, Fore, Back, Style
except:
    cmd = f'"{sys.executable}" -m pip install colorama'
    os.system(cmd)
    from colorama import init, Fore, Back, Style

def colored(text, color):
    return f"{color}{text}{Style.RESET_ALL}"

def split_filename(filename):
    split = filename.split('.')
    return ".".join(split[:-1]), split[-1]

init()

print(colored("###### BINARY VIDEO CREATOR #######\n", Fore.GREEN))

mode = 0
while True:
    print(colored("1) Create Binary Video", Fore.LIGHTMAGENTA_EX))
    print(colored("2) Read Binary Video", Fore.LIGHTMAGENTA_EX))
    inp = input(colored("Enter 1 or 2: ", Fore.MAGENTA)).strip()

    if inp == "1":
        mode = 0
        break
    elif inp == "2":
        mode = 1
        break
    else:
        print(colored("[ERROR] Invalid Input!\n", Fore.RED))
print()

root = tkinter.Tk()
root.withdraw()
root.attributes('-topmost', True)

default_width = 400
default_height = 400
default_framerate = 10

while True:
    print(colored("1) Optimal Compression Resistance", Fore.LIGHTMAGENTA_EX))
    print(colored("2) Paranoid Compression Resistance (totally won't destroy ur pc)", Fore.LIGHTMAGENTA_EX))
    print(colored("3) No Compression Resistance (Even the slightest compression will make you lose your life's work)",
                  Fore.LIGHTMAGENTA_EX))
    inp = input(colored("Enter 1, 2 or 3: ", Fore.MAGENTA)).strip()

    if inp == "1":
        width_factor, height_factor = 4, 4
        break
    elif inp == "2":
        width_factor, height_factor = 6, 6
        break
    elif inp == "3":
        width_factor, height_factor = 1, 1
        break
    else:
        print(colored("[ERROR] Invalid Input!\n", Fore.RED))
print()

if mode == 0:
    width, height = 1920, 1080
    fps = 30

    real_file = filedialog.askopenfilename(initialdir=".", title="Select Any File",
                                          filetypes=(("All Files", "*.*"),))

    if not os.path.exists(real_file):
        sys.exit(0)

    binary_video_file = filedialog.asksaveasfilename(initialdir=".", title="Save Binary Video As",
                                          filetypes=(("Binary Video", "*.mp4"),), defaultextension='.mp4')

    if binary_video_file:
        data = open(real_file, 'rb').read()
        bin_data = bytes_to_binary(data)
        print(colored(f"Bits: {len(bin_data)}", Fore.LIGHTYELLOW_EX))
        estimated_frames = int(len(bin_data) * width_factor * height_factor / (width * height)) + 1
        print(colored(f"Estimated Frames: {estimated_frames}", Fore.LIGHTYELLOW_EX))
        print(colored(f'[UPDATE] Writing "{binary_video_file}"', Fore.LIGHTBLUE_EX))
        encode(binary_video_file, os.path.basename(real_file), bin_data, width, height, fps, width_factor, height_factor)
        print(colored("[SUCCESS] Binary Video Created!", Fore.LIGHTGREEN_EX))

else:
    binary_video_file = filedialog.askopenfilename(initialdir=".", title="Select Binary Video",
                                          filetypes=(("Video files", "*.avi;*.mp4;*.mkv;*.mov;*.wmv;*.flv;*.m4v"),), defaultextension='.mp4')

    if os.path.exists(binary_video_file):
        print(colored(f'[UPDATE] Reading "{binary_video_file}"', Fore.LIGHTBLUE_EX))
        filename, data = decode(binary_video_file, width_factor, height_factor)

        file_name, ext = split_filename(filename)
        file_name = file_name + " [Recovered]"
        filename = file_name + "." + ext
        path = os.path.join(os.path.dirname(binary_video_file), filename)

        with open(path, 'wb') as f:
            f.write(data)

        print(colored(f'[SUCCESS] "{filename}" Saved!', Fore.LIGHTGREEN_EX))