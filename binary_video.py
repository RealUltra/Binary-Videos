import os, sys

try:
    import cv2
except:
    cmd = f'"{sys.executable}" -m pip install opencv-python'
    os.system(cmd)
    import cv2

try:
    import numpy
except:
    cmd = f'"{sys.executable}" -m pip install numpy'
    os.system(cmd)
    import numpy

try:
    from skimage.measure import block_reduce
except:
    cmd = f'"{sys.executable}" -m pip install scikit-image'
    os.system(cmd)
    from skimage.measure import block_reduce

try:
    from scipy.stats import mode
except:
    cmd = f'"{sys.executable}" -m pip install scipy'
    os.system(cmd)
    from scipy.stats import mode

try:
    from bs4 import UnicodeDammit
except:
    cmd = f'"{sys.executable}" -m pip install beautifulsoup4'
    os.system(cmd)
    from bs4 import UnicodeDammit

try:
    from PIL import Image, ImageDraw
except:
    cmd = f'"{sys.executable}" -m pip install pillow'
    os.system(cmd)
    from PIL import Image, ImageDraw

try:
    from qrcode import QRCode
except:
    cmd = f'"{sys.executable}" -m pip install qrcode'
    os.system(cmd)
    from qrcode import QRCode

try:
    from pyzbar.pyzbar import decode as decode_qr_code
except:
    cmd = f'"{sys.executable}" -m pip install pyzbar'
    os.system(cmd)
    from pyzbar.pyzbar import decode as decode_qr_code

try:
    from colorama import init, Fore, Back, Style
except:
    cmd = f'"{sys.executable}" -m pip install colorama'
    os.system(cmd)
    from colorama import init, Fore, Back, Style

from io import BytesIO
import time, datetime
import uuid
import math
import json
import textwrap, binascii

init()

FILENAME_SIZE_BITS = 16

def colored(text, color):
    return f"{color}{text}{Style.RESET_ALL}"

def split_filename(filename):
    split = filename.split('.')
    return ".".join(split[:-1]), split[-1]

def simplify_image(img, width_factor, height_factor):
    new_h = img.shape[0] // height_factor
    new_w = img.shape[1] // width_factor
    blocks = block_reduce(img, block_size=(height_factor, width_factor), func=numpy.mean).astype(img.dtype)
    return blocks[:new_h, :new_w]

def complexify_image(img, width_factor, height_factor):
    return numpy.kron(img, numpy.ones((height_factor, width_factor), dtype=numpy.uint8))

def remove_non_utf8_chars(b):
    dammit = UnicodeDammit(b, ["utf-8", "iso-8859-1"])
    return dammit.unicode_markup.encode("utf-8")

def split_binary(binary, bits=8):
    binary_values = textwrap.wrap("".join(reversed(binary)), bits)
    binary_values = list(reversed(list(map(lambda x: ''.join(reversed(x)), binary_values))))
    binary_values[0] = binary_values[0].rjust(bits, '0')
    return binary_values

def bytes_to_binary(data):
    binary = bin(int(binascii.hexlify(data), 16))[2:].rjust(len(data) * 8, '0')
    return binary

def binary_to_bytes(binary):
    n = int(binary, 2)
    data = n.to_bytes(len(binary) // 8, 'big')
    return data

def to_image_binary(bin_array, width, height):
    max_bits = height * width

    n = math.log((width * height), 2)
    leading_zeros_bits = int(n) if int(n) == n else int(n) + 1

    max_bits -= leading_zeros_bits

    if bin_array.shape[0] <= max_bits:
        remaining = numpy.array([], dtype=numpy.uint8)
    else:
        remaining = bin_array[max_bits:]
        bin_array = bin_array[:max_bits]

    leading_zeros = numpy.argmax(bin_array) if 1 in bin_array else bin_array.shape[0]
    leading_zeros_bin = bin(leading_zeros)[2:].rjust(leading_zeros_bits, '0')
    leading_zeros_arr = numpy.frombuffer(leading_zeros_bin.encode(), dtype=numpy.uint8) - 48
    bin_array = bin_array[leading_zeros:]

    bin_array = numpy.pad(bin_array, (max_bits - bin_array.shape[0], 0), mode='constant', constant_values=0)
    binary = numpy.concatenate([leading_zeros_arr, bin_array])

    return binary, remaining

def fix_image_binary_leading_zeros(binary):
    n = math.log(len(binary), 2)
    leading_zeros_bits = int(n) if int(n) == n else int(n) + 1

    leading_zeros_bin = bytearray(binary[:leading_zeros_bits] + 48)
    leading_zeros = int(leading_zeros_bin, 2)
    binary = binary[leading_zeros_bits:]
    binary = remove_leading_zeros(binary)
    binary = numpy.concatenate([numpy.zeros((leading_zeros,), dtype=numpy.uint8), binary])

    return binary

def make_binary_image(bin_array, img_width, img_height, width_factor, height_factor):
    width = img_width // width_factor
    height = img_height // height_factor

    binary, remaining = to_image_binary(bin_array, width, height)

    image = binary.reshape((height, width))
    image = numpy.where(image == 0, 0, 255).astype(numpy.uint8)

    complex_image = complexify_image(image, width_factor, height_factor)
    img = numpy.pad(complex_image, ((0, img_height - complex_image.shape[0]), (0, img_width - complex_image.shape[1])), mode='constant', constant_values=0)

    return img, remaining

def load_binary_image(file_path):
    return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

def remove_leading_zeros(binary):
    if type(binary) == str:
        return binary.lstrip("0")
    elif type(binary) == numpy.ndarray:
        leading_zeros = numpy.argmax(binary) if 1 in binary else binary.shape[0]
        return binary[leading_zeros:]

def read_binary_image(image, width_factor, height_factor):
    bits = numpy.where(image <= 128, 0, 1).astype(numpy.float32)
    simple_bits = simplify_image(bits, width_factor, height_factor)
    bits = numpy.where(simple_bits <= 0.5, 0, 1).astype(numpy.uint8)
    binary = bits.flatten()
    binary = fix_image_binary_leading_zeros(binary)
    return binary

def create_frame(qr_img, width, height, bg_color):
    img = Image.new("RGB", (width, height))

    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, width, height), fill=bg_color)

    center_width, center_height = qr_img.size

    if width < height:
        center_new_w = width
        center_new_h = int(center_new_w / center_width * center_height)
        qr_img = qr_img.resize((center_new_w, center_new_h))
        center_width, center_height = center_new_w, center_new_h
    else:
        center_new_h = height
        center_new_w = int(center_new_h / center_height * center_width)
        qr_img = qr_img.resize((center_new_w, center_new_h))
        center_width, center_height = center_new_w, center_new_h

    x = int((width - center_width) / 2)
    y = int((height - center_height) / 2)
    img.paste(qr_img, (x, y))

    return img

def pil_to_cv2(pil_img):
    file = BytesIO()
    pil_img.save(file, format="png")
    cv2_img = numpy.frombuffer(file.getvalue(), numpy.uint8)
    cv2_img = cv2.imdecode(cv2_img, cv2.IMREAD_COLOR)
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
    return cv2_img

def encode(video_file, actual_file, width, height, fps, width_factor, height_factor):
    bin_size = os.path.getsize(actual_file)

    filename = os.path.basename(actual_file)
    filename_bin = bytes_to_binary(filename.encode())
    filename_len = len(filename_bin)
    filename_len_bin = bin(filename_len)[2:]
    filename_len_bin = filename_len_bin.rjust(FILENAME_SIZE_BITS, '0')

    bin_data = filename_len_bin + filename_bin
    bin_array = numpy.frombuffer(bin_data.encode(), dtype=numpy.uint8) - 48

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_file, fourcc, fps, (width, height), isColor=False)

    qr = QRCode()
    data = {"width_factor": width_factor, "height_factor": height_factor, "width": width, "height": height}
    qr.add_data(json.dumps(data))
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white")
    first_frame_img = create_frame(qr_img, width, height, (255, 255, 255))
    first_frame = pil_to_cv2(first_frame_img)
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    out.write(first_frame)

    estimated_frames = int(bin_size * 8 * width_factor * height_factor / (width * height)) + 1
    last_percent = 0
    n = 0
    s = time.time()
    estimate_batch_size = 30

    file = open(actual_file, 'rb')

    while bin_array.shape[0]:
        buffer = file.read(int(width * height / width_factor / height_factor / 8) + 1)

        if buffer:
            buffer_arr = numpy.frombuffer(bytes_to_binary(buffer).encode(), dtype=numpy.uint8) - 48
            bin_array = numpy.hstack((bin_array, buffer_arr))

        frame, bin_array = make_binary_image(bin_array, width, height, width_factor, height_factor)
        out.write(frame)
        n += 1

        if n == estimate_batch_size:
            duration = (time.time() - s) / n
            seconds = estimated_frames * duration
            end_time = datetime.datetime.now() + datetime.timedelta(seconds=seconds)

            mins = int(seconds / 60)
            secs = int(seconds - (mins * 60))
            hrs = int(mins / 60)
            mins = int(mins - (hrs * 60))

            print(colored(f"[Bits] {bin_size * 8}", Fore.LIGHTYELLOW_EX))
            print(colored(f"[Estimated Frames] {estimated_frames}", Fore.LIGHTYELLOW_EX))
            print(colored(f"[Estimated Process Duration] {hrs:02d}:{mins:02d}:{secs:02d}", Fore.LIGHTYELLOW_EX))
            print(colored(f"[Estimated End Time] {end_time}", Fore.LIGHTYELLOW_EX))

        elif n > estimate_batch_size:
            percent = int(n / estimated_frames * 100)

            if percent != last_percent:
                print(f"[Encoding] {percent}% Complete!")
                last_percent = percent

    while (n // fps) < 1:
        frame = numpy.zeros((height, width), dtype=numpy.uint8)
        out.write(frame)
        n += 1

    out.release()

    seconds = time.time() - s
    mins = int(seconds / 60)
    secs = int(seconds - (mins * 60))
    hrs = int(mins / 60)
    mins = int(mins - (hrs * 60))
    print(colored(f"[Encoding Time] {hrs:02d}:{mins:02d}:{secs:02d}", Fore.LIGHTYELLOW_EX))

def decode(filename, output_dir=None):
    binary = bytearray()

    if not output_dir:
        output_dir = os.path.dirname(filename)

    cap = cv2.VideoCapture(filename)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, first_frame = cap.read()

    if not ret:
        return "", binary

    decoded = decode_qr_code(first_frame)

    if not decoded:
        return "", binary

    decoded = json.loads(decoded[0].data)
    width_factor = decoded["width_factor"]
    height_factor = decoded["height_factor"]
    height = decoded.get("height", first_frame.shape[0])
    width = decoded.get("width", first_frame.shape[1])

    n = 0
    s = time.time()
    estimate_batch_size = 30
    last_percent = 0

    file = None
    filename_length = 0
    filename = None
    temp_filename = None

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (width, height))

        buffer = read_binary_image(frame, width_factor, height_factor)
        binary.extend(buffer + 48)

        if not file:
            if not filename_length and len(binary) >= FILENAME_SIZE_BITS:
                filename_len_bin = bytearray(binary[:FILENAME_SIZE_BITS])
                filename_length = int(filename_len_bin, 2)
                binary = bytearray(binary[FILENAME_SIZE_BITS:])

            if filename_length and len(binary) >= filename_length:
                filename_bin = bytearray(binary[:filename_length])
                filename_bytes = binary_to_bytes(filename_bin)
                filename = remove_non_utf8_chars(filename_bytes).decode()
                binary = bytearray(binary[filename_length:])

                file_name, file_ext = split_filename(filename)
                temp_filename = f"{file_name} [Recovering].{file_ext}"
                file = open(os.path.join(output_dir, temp_filename), 'wb')

        if file:
            if (len(binary) // 8) > 0:
                bin_len = len(binary) // 8 * 8
                buffer = bytearray(binary[:bin_len])
                buffer_bytes = binary_to_bytes(buffer)
                file.write(buffer_bytes)
                binary = bytearray(binary[bin_len:])

        n += 1

        if n == estimate_batch_size:
            duration = (time.time() - s) / n
            seconds = num_frames * duration
            end_time = datetime.datetime.now() + datetime.timedelta(seconds=seconds)

            mins = int(seconds / 60)
            secs = int(seconds - (mins * 60))
            hrs = int(mins / 60)
            mins = int(mins - (hrs * 60))

            print(colored(f"[Frames] {num_frames}", Fore.LIGHTYELLOW_EX))
            print(colored(f"[Estimated Process Duration] {hrs:02d}:{mins:02d}:{secs:02d}", Fore.LIGHTYELLOW_EX))
            print(colored(f"[Estimated End Time] {end_time}", Fore.LIGHTYELLOW_EX))

        elif n > estimate_batch_size:
            percent = int(n / num_frames * 100)

            if percent != last_percent:
                print(f"[Decoding] {percent}% Complete!")
                last_percent = percent

    file.close()

    seconds = time.time() - s
    mins = int(seconds / 60)
    secs = int(seconds - (mins * 60))
    hrs = int(mins / 60)
    mins = int(mins - (hrs * 60))
    print(colored(f"[Decoding Time] {hrs:02d}:{mins:02d}:{secs:02d}", Fore.LIGHTYELLOW_EX))

    file_name, file_ext = split_filename(filename)
    filename = f"{file_name} [Recovered].{file_ext}"
    os.rename(os.path.join(output_dir, temp_filename), os.path.join(output_dir, filename))

    return filename