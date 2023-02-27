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
    from pyzbar.pyzbar import decode  as decode_qr_code
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
import math
import json
import textwrap, binascii

init()

FILENAME_SIZE_BITS = 16

def colored(text, color):
    return f"{color}{text}{Style.RESET_ALL}"

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
    return bin(int(binascii.hexlify(data), 16))[2:]

def binary_to_bytes(binary):
    n = int(binary, 2)
    data = n.to_bytes((n.bit_length() + 7) // 8, 'big')
    return data

def to_image_binary(bin_data, width, height):
    binary = ""
    max_bits = height * width

    n = math.log((width * height), 2)
    leading_zeros_bits = int(n) if int(n) == n else int(n) + 1

    max_bits -= leading_zeros_bits

    if len(bin_data) <= max_bits:
        remaining = []
    else:
        remaining = bin_data[:-max_bits]
        bin_data = bin_data[-max_bits::]

    leading_zeros = bin_data.index('1') if '1' in bin_data else len(bin_data)
    leading_zeros_bin = bin(leading_zeros)[2:].rjust(leading_zeros_bits, '0')
    binary += leading_zeros_bin
    bin_data = remove_leading_zeros(bin_data)

    bin_data = bin_data.rjust(max_bits, '0')
    binary += bin_data

    return binary, remaining

def fix_image_binary_leading_zeros(binary):
    n = math.log(len(binary), 2)
    leading_zeros_bits = int(n) if int(n) == n else int(n) + 1

    leading_zeros_bin = binary[:leading_zeros_bits]
    leading_zeros = int(leading_zeros_bin, 2)
    binary = binary[leading_zeros_bits:]
    binary = remove_leading_zeros(binary)
    binary = ("0" * leading_zeros) + binary
    return binary

def make_binary_image(bin_data, img_width, img_height, width_factor, height_factor):
    width = img_width // width_factor
    height = img_height // height_factor

    binary, remaining = to_image_binary(bin_data, width, height)

    image = numpy.array(list(binary), dtype=numpy.uint8)
    image = image.reshape((height, width))
    image = numpy.where(image == 0, 0, 255).astype(numpy.uint8)

    complex_image = complexify_image(image, width_factor, height_factor)
    img = numpy.pad(complex_image, ((0, img_height - complex_image.shape[0]), (0, img_width - complex_image.shape[1])), mode='constant', constant_values=0)

    return img, remaining

def load_binary_image(file_path):
    return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

def remove_leading_zeros(binary):
    return binary.lstrip("0")

def read_binary_image(image, width_factor, height_factor):
    bits = numpy.where(image <= 128, 0, 1).astype(numpy.float32)
    simple_bits = simplify_image(bits, width_factor, height_factor)
    bits = numpy.where(simple_bits <= 0.5, 0, 1).astype(numpy.uint8)
    binary = ''.join(bits.flatten().astype(str))
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

def encode(video_filename, filename, bin_data, width, height, fps, width_factor, height_factor):
    filename_bin = bytes_to_binary(filename.encode())
    filename_len = len(filename_bin)
    filename_len_bin = bin(filename_len)[2:]
    filename_len_bin = filename_len_bin.rjust(FILENAME_SIZE_BITS, '0')

    bin_data = filename_len_bin + filename_bin + bin_data

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height), isColor=False)

    qr = QRCode()
    data = {"width_factor": width_factor, "height_factor": height_factor}
    qr.add_data(json.dumps(data))
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white")
    first_frame_img = create_frame(qr_img, width, height, (255, 255, 255))
    first_frame = pil_to_cv2(first_frame_img)
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    out.write(first_frame)

    n = 0
    s = time.time()
    estimate_batch_size = 10

    while bin_data:
        frame, bin_data = make_binary_image(bin_data, width, height, width_factor, height_factor)
        out.write(frame)
        n += 1

        if n == estimate_batch_size:
            duration = (time.time() - s) / n
            seconds = len(bin_data) * width_factor * height_factor / (width * height) * duration
            end_time = datetime.datetime.now() + datetime.timedelta(seconds=seconds)

            mins = int(seconds / 60)
            secs = int(seconds - (mins * 60))
            hrs = int(mins / 60)
            mins = int(mins - (hrs * 60))

            print(f"Estimated Process Duration: {hrs:02d}:{mins:02d}:{secs:02d}")
            print("Estimated End Time:", end_time)

        elif n > estimate_batch_size:
            print("Frames Encoded:", n)

    while (n // fps) < 1:
        frame = numpy.zeros((height, width), dtype=numpy.uint8)
        out.write(frame)
        n += 1

    out.release()

def decode(filename):
    binary = ""

    cap = cv2.VideoCapture(filename)
    ret, first_frame = cap.read()

    if not ret:
        return "", binary

    decoded = decode_qr_code(first_frame)

    if not decoded:
        return "", binary

    decoded = json.loads(decoded[0].data)
    width_factor = decoded["width_factor"]
    height_factor = decoded["height_factor"]

    n = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        buffer = read_binary_image(frame, width_factor, height_factor)

        binary = buffer + binary
        #binary += buffer
        n += 1
        print("Frames Decoded:", n)

    filename_len_bin = binary[:FILENAME_SIZE_BITS]
    filename_len = int(filename_len_bin, 2)
    filename_bin = binary[FILENAME_SIZE_BITS : FILENAME_SIZE_BITS+filename_len]
    filename_bytes = binary_to_bytes(filename_bin)
    filename = remove_non_utf8_chars(filename_bytes).decode()
    binary = binary[FILENAME_SIZE_BITS+filename_len:]

    return filename, binary_to_bytes(binary)
