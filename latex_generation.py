

import numpy as np
import pdf2image
import os
import cv2
import pytesseract
from tqdm import tqdm
from utils import escape_special_chars



def determine_content(block_img):
    # TODO: This is a very naive way of determining content. We should use a better method.
    data_dict = pytesseract.image_to_data(block_img, output_type=pytesseract.Output.DICT)
    confidence_leval = np.mean([abs(int(c)) for c in data_dict['conf']])

    # If the confidence level is high enough, we can trust the text
    if confidence_leval > 40:
        return ("text", ' '.join(data_dict['text']))

    # If the confidence level is too low, we can't trust the text. So we will treat it as an image
    else:
        return ("image", block_img)


def generate_latex_block(block_type, content):
    if block_type == "text":
        #content = escape_special_chars(content)
        return content

    elif block_type == "image":
        # TODO - Check if its math or pure image
        # http://www.inftyproject.org/en/software.html
        # https://github.com/UW-COSMOS/latex-ocr
        # https://github.com/blaisewang/img2latex-mathpix

        # TODO
        pass

def generate_latex(block_img):
    block_type, content = determine_content(block_img)
    content = generate_latex_block(block_type, content)
    return block_type, content
