from pathlib import Path
import easyocr
import cv2
from utils import *


class OCRModel:
    def __init__(self):
        self.reader = easyocr.Reader(['ch_sim', 'en'])

    def postprocess_text(self, reader_results):
        """
        Sorts found text by ascending (y, x) and adds newlines and tabs to it
        :param reader_results: results from easyOCR
        :return: detected text
        """

        def combine_lines(boxes_texts):
            """
            Combines entries of boxes_texts into horizontal lines and sorts them in ascending order by y, x
            :param boxes_texts: list of [box, text]
            :return:
            """
            mean_w, mean_h = np.mean(
                [[box_width(x[0]) / len(x[1]), box_height(x[0])] for x in boxes_texts],
                axis=0
            ).astype(int)

            max_y = np.max([x[0][3][1] for x in boxes_texts])

            min_x = np.min([x[0][0][0] for x in boxes_texts])

            lines = [[] for _ in range(max_y // mean_h + 1)]
            for bt in boxes_texts:
                center = box_center(bt[0])
                lines[int(center[1]) // mean_h].append(bt)

            for i in range(len(lines) - 1, -1, -1):
                if len(lines[i]) > 0:
                    lines = lines[:i + 1]
                    break

            lines = [sorted(l, key=lambda x: box_center(x[0][0])) for l in lines]
            for i in range(len(lines)):
                if len(lines[i]) > 0:
                    line_x = lines[i][0][0][0][0]
                    lines[i][0] = (lines[i][0][0], " " * ((line_x - min_x) // mean_w) + lines[i][0][1])
            return lines

        combined_lines = combine_lines([x[:2] for x in reader_results])
        result_text = '\n'.join([' '.join([entry[1] for entry in line]) for line in combined_lines])

        return result_text

    def recognize_text(self, image_path: Path) -> str:
        """
        This method takes an image file path as input and returns the recognized text from the image.

        :param image_path: The path to the image file.
        :return: The recognized text from the image.
        """
        im = cv2.imread(str(image_path))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        result = self.reader.readtext(im, slope_ths=0.15, add_margin=0.2, width_ths=1, height_ths=1)

        return self.postprocess_text(result)
