from pathlib import Path
import easyocr
import torch
import cv2
from utils import *


class OCRModel:
    def __init__(self):
        self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=torch.cuda.is_available())

    def postprocess_text(self, reader_results):
        def combine_lines(boxes_texts):
            mean_w, mean_h = np.mean(
                [[box_width(x[0]) / len(x[1]), box_height(x[0])] for x in boxes_texts],
                axis=0
            ).astype(int)
            max_y = np.max(
                [x[0][3][1] for x in boxes_texts]
            )

            lines = [[] for _ in range(max_y // mean_h + 1)]
            for bt in boxes_texts:
                center = box_center(bt[0])
                lines[int(center[1]) // mean_h].append(bt)

            for i in range(len(lines) - 1, -1, -1):
                if len(lines[i]) > 0:
                    lines = lines[:i + 1]
                    break

            return [sorted(l, key=lambda x: box_center(x[0][0])) for l in lines]

        combined_lines = combine_lines([x[:2] for x in reader_results])
        result_text = '\n'.join([' '.join([entry[1] for entry in line]).strip() for line in combined_lines])

        return result_text.strip().replace(" ;", ';').replace(' .', '.').replace('. ', '.')

    def recognize_text(self, image_path: Path) -> str:
        """
        This method takes an image file as input and returns the recognized text from the image.

        :param image_path: The path to the image file.
        :return: The recognized text from the image.
        """
        im = cv2.imread(str(image_path))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        result = self.reader.readtext(im, slope_ths=0.15, add_margin=0.2)

        return self.postprocess_text(result)
