from pathlib import Path
import easyocr
import torch


class OCRModel:
    def __init__(self):
        self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=torch.cuda.is_available())

    def recognize_text(self, image_path: Path) -> str:
        """
        This method takes an image file as input and returns the recognized text from the image.

        :param image_path: The path to the image file.
        :return: The recognized text from the image.
        """
        result = self.reader.readtext(str(image_path), detail=0)
        return ''.join(result)
