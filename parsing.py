"""Parsing code for DICOMS and contour files"""

from abc import ABC, abstractmethod
import os
import csv

import dicom
from dicom.errors import InvalidDicomError

import numpy as np
from PIL import Image, ImageDraw

class DataParser(ABC):
    """ Data Parser Abstract Class
    """
    def __init__(self, data_dir):
        """ Initialize parser

        :param data_dir: path to directory with parseable files
        """
        self.dir = os.fsencode(data_dir)
        super().__init__()

    @abstractmethod
    def parse(self):
        """ Parse files from the initialized directory

        :return: map of filenames to parsed data files
        """
        pass

class IContourParser(DataParser):
    def parse(self):
        """ Parse contour files from the initialized directory

        :return: map of the filenames to parsed contour files
        """
        fn_to_data = {}
        for file in os.listdir(self.dir):
            data_dir = os.fsdecode(self.dir)
            filename = os.fsdecode(file)
            if filename.endswith(".txt"):
                file_path = os.path.join(data_dir, filename)
                filename = os.path.splitext(filename)[0]
                fn_to_data[filename] = self._parse_contour_file(file_path)
            else:
                continue
        return fn_to_data

    def _parse_contour_file(self, filename):
        """Parse the given contour filename

        :param filename: filepath to the contourfile to parse
        :return: list of tuples holding x, y coordinates of the contour
        """

        coords_lst = []

        with open(filename, 'r') as infile:
            for line in infile:
                coords = line.strip().split()

                x_coord = float(coords[0])
                y_coord = float(coords[1])
                coords_lst.append((x_coord, y_coord))

        return coords_lst

    def poly_to_mask(self, polygon, width, height):
        """Convert polygon to mask

        :param polygon: list of pairs of x, y coords [(x1, y1), (x2, y2), ...]
        in units of pixels
        :param width: scalar image width
        :param height: scalar image height
        :return: Boolean mask of shape (height, width)
        """

        # http://stackoverflow.com/a/3732128/1410871
        img = Image.new(mode='L', size=(width, height), color=0)
        ImageDraw.Draw(img).polygon(xy=polygon, outline=0, fill=1)
        mask = np.array(img).astype(bool)
        return mask

class DicomParser(DataParser):
    def parse(self):
        """ Parse dicom files from the initialized directory

        :return: map of the filenames to parsed dicom files
        """
        fn_to_data = {}
        for file in os.listdir(self.dir):
            data_dir = os.fsdecode(self.dir)
            filename = os.fsdecode(file)
            if filename.endswith(".dcm"):
                file_path = os.path.join(data_dir, filename)
                filename = os.path.splitext(filename)[0]
                fn_to_data[filename] = self._parse_dicom_file(file_path)
            else:
                continue
        return fn_to_data

    def _parse_dicom_file(self, filename):
        """Parse the given DICOM filename

        :param filename: filepath to the DICOM file to parse
        :return: dictionary with DICOM image data
        """

        try:
            dcm = dicom.read_file(filename)
            dcm_image = dcm.pixel_array

            try:
                intercept = dcm.RescaleIntercept
            except AttributeError:
                intercept = 0.0
            try:
                slope = dcm.RescaleSlope
            except AttributeError:
                slope = 0.0

            if intercept != 0.0 and slope != 0.0:
                dcm_image = dcm_image*slope + intercept
            dcm_dict = {'pixel_data' : dcm_image}
            return dcm_dict
        except InvalidDicomError:
            return None


class MRIDataLoader(object):
    def __init__(self, contour_dir, dicom_dir, map_filename):
        """ Initialize the MRI Data Loader

        :param contour_dir: path to directory with parseable contour_dir
        :param dicom_dir: path to directory with parseable dicom_dir
        :param map_filenam: path to csv file matching contour subdirectories with dicom subdirectories
        :return: dictionary with DICOM image data
        """
        self.contour_dir = contour_dir
        self.dicom_dir = dicom_dir
        self.map_filename = map_filename
        super().__init__()

    def load(self):
        """ Load training data (contours, dicoms) pairs
        """
        contours, dicoms = self._match_contour_to_dicom()
        print(contours)
        print(dicoms)

    def _parse_map_file(self):
        """ Parse map file, linking contour subdirectories with dicom subdirectories

        :return: dictionary of contour_id -> dicom_id
        """
        dir_ids = []
        with open(self.map_filename) as map_file:
            reader = csv.DictReader(map_file)
            for row in reader:
                dir_ids.append((row['original_id'], row['patient_id']))
        return dir_ids

    def _match_contour_to_dicom(self):
        """ Parse contours and dicoms. Then pair appropriate contours with dicoms.

        :return: list of parsed contour masks
        :return: list of parsed dicom image files
        """
        selected_contours = []
        selected_dicoms = []
        dir_ids = self._parse_map_file()
        for contour_id, dicom_id in dir_ids:
            # parse i-contour files in desired directory
            icontour_path = os.path.join(self.contour_dir, contour_id, 'i-contours')
            icountour_parser = IContourParser(icontour_path)
            contours = icountour_parser.parse()

            # parse dicom files in desired directory
            dicom_path = os.path.join(self.dicom_dir, dicom_id)
            dicom_parser = DicomParser(dicom_path)
            dicoms = dicom_parser.parse()

            # Match i-contour files to dicom files
            def extract_dicom_id(filename):
                end_ind = filename.find('-icontour')
                num = filename[end_ind-4:end_ind]
                return num.lstrip('0')
            extracted_dicom_ids = {
                extract_dicom_id(contour_filename) : contour_filename for contour_filename in contours.keys()
            }
            shared_ids = set(extracted_dicom_ids.keys()).intersection(set(dicoms.keys()))
            selected_contours.extend([contours[extracted_dicom_ids[x]] for x in shared_ids])
            selected_dicoms.extend([dicoms[x] for x in shared_ids])
        return selected_contours, selected_dicoms

if __name__ == '__main__':
    dataLoader = MRIDataLoader(
        './final_data/contourfiles',
        './final_data/dicoms',
        './final_data/link.csv'
    )
    dataLoader.load()