"""Parsing code for DICOMS and contour files"""

from abc import ABC, abstractmethod
import os
import csv
import random

import dicom
from dicom.errors import InvalidDicomError

import numpy as np
from PIL import Image, ImageDraw
from matplotlib.pyplot import imshow

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
    """ Data Parser for i-contour files
    """
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


class DicomParser(DataParser):
    """ Data Parser for dicom image files
    """
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
    """ MRI Data Loader:
            1) Loads contour and dicom files
            2) Pairs contour and dicom files
            3) Computes binary mask for contours
            4) Returns batches of contour_mask, dicom image
    """
    def __init__(self, contour_dir, dicom_dir, map_filename, minibatch_size):
        """ Initialize the MRI Data Loader

        :param contour_dir: path to directory with parseable contour_dir
        :param dicom_dir: path to directory with parseable dicom_dir
        :param map_filename: path to csv file matching contour
                             subdirectories with dicom subdirectories
        :param minibatch_size: minibatch size for data fetching
        """
        self.contour_dir = contour_dir
        self.dicom_dir = dicom_dir
        self.map_filename = map_filename
        self.minibatch_size = minibatch_size
        self.contour_masks = None
        self.dicoms = None
        self.contour_mask_files = None
        self.dicom_files = None
        self.num_data_fetched = 0
        self.data_size = 0
        super().__init__()

    def load(self):
        """ Load training data (contour masks, dicoms) pairs
        """
        self.contour_masks, self.dicoms, self.contour_mask_files, self.dicom_files = self._match_contour_to_dicom()
        self._shuffle_data()

    def fetch_minibatch(self):
        """ Fetch minibatch of contour mask, dicom image data.
            Upon each epoch, all training data is randomly shuffled.
            Throws error if the data hasn't already been loaded.

        :return: numpy array of contour masks
        :return: numpy arary of dicom images
        """
        if self.contour_masks is None or self.dicoms is None:
            raise Exception('Please run .load(), to load training data from disk \
                before attempting to fetch a minibatch')

        contour_masks = None
        dicom_images = None
        contour_mask_files = None
        dicom_files = None

        if (self.num_data_fetched + self.minibatch_size) >= self.data_size:
            # Last minibatch, before starting the next epoch.
            contour_masks = self.contour_masks[self.num_data_fetched : ]
            dicom_images = self.dicoms[self.num_data_fetched : ]
            contour_mask_files = self.contour_mask_files[self.num_data_fetched : ]
            dicom_files = self.dicom_files[self.num_data_fetched : ]

            # Reset num_data_fetched and reshuffle data
            self.num_data_fetched = 0
            self._shuffle_data()
        else:
            contour_masks = self.contour_masks[self.num_data_fetched : self.num_data_fetched+self.minibatch_size]
            dicom_images = self.dicoms[self.num_data_fetched:self.num_data_fetched+self.minibatch_size]
            contour_mask_files = self.contour_mask_files[self.num_data_fetched : self.num_data_fetched+self.minibatch_size]
            dicom_files = self.dicom_files[self.num_data_fetched : self.num_data_fetched+self.minibatch_size]
            self.num_data_fetched += self.minibatch_size

        return np.asarray(contour_masks), np.asarray(dicom_images), contour_mask_files, dicom_files      

    def _parse_map_file(self):
        """ Parse map file, linking contour subdirectories with dicom subdirectories

        :return: dictionary of contour_name -> dicom_name
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
        selected_dicoms = []   
        selected_dicom_files = []
        selected_contour_masks = []
        selected_contour_files = []

        dir_pairs = self._parse_map_file()
        for contour_name, dicom_name in dir_pairs:
            # parse i-contour files in desired directory
            icontour_path = os.path.join(self.contour_dir, contour_name, 'i-contours')
            icountour_parser = IContourParser(icontour_path)
            contours = icountour_parser.parse()

            # parse dicom files in desired directory
            dicom_path = os.path.join(self.dicom_dir, dicom_name)
            dicom_parser = DicomParser(dicom_path)
            dicoms = dicom_parser.parse()

            # Extract dicom id for each of the contour files
            def extract_dicom_id(filename):
                """ Extract dicom id from the contour file name.
                    Note: Here, we make the assumption that the file `IM-0001-0060-icontour-manual.txt`
                          matches with the 60.dcm file, in the appropriate dicom subdirectory.
                
                    :return: string version of the dicom id for the given contour filename
                """
                end_ind = filename.find('-icontour')
                num = filename[end_ind-4:end_ind]
                return num.lstrip('0')
            extracted_dicom_ids = {
                extract_dicom_id(contour_filename) : contour_filename for contour_filename in contours.keys()
            }

            # Determine overlap between dicom images and the countour files
            shared_ids = set(extracted_dicom_ids.keys()).intersection(set(dicoms.keys()))
            selected_dicoms.extend([dicoms[x] for x in shared_ids])
            selected_dicom_files.extend([os.path.join(dicom_path, x) for x in shared_ids])

            # Note: Height, Width for contour map is surmised from the dimensions of the dicom image.
            for id in shared_ids:
                contour = contours[extracted_dicom_ids[id]]
                img_sz = dicoms[id]['pixel_data'].shape
                contour_mask = self._poly_to_mask(contour, img_sz[1], img_sz[0])
                selected_contour_masks.append(contour_mask)
                selected_contour_files.append(os.path.join(icontour_path, extracted_dicom_ids[id]))

        self.data_size = len(selected_contour_masks)
        return selected_contour_masks, selected_dicoms, selected_contour_files, selected_dicom_files

    def _poly_to_mask(self, polygon, width, height):
        """Convert polygon to mask

        :param polygon: list of pairs of x, y coords [(x1, y1), (x2, y2), ...]
        in units of pixels
        :param width: scalar image width
        :param height: scalar image height
        :return: Boolean mask of shape (height, width)
        """
        # http://stackoverflow.com/a/3732128/1410871
        img = Image.new(mode='L', size=(width, height), color=0)
        draw = ImageDraw.Draw(img, mode='L')
        draw.polygon(xy=polygon, outline="green", fill="red")
        mask = np.array(img).astype(bool)
        return mask

    def _shuffle_data(self):
        """ Shuffle training data
            Note: This is an internal function and should only be called,
                  once it's been guaranteed that the data has been loaded.
        """
        data = list(zip(self.contour_masks, self.dicoms, self.contour_mask_files, self.dicom_files))
        random.shuffle(data)
        self.contour_masks[:], self.dicoms[:], self.contour_mask_files[:], self.dicom_files[:] = zip(*data)


if __name__ == '__main__':
    dataLoader = MRIDataLoader(
        './final_data/contourfiles',
        './final_data/dicoms',
        './final_data/link.csv',
        8
    )
    dataLoader.load()
    contour_masks, dicoms = dataLoader.fetch_minibatch()