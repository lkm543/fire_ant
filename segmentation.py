import glob
import os

import cv2
import numpy as np
import scipy.ndimage

from tqdm import tqdm


class image_process():
    def process_all(self, input_path, output_path):
        saved_filenames = []
        filenames = glob.glob(input_path + '/**', recursive=True)
        for filename in tqdm(filenames):
            if os.path.isfile(filename) and "HEIC" not in filename:
                image = self.load_image(filename)
                if image is not None:
                    filename_without_path = filename.split('\\')[-1]
                    # Check if the filename is used or not?
                    if filename_without_path not in saved_filenames:
                        saved_filenames.append(filename_without_path)
                    else:
                        print('Error!! Filename reused!!')
                    output_filename = output_path + filename_without_path
                    image = image_process.segmentation(image)
                    image = image_process.crop_image(image)
                    cv2.imwrite(output_filename, image)
                else:
                    print(f'Error in file: {filename}')

    def load_image(self, filename):
        img = cv2.imread(filename)
        return img

    @staticmethod
    def segmentation(image_ndarray):
        gray = cv2.cvtColor(image_ndarray, cv2.COLOR_BGR2GRAY)
        cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX)
        ret, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        binary = cv2.erode(otsu_mask, None, iterations=30)
        binary = cv2.dilate(binary, None, iterations=30)
        (cnts,_) = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        area = np.array([cv2.contourArea(cnts[i]) for i in range(len(cnts))])
        maxa_ind = np.argmax(area)
        hull = cv2.convexHull(cnts[maxa_ind], False)
        result = image_ndarray.copy()    
        length = len(hull)
        # for i in range(len(hull)):
        #     cv2.line(result, tuple(hull[i][0]), tuple(hull[(i+1)%length][0]), (0,255,0), 10)
        result = image_process.extract_image(result, hull)
        return result

    @staticmethod
    def segmentation_level_Set(image_ndarray):
        dt = 1
        it = 10
        sigma = 20
        gray = cv2.cvtColor(image_ndarray, cv2.COLOR_BGR2GRAY)
        img_smooth = scipy.ndimage.filters.gaussian_filter(gray, sigma)

        dphi_x, dphi_y = np.gradient(img_smooth)
        dphi_pow = np.square(dphi_x) + np.square(dphi_y)
        Force = 1. / (1. + dphi_pow)

        for i in range(it):
            print(i)
            dphi_x, dphi_y = np.gradient(img_smooth)
            dphi = np.square(dphi_x) + np.square(dphi_y)
            dphi_norm = np.sqrt(dphi)
            Force = 1. / (1. + dphi)

            img_smooth = img_smooth + dt * Force * dphi_norm

        img_show = img_smooth.copy()
        img_show = np.uint8(img_show)
        ret, img_show = cv2.threshold(img_show, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        (cnts,_) = cv2.findContours(img_show, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        area = np.array([cv2.contourArea(cnts[i]) for i in range(len(cnts))])
        maxa_ind = np.argmax(area)
        result = image_ndarray.copy()    
        result = image_process.extract_image(result, cnts[maxa_ind])
        return result

    @staticmethod
    def extract_image(origin_image, contour):
        x,y,w,h = cv2.boundingRect(contour)
        gray = cv2.cvtColor(origin_image, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gray)
        cv2.fillConvexPoly(mask, points=contour, color=(255))
        mask_out = np.zeros(origin_image.shape, np.uint8)
        mask_out[mask == 255] = origin_image[mask == 255]
        return mask_out

    @staticmethod
    def crop_image(origin_image):
        # Crop image
        x,y,w,h = cv2.boundingRect(origin_image)
        crop_out = origin_image[y:y+h, x:x+w]
        return crop_out


if __name__ == "__main__":
    Image = image_process()
    Image.process_all('fire_ant/Image', 'output_image/')
