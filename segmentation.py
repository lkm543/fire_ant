import glob
import os

import cv2
import numpy as np

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
        result = image_process.crop_image(result, hull)
        return result

    @staticmethod
    def crop_image(origin_image, contour):
        x,y,w,h = cv2.boundingRect(contour)
        gray = cv2.cvtColor(origin_image, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gray)
        cv2.fillConvexPoly(mask, points=contour, color=(255))
        mask_out = np.zeros(origin_image.shape, np.uint8)
        mask_out[mask == 255] = origin_image[mask == 255]

        # Crop image
        x,y,w,h = cv2.boundingRect(contour)
        mask_out = mask_out[y:y+h, x:x+w]
        return mask_out


if __name__ == "__main__":
    Image = image_process()
    Image.process_all('fire_ant/Image', 'output_image/')
