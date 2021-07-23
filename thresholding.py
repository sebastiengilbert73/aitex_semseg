"""
Code by Sébastien Gilbert

Reference for the dataset:
https://www.aitex.es/afid/
AFID: a  public fabric image database for defect detection.
Javier Silvestre-Blanes, Teresa Albero-Albero, Ignacio Miralles, Rubén Pérez-Llorens, Jorge Moreno
AUTEX Research Journal, No. 4, 2019

Note: Mask_images/0044_019_04_mask1.png and
                  0044_019_04_mask2.png
                ... have been merged into
                  0044_019_04_mask.png
      Mask_images/0097_030_03_mask1.png and
                  0097_030_03_mask2.png
                ... have been merged into
                  0097_030_03_mask.png
      Mask_images/0100_025_08_mask.png was created manually since it was missing in the original dataset
"""

import logging
import argparse
import os
import cv2
import numpy as np
import sys
import utilities.blob_analysis as blob_analysis

parser = argparse.ArgumentParser()
parser.add_argument('--imageFilepath', help="The filepath of the test image. Default: './data/Defect_images/0001_002_00.png'", default='./data/Defect_images/0001_002_00.png')
parser.add_argument('--outputDirectory', help="The output directory. Default: './outputs/'", default='./outputs/')
parser.add_argument('--thresholdDeltaAboveMedian', help="The additive value to the median gray level for the bright pixels threshold. Default: 34", type=int, default=34)
parser.add_argument('--thresholdDeltaBelowMedian', help="The subtractive value to the median gray level for the dark pixels inverse threshold. Default: 14", type=int, default=14)
parser.add_argument('--blobMinimumDimension', help="The minimum dimension of a blob, in pixels. Default: 9", type=int, default=9)
parser.add_argument('--blobMaximumDimension', help="The maximum dimension of a blob, in pixels. Default: 300", type=int, default=300)
args = parser.parse_args()

file_handler = logging.FileHandler(filename=os.path.join(args.outputDirectory, 'texture_analysis.log'))
stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    handlers=handlers
)

def main():
    logging.info("thresholding.py main()")

    # Create the output directory
    if not os.path.exists(args.outputDirectory):
        os.makedirs(args.outputDirectory)

    # Load the test image
    testImg = cv2.imread(args.imageFilepath, cv2.IMREAD_GRAYSCALE)
    image_shapeHWC = testImg.shape
    blurredImg = cv2.medianBlur(testImg, 3)
    blurredImg = cv2.blur(blurredImg, ksize=(7, 7))

    # Mask the saturated  area at the left of the image, plus an additional band
    _, saturation_mask = cv2.threshold(testImg, 254, 255, cv2.THRESH_BINARY)
    saturation_mask = cv2.dilate(saturation_mask, kernel=np.ones((21, 21), dtype=np.uint8))
    no_saturation_img = cv2.min(testImg, (255 - saturation_mask))
    no_saturation_blurred_img = cv2.min(blurredImg, (255 - saturation_mask))
    # First x to consider
    start_x = NoSaturationStart(no_saturation_blurred_img)
    logging.info("start_x = {}".format(start_x))

    # Average gray level as a function of x
    vertical_averages = VerticalAverages(no_saturation_blurred_img)
    # To do: RANSAC polynomial to describe the vertical average as a function of x
    with open(os.path.join(args.outputDirectory, "thresholding_verticalAverages.csv"), 'w+') as vertical_averages_file:
        vertical_averages_file.write("x,average\n")
        for x in range(len(vertical_averages)):
            vertical_averages_file.write("{},{}\n".format(x, vertical_averages[x]))

    # Compute the image median
    image_median = np.median(blurredImg[:, start_x:])
    logging.info("image_median = {}".format(image_median))

    # Apply inverse threshold to detect dark areas
    _, inverse_thresholded_img = cv2.threshold(blurredImg, image_median - args.thresholdDeltaBelowMedian, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite(os.path.join(args.outputDirectory, "thresholding_inverseThresholded.png"), inverse_thresholded_img)

    # Apply threshold to detect bright areas
    _, thresholded_img = cv2.threshold(blurredImg, image_median + args.thresholdDeltaAboveMedian, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(args.outputDirectory, "thresholding_thresholded.png"), thresholded_img)

    # Erode and dilate both masks
    erosion_kernel = np.ones((3, 3), dtype=np.uint8)
    inverse_thresholded_img = cv2.erode(inverse_thresholded_img, erosion_kernel)
    inverse_thresholded_img = cv2.dilate(inverse_thresholded_img, erosion_kernel)
    thresholded_img = cv2.erode(thresholded_img, erosion_kernel)
    thresholded_img = cv2.dilate(thresholded_img, erosion_kernel)

    # Merge the masks
    anomalies_mask = cv2.max(inverse_thresholded_img, thresholded_img)
    anomalies_mask[:, 0: start_x] = 0

    # Analyze the blob dimensions
    blob_detector = blob_analysis.BinaryBlobDetector()
    seedPoint_boundingBox_list, blobs_annotated_img = blob_detector.DetectBlobs(anomalies_mask)

    # Filter out the small blobs
    filtered_anomalies_mask = anomalies_mask.copy()
    filtered_seedPoint_boundingBox_list = []
    floodfill_mask = np.zeros((filtered_anomalies_mask.shape[0] + 2, filtered_anomalies_mask.shape[1] + 2), dtype=np.uint8)
    for (seed_point, bounding_box) in seedPoint_boundingBox_list:
        blob_dimension = max(bounding_box[2], bounding_box[3])
        if blob_dimension < args.blobMinimumDimension or \
            blob_dimension > args.blobMaximumDimension:
            cv2.floodFill(filtered_anomalies_mask, floodfill_mask, seed_point, 0)
        else:
            filtered_seedPoint_boundingBox_list.append((seed_point, bounding_box))


    # Save intermediary images
    cv2.imwrite(os.path.join(args.outputDirectory, "thresholding_original.png"), testImg)
    cv2.imwrite(os.path.join(args.outputDirectory, "thresholding_blurred.png"), blurredImg)
    cv2.imwrite(os.path.join(args.outputDirectory, "thresholding_saturationMask.png"), saturation_mask)
    cv2.imwrite(os.path.join(args.outputDirectory, "thresholding_inverseThresholdedEroded.png"), inverse_thresholded_img)
    cv2.imwrite(os.path.join(args.outputDirectory, "thresholding_thresholdedEroded.png"), thresholded_img)
    cv2.imwrite(os.path.join(args.outputDirectory, "thresholding_anomalies.png"), anomalies_mask)
    cv2.imwrite(os.path.join(args.outputDirectory, "thresholding_blobs.png"), blobs_annotated_img)
    cv2.imwrite(os.path.join(args.outputDirectory, "thresholding_filteredAnomalies.png"), filtered_anomalies_mask)

def NoSaturationStart(no_saturation_mask):
    x = 0
    while x < no_saturation_mask.shape[1]:
        if cv2.countNonZero(no_saturation_mask[:, x]) > 0:
            return x
        x += 1
    # We reached the end without finding a column with non-zero values
    return 0

def VerticalAverages(image):
    vertical_averages = []
    for x in range(image.shape[1]):
        vertical_average = np.mean(image[:, x])
        vertical_averages.append(vertical_average)
    return vertical_averages

if __name__ == '__main__':
    main()