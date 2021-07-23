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
import PCA.ImagePCAModel


parser = argparse.ArgumentParser()
parser.add_argument('--imageFilepath', help="The filepath of the test image. Default: './data/Defect_images/0001_002_00.png'", default='./data/Defect_images/0001_002_00.png')
parser.add_argument('--outputDirectory', help="The output directory. Default: './outputs/'", default='./outputs/')
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
    logging.info("create_semseg_population.py main()")

    # Create the output directory
    if not os.path.exists(args.outputDirectory):
        os.makedirs(args.outputDirectory)

    # Load the test image
    testImg = cv2.imread(args.imageFilepath, cv2.IMREAD_GRAYSCALE)
    image_shapeHWC = testImg.shape
    blurredImg = cv2.medianBlur(testImg, 3)

    # Mask the saturated  area at the left of the image, plus an additional band
    _, saturation_mask = cv2.threshold(testImg, 254, 255, cv2.THRESH_BINARY)
    saturation_mask = cv2.dilate(saturation_mask, kernel=np.ones((21, 21), dtype=np.uint8))
    no_saturation_img = cv2.min(testImg, (255 - saturation_mask))
    no_saturation_blurred_img = cv2.min(blurredImg, (255 - saturation_mask))

    # Apply Sobel filter in the x and y directions
    sobel_x_img = cv2.Sobel(no_saturation_blurred_img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    sobel_x_img = (127 + sobel_x_img).astype(np.uint8)
    sobel_y_img = cv2.Sobel(no_saturation_blurred_img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
    sobel_y_img = (127 + sobel_y_img).astype(np.uint8)
    sobel_xy_img = cv2.Sobel(no_saturation_blurred_img, ddepth=cv2.CV_32F, dx=1, dy=1, ksize=3)
    sobel_xy_img = (127 + sobel_xy_img).astype(np.uint8)
    sobel_x2_img = cv2.Sobel(no_saturation_blurred_img, ddepth=cv2.CV_32F, dx=2, dy=0, ksize=3)
    sobel_x2_img = (127 + sobel_x2_img).astype(np.uint8)
    sobel_y2_img = cv2.Sobel(no_saturation_blurred_img, ddepth=cv2.CV_32F, dx=0, dy=2, ksize=3)
    sobel_y2_img = (127 + sobel_y2_img).astype(np.uint8)

    # Select surface samples
    number_of_surface_samples = 300
    sample_surface_sizeHW = (5, 5)
    surface_sample_rect_list = []
    surface_samples_img = cv2.cvtColor(testImg, cv2.COLOR_GRAY2BGR)
    while len(surface_sample_rect_list) < number_of_surface_samples:
        center = (np.random.randint(sample_surface_sizeHW[0]//2, image_shapeHWC[1] - 1 - sample_surface_sizeHW[0]//2),
                  np.random.randint(sample_surface_sizeHW[1] // 2, image_shapeHWC[0] - 1 - sample_surface_sizeHW[1] // 2))
        sample_surface_rect = (center[0] - sample_surface_sizeHW[0]//2, center[1] - sample_surface_sizeHW[1]//2,
                               sample_surface_sizeHW[0], sample_surface_sizeHW[1])
        if cv2.countNonZero(saturation_mask[sample_surface_rect[1]: sample_surface_rect[1] + sample_surface_rect[3],
                            sample_surface_rect[0]: sample_surface_rect[0] + sample_surface_rect[2]]) == 0:
            surface_sample_rect_list.append(sample_surface_rect)

    # Collect vectors of numbers in the surface samples
    samples = []  # Will be a list of color images
    for surface_sample_rect in surface_sample_rect_list:
        cv2.rectangle(surface_samples_img, (surface_sample_rect[0], surface_sample_rect[1]),
                      (surface_sample_rect[0] + surface_sample_rect[2], surface_sample_rect[1] + surface_sample_rect[3]), (255, 0, 0))
        x = surface_sample_rect[0]
        y = surface_sample_rect[1]
        width = surface_sample_rect[2]
        height = surface_sample_rect[3]
        sample_color_img = np.zeros((sample_surface_sizeHW[0], sample_surface_sizeHW[1], 3), dtype=np.uint8)
        sample_color_img[:, :, 0] = blurredImg[y: y + height, x: x + width]
        sample_color_img[:, :, 1] = sobel_x_img[y: y + height, x: x + width]
        sample_color_img[:, :, 2] = sobel_y_img[y: y + height, x: x + width]
        samples.append(sample_color_img)

    # Create a color PCA model
    color_pca_model = PCA.ImagePCAModel.MultiChannelModel(samples)
    model_average_img = color_pca_model.AverageImage()
    eigenimages = color_pca_model.EigenImagesForDisplay()
    for eigenNdx in range(len(eigenimages)):
        filepath = os.path.join(args.outputDirectory, "textureAnalysis_eigenImage" + str(eigenNdx) + ".png")
        cv2.imwrite(filepath, eigenimages[eigenNdx])
    variance_proportions = color_pca_model.VarianceProportion()
    running_proportions = RunningSums(variance_proportions)
    logging.info("variance_proportion = {}".format(variance_proportions))
    logging.info("running_proportions = {}".format(running_proportions))
    # Index of the eigenvector to keep the desired variance
    desired_variance_proportion = 0.5
    eigen_index_to_keep = 0
    while running_proportions[eigen_index_to_keep] < desired_variance_proportion:
        eigen_index_to_keep += 1
    logging.info("Keeping eigenvectors up to index {}".format(eigen_index_to_keep))
    color_pca_model.TruncateModel(eigen_index_to_keep)

    # Differential image
    differential_img = np.ones((image_shapeHWC[0], image_shapeHWC[1], 3), dtype=np.float32) * 127
    for y in range(sample_surface_sizeHW[0]//2, image_shapeHWC[0] - sample_surface_sizeHW[0]//2 - 1, sample_surface_sizeHW[0]):
        print("{}, ".format(y), end='', flush=True)
        for x in range(sample_surface_sizeHW[1]//2, image_shapeHWC[1] - sample_surface_sizeHW[1]//2 - 1, sample_surface_sizeHW[1]):
            if cv2.countNonZero(saturation_mask[y - sample_surface_sizeHW[0]//2: y + sample_surface_sizeHW[0]//2 + 1,
                           x - sample_surface_sizeHW[1] // 2: x + sample_surface_sizeHW[1] // 2 + 1]) == 0:

                roi = np.zeros((sample_surface_sizeHW[0], sample_surface_sizeHW[1], 3), dtype=np.uint8)
                roi[:, :, 0] = blurredImg[y - sample_surface_sizeHW[0]//2: y + sample_surface_sizeHW[0]//2 + 1,
                               x - sample_surface_sizeHW[1] // 2: x + sample_surface_sizeHW[1] // 2 + 1]
                roi[:, :, 1] = sobel_x_img[y - sample_surface_sizeHW[0]//2: y + sample_surface_sizeHW[0]//2 + 1,
                               x - sample_surface_sizeHW[1] // 2: x + sample_surface_sizeHW[1] // 2 + 1]
                roi[:, :, 2] = sobel_y_img[y - sample_surface_sizeHW[0] // 2: y + sample_surface_sizeHW[0] // 2 + 1,
                               x - sample_surface_sizeHW[1] // 2: x + sample_surface_sizeHW[1] // 2 + 1]
                projection = color_pca_model.Project(roi)
                reconstruction = color_pca_model.Reconstruct(projection)
                difference = roi.astype(np.float32) - reconstruction.astype(np.float32)
                differential_img[y - sample_surface_sizeHW[0] // 2: y + sample_surface_sizeHW[0] // 2 + 1,
                               x - sample_surface_sizeHW[1] // 2: x + sample_surface_sizeHW[1] // 2 + 1] += difference



    # Save intermediary images
    cv2.imwrite(os.path.join(args.outputDirectory, "textureAnalysis_original.png"), testImg)
    cv2.imwrite(os.path.join(args.outputDirectory, "textureAnalysis_blurred.png"), blurredImg)
    cv2.imwrite(os.path.join(args.outputDirectory, "textureAnalysis_saturationMask.png"), saturation_mask)
    cv2.imwrite(os.path.join(args.outputDirectory, "textureAnalysis_noSaturation.png"), no_saturation_img)
    cv2.imwrite(os.path.join(args.outputDirectory, "textureAnalysis_sobelx.png"), sobel_x_img)
    cv2.imwrite(os.path.join(args.outputDirectory, "textureAnalysis_sobely.png"), sobel_y_img)
    cv2.imwrite(os.path.join(args.outputDirectory, "textureAnalysis_sobelxy.png"), sobel_xy_img)
    cv2.imwrite(os.path.join(args.outputDirectory, "textureAnalysis_sobelx2.png"), sobel_x2_img)
    cv2.imwrite(os.path.join(args.outputDirectory, "textureAnalysis_sobely2.png"), sobel_y2_img)
    cv2.imwrite(os.path.join(args.outputDirectory, "textureAnalysis_surfaceSamples.png"), surface_samples_img)
    cv2.imwrite(os.path.join(args.outputDirectory, "textureAnalysis_modelAverage.png"), model_average_img)
    cv2.imwrite(os.path.join(args.outputDirectory, "textureAnalysis_differential.png"), differential_img.astype(np.uint8))


def RunningSums(numbers):
    running_sums = []
    for index in range(len(numbers)):
        sum = 0
        for runningNdx in range(0, index + 1):
            sum += numbers[runningNdx]
        running_sums.append(sum)
    return running_sums

if __name__ == '__main__':
    main()