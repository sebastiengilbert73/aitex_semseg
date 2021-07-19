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
import ast
import random
import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import vision_genprog.tasks.image_processing as image_processing
import vision_genprog.semanticSegmentersPop as semsegPop
import sys
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('--imagesDirectory', help="The directory containing the image directories. Default: './data'", default='./data')
parser.add_argument('--outputDirectory', help="The output directory. Default: './outputs/'", default='./outputs/')
parser.add_argument('--constantCreationParametersList', help="The parameters to use when creating constants: [minFloat, maxFloat, minInt, maxInt, width, height]. Default: '[-1, 1, 0, 255, 4096, 256]'", default='[-1, 1, 0, 255, 4096, 256]')
parser.add_argument('--primitivesFilepath', help="The filepath to the XML file for the primitive functions. Default: './vision_genprog/tasks/image_processing.xml'", default='./vision_genprog/tasks/image_processing.xml')
parser.add_argument('--levelToFunctionProbabilityDict', help="The probability to generate a function, at each level. Default: '{0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}'", default='{0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}')
parser.add_argument('--numberOfIndividuals', help="The number of individuals. Default: 64", type=int, default=64)
parser.add_argument('--proportionOfConstants', help='The probability to generate a constant, when a variable could be used. Default: 0', type=float, default=0)
parser.add_argument('--weightForNumberOfNodes', help="Penalty term proportional to the number of nodes. Default: 0.001", type=float, default=0.001)
parser.add_argument('--numberOfTournamentParticipants', help="The number of participants in selection tournaments. Default: 2", type=int, default=2)
parser.add_argument('--mutationProbability', help="The probability to mutate a child. Default: 0.2", type=float, default=0.2)
parser.add_argument('--proportionOfNewIndividuals', help="The proportion of randomly generates individuals per generation. Default: 0.2", type=float, default=0.2)
parser.add_argument('--maximumNumberOfMissedCreationTrials', help="The maximum number of missed creation trials. Default: 1000", type=int, default=1000)
parser.add_argument('--maximumValidationIoUToStop', help="The champion validation average intersection over union to stop. Default: 0.05", type=float, default=0.05)
parser.add_argument('--maximumNumberOfGenerations', help="The maximum number of generations. Default: 32", type=int, default=32)
args = parser.parse_args()

constantCreationParametersList = ast.literal_eval(args.constantCreationParametersList)
image_shapeHW = (constantCreationParametersList[5], constantCreationParametersList[4])
levelToFunctionProbabilityDict = ast.literal_eval(args.levelToFunctionProbabilityDict)

#logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

file_handler = logging.FileHandler(filename=os.path.join(args.outputDirectory, 'create_semseg_population.log'))
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

    # Check if the expected directories exist
    defect_images_directory = os.path.join(args.imagesDirectory, "Defect_images")
    mask_images_directory = os.path.join(args.imagesDirectory, "Mask_images")
    noDefect_images_directory = os.path.join(args.imagesDirectory, "NODefect_images")
    if not os.path.exists(defect_images_directory):
        raise IsADirectoryError("main(): The directory '{}' doesn't exist".format(defect_images_directory))
    if not os.path.exists(mask_images_directory):
        raise IsADirectoryError("main(): The directory '{}' doesn't exist".format(mask_images_directory))
    if not os.path.exists((noDefect_images_directory)):
        raise IsADirectoryError("main(): The directory '{}' doesn't exist".format(noDefect_images_directory))

    inputOutput_tuples = InputOutputTuplesList(defect_images_directory, mask_images_directory,
                                               noDefect_images_directory, image_shapeHW)

    # Split in train - validation - test
    # Shuffle the list
    random.shuffle(inputOutput_tuples)
    validation_start_ndx = round(0.6 * len(inputOutput_tuples))
    test_start_ndx = round(0.8 * len(inputOutput_tuples))
    train_tuples = inputOutput_tuples[0: validation_start_ndx]
    validation_tuples = inputOutput_tuples[validation_start_ndx: test_start_ndx]
    test_tuples = inputOutput_tuples[test_start_ndx:]
    logging.debug("len(train_tuples) = {}; len(validation_tuples) = {}; len(test_tuples) = {}".format(len(train_tuples), len(validation_tuples), len(test_tuples)))

    # Create the interpreter
    primitive_functions_tree = ET.parse(args.primitivesFilepath)
    interpreter = image_processing.Interpreter(primitive_functions_tree, image_shapeHW)

    variableName_to_type = {'image': 'grayscale_image'}
    return_type = 'binary_image'  # We're doing semantic segmentation with only two classes

    # Create a population
    logging.info("Creating a population...")
    semseg_pop = semsegPop.SemanticSegmentersPopulation()
    semseg_pop.Generate(
        numberOfIndividuals=args.numberOfIndividuals,
        interpreter=interpreter,
        returnType=return_type,
        levelToFunctionProbabilityDict=levelToFunctionProbabilityDict,
        proportionOfConstants=args.proportionOfConstants,
        constantCreationParametersList=constantCreationParametersList,
        variableNameToTypeDict=variableName_to_type,
        functionNameToWeightDict=None
    )

    # Evaluate the original population
    logging.info("Evaluating the original population...")
    individual_to_cost_dict = semseg_pop.EvaluateIndividualCosts(
        inputOutputTuplesList=train_tuples,
        variableNameToTypeDict=variableName_to_type,
        interpreter=interpreter,
        returnType=return_type,
        weightForNumberOfElements=args.weightForNumberOfNodes
    )
    print ("individual_to_cost_dict.values() = {}".format(individual_to_cost_dict.values()))

    logging.info("Starting the population evolution...")
    final_champion = None
    lowest_validation_IoU = sys.float_info.max
    evolution_must_continue = True
    with open(os.path.join(args.outputDirectory, "generations.csv"), 'w+') as generations_file:
        generations_file.write("generation,train_lowest_cost,train_median_cost,champion_validation_averageIoU\n")

    # for generationNdx in range(1, args.numberOfGenerations + 1):
    generationNdx = 1
    while evolution_must_continue:
        logging.info(" ***** Generation {} *****".format(generationNdx))
        individual_to_cost_dict = semseg_pop.NewGenerationWithTournament(
            inputOutputTuplesList=train_tuples,
            variableNameToTypeDict=variableName_to_type,
            interpreter=interpreter,
            returnType=return_type,
            numberOfTournamentParticipants=args.numberOfTournamentParticipants,
            mutationProbability=args.mutationProbability,
            currentIndividualToCostDict=individual_to_cost_dict,
            proportionOfConstants=args.proportionOfConstants,
            levelToFunctionProbabilityDict=levelToFunctionProbabilityDict,
            functionNameToWeightDict=None,
            constantCreationParametersList=constantCreationParametersList,
            proportionOfNewIndividuals=args.proportionOfNewIndividuals,
            weightForNumberOfElements=args.weightForNumberOfNodes,
            maximumNumberOfMissedCreationTrials=args.maximumNumberOfMissedCreationTrials
        )

        (champion, lowest_cost) = semseg_pop.Champion(individual_to_cost_dict)
        median_cost = semseg_pop.MedianCost(individual_to_cost_dict)

        # Validation
        champion_validation_intersection_over_union_list = semseg_pop.BatchIntersectionOverUnion(champion,
                                                                                        validation_tuples, variableName_to_type,
                                                                                        interpreter, return_type)
        champion_validation_averageIoU = statistics.mean(champion_validation_intersection_over_union_list)

        logging.info(
            "Generation {}: lowest cost = {}; median cost = {}; champion_validation_averageIoU = {}".format(generationNdx,
                                                                                                 lowest_cost,
                                                                                                 median_cost,
                                                                                                 champion_validation_averageIoU))
        with open(os.path.join(args.outputDirectory, "generations.csv"), 'a+') as generations_file:
            generations_file.write("{},{},{},{}\n".format(generationNdx, lowest_cost, median_cost, champion_validation_averageIoU))

        # Save the champion
        champion_filepath = os.path.join(args.outputDirectory,
                                         "champion_{}_{:.4f}_{:.4f}.xml".format(generationNdx, lowest_cost,
                                                                                champion_validation_averageIoU))
        champion.Save(champion_filepath)
        if champion_validation_averageIoU < lowest_validation_IoU:
            lowest_validation_IoU = champion_validation_averageIoU
            final_champion = champion
        if champion_validation_averageIoU <= args.maximumValidationIoUToStop:
            evolution_must_continue = False
        generationNdx += 1
        if generationNdx > args.maximumNumberOfGenerations:
            evolution_must_continue = False

    logging.info("Testing the final champion...")
    champion_test_intersection_over_union_list = semseg_pop.BatchIntersectionOverUnion(final_champion,
                                                                                             test_tuples,
                                                                                             variableName_to_type,
                                                                                             interpreter, return_type)
    champion_test_averageIoU = statistics.mean(champion_test_intersection_over_union_list)
    logging.info("champion_test_averageIoU = {}".format(champion_test_averageIoU))


def InputOutputTuplesList(defect_images_directory, mask_images_directory, noDefect_images_directory, image_shapeHW):
    # List[Tuple[Dict[str, Any], Any]]
    inputOutputTuples = []
    defect_image_filepaths = ImageFilepaths(defect_images_directory)
    mask_image_filepaths = ImageFilepaths(mask_images_directory)
    noDefect_directories = [os.path.join(noDefect_images_directory, o) for o in os.listdir(noDefect_images_directory)
                    if os.path.isdir(os.path.join(noDefect_images_directory, o))]  # Cf. https://stackoverflow.com/questions/973473/getting-a-list-of-all-subdirectories-in-the-current-directory
    #logging.debug("InputOutputTuplesList(): noDefect_directories = {}".format(noDefect_directories))
    for defect_image_filepath in defect_image_filepaths:
        defect_image_filename = os.path.basename(defect_image_filepath)
        mask_image_filename = defect_image_filename[: -4] + '_mask.png'
        corresponding_mask_filepath = os.path.join(mask_images_directory, mask_image_filename)
        if not corresponding_mask_filepath in mask_image_filepaths:
            raise FileNotFoundError("The filepath '{}' doesn't exist".format(corresponding_mask_filepath))
        image = cv2.imread(defect_image_filepath, cv2.IMREAD_GRAYSCALE)
        if image.shape != image_shapeHW:
            logging.warning("InputOutputTuplesList(): Resizing image {} to {}".format(defect_image_filepath, image_shapeHW))
            image = cv2.resize(image, image_shapeHW)

        mask = cv2.imread(corresponding_mask_filepath, cv2.IMREAD_GRAYSCALE)
        if mask.shape != image_shapeHW:
            logging.warning("InputOutputTuplesList(): Resizing mask {} to {}".format(corresponding_mask_filepath, image_shapeHW))
            mask = cv2.resize(mask, image_shapeHW)
        # Apply threshold to get a truly binary image
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        input = {'image': image}
        inputOutputTuples.append((input, mask))

    for noDefect_directory in noDefect_directories:
        noDefect_image_filepaths = ImageFilepaths(noDefect_directory)
        for noDefect_image_filepath in noDefect_image_filepaths:
            image = cv2.imread(noDefect_image_filepath, cv2.IMREAD_GRAYSCALE)
            if image.shape != image_shapeHW:
                logging.warning("InputOutputTuplesList(): Resizing image {} to {}".format(noDefect_image_filepath, image_shapeHW))
                image = cv2.resize(image, image_shapeHW)
            mask = np.zeros(image_shapeHW, dtype=np.uint8)  # The mask is completely black since there is no defect
            input = {'image': image}
            inputOutputTuples.append((input, mask))

    return inputOutputTuples

def ImageFilepaths(images_directory):
    image_filepaths_in_directory = [os.path.join(images_directory, filename) for filename in os.listdir(images_directory)
                              if os.path.isfile(os.path.join(images_directory, filename))
                              and filename.upper().endswith('.PNG')]
    return image_filepaths_in_directory



if __name__ == '__main__':
    main()
