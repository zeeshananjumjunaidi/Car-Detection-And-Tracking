'''
Main File: Vehicle Detection And Training
'''
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from VehicleDetector import VehicleDetector
from VehicleClassifier import VehicleClassifier
from moviepy.editor import VideoFileClip


# Training images paths
TRAINING_PATH_VEHICLES = "data/vehicles"
TRAINING_PATH_NON_VEHICLES = "data/non-vehicles"
TRAINING_PATH_TRAINED_MODEL = "."

# Test image constants
TEST_IMAGES_PATH = './test_images'
TEST_FILE_NAME = "test3.jpg"
OUTPUT_PATH = './output_images'
MODEL_NAME = 'VDT6.p'

def process_single_image(file_name, vehicle_classifier,verbose=False,save_output=False):
    '''Detect vehicles in a single test image, show images of subsequent pipeline steps'''
    vehicle_detector = VehicleDetector(vehicle_classifier, OUTPUT_PATH)

    image = mpimg.imread(file_name)

    # Perform vehicle detection
    if verbose:
        image_windows, sliding_windows, image_det, detected_windows = vehicle_detector.detect(image, verbose=True)


        # Plot and show the resulting image (for the README.md)
        if image_windows is not None:
            plt.figure(figsize=(12,8))
            plt.imshow(image_windows)
            plt.show()

            # Save a copy of the result reduced to 25% of the original size
            image_windows = cv2.resize(image_windows, (int(image_windows.shape[1]/2), int(image_windows.shape[0]/2)))
            if save_output:
                plt.imsave(OUTPUT_PATH + "/test_sliding_windows_grid.jpg", image_windows)
                # Plot and show the resulting image (for the README.md)
        if image_det is not None:
            plt.figure(figsize=(12, 8))
            plt.imshow(image_det)
            plt.show()

            # Save a copy of the result reduced to 25% of the original size
            image_det = cv2.resize(image_det, (int(image_det.shape[1] / 2), int(image_det.shape[0] / 2)))
            if save_output:
                plt.imsave(OUTPUT_PATH + "/test_hot_windows.jpg", image_det)
    else:
        image_output = vehicle_detector.detect(image, verbose=False)
        # Plot and show the resulting image (for the README.md)
        if image_output is not None:
            plt.figure(figsize=(12,8))
            plt.imshow(image_output)
            plt.show()

            # Save a copy of the result reduced to 25% of the original size
            image_det = cv2.resize(image_output, (int(image_output.shape[1]/2), int(image_output.shape[0]/2)))
            if save_output:
                plt.imsave(OUTPUT_PATH + "/test_hot_windows.jpg", image_det)


def process_test_images(vehicle_classifier):
    '''Detect vehicles in all six test images'''
    vehicle_detector = VehicleDetector(vehicle_classifier, OUTPUT_PATH)

    images = []
    for i in range(0, 17):
        images.append(mpimg.imread(TEST_IMAGES_PATH + '/test{}.jpg'.format(i+1)))

    # Plot the original image next to the image showing the detected vehicles
    fig, axis = plt.subplots(len(images), 2)
    for row in range(len(images)):
        image_org = images[row]
        axis[row, 0].imshow(image_org)
        axis[row, 0].axis('off')

        # Detect the vehicles
        image_det = vehicle_detector.detect(image_org, verbose=False, new_image=True)

        axis[row, 1].imshow(image_det)
        axis[row, 1].axis('off')
        image_tmp = cv2.resize(image_det, (int(image_det.shape[1] / 2), int(image_det.shape[0] / 2)))
        plt.imsave(OUTPUT_PATH + '/test{}_detected.jpg'.format(row+1), image_tmp)

    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.show()          # Manually save the image to disk for use in the README.md


def process_video(video_name, vehicle_classifier):
    '''Detect vehicles in an entire video and write the result to disc'''
    vehicle_detector = VehicleDetector(vehicle_classifier, OUTPUT_PATH)

    video_input = VideoFileClip(video_name + ".mp4")
    video_output = 'output_' + video_name + ".mp4"
    output = video_input.fl_image(vehicle_detector.detect)
    output.write_videofile(video_output, audio=False)

def train_classifier():
    vc = VehicleClassifier(TRAINING_PATH_TRAINED_MODEL)
    vehicles, non_vehicles = vc.load_training_images(TRAINING_PATH_VEHICLES, TRAINING_PATH_NON_VEHICLES)
    X, y = vc.extract_features(vehicles, non_vehicles)
    vc.train(X, y)
    vc.save_model(MODEL_NAME)


import FeatureExtractor
from Config import *
def generate_sample_for_write_up():
    img=plt.imread('./data/vehicles/GTI_Left/image0009.png')
    cvt_img=cv2.cvtColor(img,cv2.COLOR_RGB2YCrCb)
    gray=cv2.cvtColor(cvt_img,cv2.COLOR_RGB2GRAY)
    hog_img = FeatureExtractor.get_hog_features(gray, orientations=HOG_ORIENT_BINS, pixels_per_cell=(HOG_PIX_PER_CELL, HOG_PIX_PER_CELL),
                       cells_per_block=(HOG_CELL_PER_BLOCK, HOG_CELL_PER_BLOCK), transform_sqrt=True,
                       visualise=True,feature_vector=True)
    print(hog_img[1].shape)
    #hog_img._g
    plt.imshow(gray,cmap='gray')
    plt.show()
    plt.imshow(hog_img[1],cmap='gray')
    plt.show()

if __name__ == "__main__":
    # Train a VehicleClassifier and save it to disc
    #train_classifier()

    # Load a previously trained VehicleClassifier
    vc2 = VehicleClassifier(TRAINING_PATH_TRAINED_MODEL)
    vc2.load_model(MODEL_NAME)

    # Process a single test image
    TEST_FILE_NAME = "test17.jpg"
    process_single_image(TEST_IMAGES_PATH + '/' + TEST_FILE_NAME, vc2,verbose=True)

    # Process all images in test_image folder
    #process_test_images(vc2)

    # Process the video using the trained classifier
    #process_video('difficult_video',vc2)
    #process_video("test_video", vc2)
    process_video("project_video", vc2)

    #to get hog feature image for write_up document
    #generate_sample_for_write_up()

