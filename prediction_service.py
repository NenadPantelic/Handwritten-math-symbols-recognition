
import sys
from os import listdir, sep
import numpy as np
import pickle
from PIL import Image
import cv2
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from PIL import Image


DEFAULT_IMAGE_SIZE = (45,45)
SPLIT_POINT_COEFF = 0.8

class PredictionService:


        def __init__(self):
            #self.dataset = datasetDir
            #self.testImage = testImage
            pass

        def getImageVector(self, image):
            try:

                #NOTE:from docs
                #When translating a color image to black and white (mode “L”), the library uses the ITU-R 601-2 luma transform:
                #L = R * 299/1000 + G * 587/1000 + B * 114/1000

                imageGrayscale = Image.open(image).convert('L')
                #resize image to default image size - 45 x 45
                imageGrayscale = imageGrayscale.resize(DEFAULT_IMAGE_SIZE, Image.ANTIALIAS)
                #
                imageNP = np.array(imageGrayscale)
                imgList = []
                for line in imageNP:
                    for value in line:
                        imgList.append(value)
                #imgList is 2025 long vector
                return imgList
            except Exception as e:
                print("Error : {}".format(e))
                return None

        def addImagesToSet(self, rootPath, imageList, label, completeImageList = [], labelList = []):
            dashes = ['-','/','-','\\']
            counter = 0
            for image in imageList:
                print('[{}] Images loading...'.format(dashes[counter]))
                counter = (counter + 1) % len(dashes)
                completeImageList.append(self.getImageVector(rootPath + image))
                labelList.append(label)


        def getTrainingAndTestData(self, directoryPath):

            dirList = listdir(directoryPath)
            xTrain, yTrain, xTest, yTest = [], [], [], []
            try:
                if len(dirList) < 1:
                    return None

                imageDirPath = None

                counter = 1
                for directory in dirList:

                    imageDir = listdir('{}/{}'.format(directoryPath, directory))
                    splitPoint = int(SPLIT_POINT_COEFF * len(imageDir))

                    print('[{}] Loading dataset - {} images'.format(counter, directory))
                    counter += 1

                    trainImages, testImages = imageDir[:splitPoint], imageDir[splitPoint:]
                    imageDirPath = directoryPath + sep + directory + sep
                    self.addImagesToSet(imageDirPath, trainImages, directory, xTrain, yTrain)
                    self.addImagesToSet(imageDirPath, testImages, directory, xTest, yTest)

            except Exception as e:
                print('Error: {}'.format(e))
                return [],[],[],[]

            return xTrain, yTrain, xTest, yTest

        def trainModel(self, trainDatasetDir):
            #train_dataset_dir = "./Dataset/extracted_images/"
            print('Training.....')
            xTrain, yTrain, xTest , yTest = self.getTrainingAndTestData(trainDatasetDir)
            if [] not in (xTrain, yTrain, xTest , yTest):
                randomForestClassifier = RandomForestClassifier()
                randomForestClassifier.fit(xTrain,yTrain)
                accuracyScore = randomForestClassifier.score(xTrain,yTrain)
                # save classifier
                pickle.dump(randomForestClassifier,open("Model/math_recognition_model.pkl",'wb'))
                print("Model Accuracy Score : {}".format(accuracyScore))
                testAccuracyScore = randomForestClassifier.score(xTest,yTest)
                print("Model Accuracy Score (Test) : {}".format(testAccuracyScore))
            else :
                print("An error occurred.")

        def predict(self, imagePath):
            try:
                image = [self.getImageVector(imagePath)]
                # load saved model
                try:
                    decisionTreeClassifierModel = pickle.load(open("Model/random_forest_classifier.pkl",'rb'))
                    modelPrediction = decisionTreeClassifierModel.predict(image)
                    print(modelPrediction)
                    print("Recognized expression:" +  str(modelPrediction[0]))
                except FileNotFoundError as modelFileError:
                    print("Error : {}".format(modelFileError))
                    self.trainModel(datasetDir)
                    self.predict(imagePath)

            except FileNotFoundError as fileError:
                print("Error : {}".format(fileError))
            except Exception as e:
                print("Error : {}".format(e))
