from prediction_service import PredictionService



ps = PredictionService()
#print(imgs.getImageVector())
#ps.trainModel("./Dataset/extracted_images/")
ps.predict("./TestData/test2.jpg")
