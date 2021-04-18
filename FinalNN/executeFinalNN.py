from FinalNN import FinalNN

SAVEPATH = "../SavedNNs/FinalNN/"
NAME = "FinalNN"
IMAGESPATH = "../FER2013/icml_face_data.csv"
LANDMARKSPATH = "../FER2013/fer2013_landmarks.csv"
NNIMAGESPATH = "../SavedNNs/NN1/"
NNLANDMARKSPATH = "../SavedNNs/NN2/"
NNIMAGESNAME = "NN1"
NNLANDMARKSNAME = "NN2"

model = FinalNN(SAVEPATH, IMAGESPATH, LANDMARKSPATH, NNIMAGESPATH, NNLANDMARKSPATH, NNIMAGESNAME, NNLANDMARKSNAME, name=NAME)
model.train()
#model.loadModel()
model.evaluateOnValid()
model.evaluateOnTest()
#model.generatePlots()
#model.generateConfusionMatrices()