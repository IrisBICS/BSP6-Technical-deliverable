from FinalNN import FinalNN

SAVEPATH = "../SavedNNs/FinalNN/featureBased/"  # decisionBased/
NAME = "FinalNN"
IMAGESPATH = "../FER2013/icml_face_data.csv"
LANDMARKSPATH = "../FER2013/fer2013_landmarks.csv"
WEIGHTSPATH = "../FER2013/fer2013_weights.csv"
NNIMAGESPATH = "../SavedNNs/NN1/"
NNLANDMARKSPATH = "../SavedNNs/NN2/"
NNIMAGESNAME = "NN1"
NNLANDMARKSNAME = "NN2"

COMBINEMODE = "features"  # "decisions"
EPOCHS = 16
LR = 0.00005

model = FinalNN(SAVEPATH, IMAGESPATH, LANDMARKSPATH, WEIGHTSPATH, NNIMAGESPATH, NNLANDMARKSPATH, NNIMAGESNAME, NNLANDMARKSNAME, name=NAME, combine_mode=COMBINEMODE)
#model.train(epochs=EPOCHS, learning_rate=LR)
model.loadModel()
model.evaluateOnValid()
model.evaluateOnTest()
model.generatePlots()
model.generateConfusionMatrices()
model.generateConfusionMatrices(percentage=False)
model.generateConfusionMatrices(normalize=True)