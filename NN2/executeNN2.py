from NN2 import NN2

SAVEPATH = "../SavedNNs/NN2/"
DATAPATH = "../FER2013/fer2013_landmarks.csv"
WEIGHTSPATH = "../FER2013/fer2013_weights.csv"
NAME = "NN2"

EPOCHS = 16

model = NN2(SAVEPATH, DATAPATH, WEIGHTSPATH, name=NAME)
#model.train(epochs=EPOCHS)
model.loadModel()
model.evaluateOnValid()
model.evaluateOnTest()
model.generatePlots()
model.generateConfusionMatrices()
model.generateConfusionMatrices(percentage=False)
model.generateConfusionMatrices(normalize=True)