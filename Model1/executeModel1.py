from model1 import Model1

SAVEPATH = "../SavedModels/Model1/"
DATAPATH = "../FER2013/icml_face_data.csv"
NAME = "model1"

model = Model1(SAVEPATH, DATAPATH, name=NAME)
#model.train()
model.loadModel()
#model.evaluateOnValid()
#model.evaluateOnTest()
model.generatePlots()