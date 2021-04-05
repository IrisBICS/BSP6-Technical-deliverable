from model1 import Model1

SAVEPATH = "../SavedModels/Model1/"
DATAPATH = "../FER2013/icml_face_data.csv"

model = Model1(SAVEPATH)
#model.train(DATAPATH)
model.loadModel()