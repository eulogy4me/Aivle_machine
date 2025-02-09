from models.Struct import Model
import os

if __name__ == "__main__":
    path = os.getcwd()
    dataset_path = path + "/datasets"
    models_path = path + "/pkl/struct.pth"

    trainer = Model(dataset_path)
    trainer.train(models_path)
    trainer.evaluate(models_path)