from preprocessing import preprocessData
from training import trainModels
from validation import validateModel

def main() -> None:
    preprocessData()
    trainModels()
    validateModel()

if __name__ == "__main__":
    main()