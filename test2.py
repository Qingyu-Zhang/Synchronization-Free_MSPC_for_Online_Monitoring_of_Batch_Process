from src.preprocessing import load_data
from run_demo import run_whole_process

def main():
    X_augmented_filepath = "data/BatchModel_TrainingDataset.csv"
    X_augmented = load_data(X_augmented_filepath)
    X_augmented = X_augmented.iloc[:, 1:10]

    new_batch_filepath = "data/example_newbatch_B64_Faulty.csv"
    new_batch = load_data(new_batch_filepath).iloc[:,2:]

    run_whole_process(X_augmented, new_batch, n_components=3,  Mannual_Path=True)



if __name__ == "__main__":
    main()