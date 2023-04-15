from csv_shuffler import csv_shuffler

shuffler = csv_shuffler.ShuffleCSV(input_file='./Data/TEST_DATA_Q.csv', header=True, batch_size=20000)

shuffler.shuffle_csv()