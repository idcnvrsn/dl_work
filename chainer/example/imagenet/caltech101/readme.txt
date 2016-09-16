<http://www.vision.caltech.edu/Image_Datasets/Caltech101/#Download>
からCaltech101のデータを取得しdataフォルダ直下に展開しておく

python compute_mean.py train_data\norandom.txt
python train_imagenet.py -g 0 data\train_norandom.txt data\test_norandom.txt > log.txt 