# Face Mask Classification using CNN

Proyek ini merupakan tugas akhir mata kuliah Visi Komputer yang bertujuan untuk mengklasifikasikan kondisi penggunaan masker pada citra wajah menggunakan Convolutional Neural Network (CNN).

## Kelas Klasifikasi
- with_mask
- without_mask
- incorrect_mask

## Dataset
Dataset berasal dari Kaggle (Face Mask Detection Dataset) yang telah dikonversi dari dataset object detection menjadi dataset klasifikasi.

## Model
Model yang digunakan adalah MobileNetV2 dengan pendekatan transfer learning menggunakan framework PyTorch.

## Tahapan Proyek
1. Persiapan dan preprocessing dataset  
2. Training model CNN  
3. Evaluasi model menggunakan confusion matrix dan classification report  

## Tools & Library
- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Scikit-learn

## Cara Menjalankan
1. Install dependency:
pip install -r requirements.txt

2. Jalankan training:
python src/train_mobilenet.py

3. Evaluasi model:
python src/evaluate_model.py

## Hasil
Model mencapai akurasi validasi sebesar 87%.

## Catatan
Repository ini dibuat untuk keperluan akademik.
