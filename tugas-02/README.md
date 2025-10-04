# Tugas-02

Studi kasus yang digunakan: https://www.mdpi.com/2076-3417/15/13/6980

---

### Konsep Dasar Fuzzy Logic

Fuzzy logic bekerja dengan **derajat keanggotaan**, bukan nilai biner. Artinya:
- Suatu nilai bisa **sebagian** masuk ke beberapa kategori sekaligus.
- Contoh: petal length = 5.1 bisa dianggap *agak pendek*, *sedang*, dan *agak panjang*—dengan bobot berbeda.

---

### Cara Kerja: 

### 1. **Fuzzy Membership Function (Gaussian)**
```python
def gaussian_mf(x, mean, sigma):
    return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))
```
- Fungsi ini menilai seberapa “cocok” nilai input terhadap suatu kategori.
	-  `x`: nilai input yang ingin diuji (misalnya panjang kelopak bunga)
	- `mean`: pusat dari kategori fuzzy (misalnya rata-rata panjang kelopak untuk Setosa)
	- `sigma`: sebaran/toleransi kategori (semakin besar, semakin lebar kurvanya)
- Semakin dekat nilai `x` ke `mean`, semakin tinggi nilai keanggotaannya (mendekati 1).
- `sigma` menentukan seberapa lebar toleransi kategori tersebut.

---

### 2. **Rule-Based Fuzzy Classification**
```python
setosa_score = gaussian_mf(...) * gaussian_mf(...) * ...
```
#### Tujuan Fungsi

Fungsi ini bertujuan untuk:

- Menghitung **derajat keanggotaan fuzzy** dari input (panjang dan lebar sepal & petal)
- Menentukan seberapa cocok input tersebut dengan masing-masing kelas bunga:
    - **Setosa**
    - **Versicolor**
    - **Virginica**

####  Komponen Utama

Setiap kelas memiliki **parameter Gaussian** yang mewakili karakteristik umum bunga tersebut.

#####  Setosa

```python
setosa_score = (
    gaussian_mf(sepal_length, 5.0, 0.4) *
    gaussian_mf(sepal_width, 3.4, 0.3) *
    gaussian_mf(petal_length, 1.5, 0.2) *
    gaussian_mf(petal_width, 0.2, 0.1)
)
```

- Nilai `mean` dan `sigma` di sini diambil dari **rata-rata dan sebaran data Setosa**.
- Semakin mirip input dengan ciri khas Setosa, semakin tinggi skornya.

#####  Versicolor

```python
versicolor_score = (
    gaussian_mf(sepal_length, 5.9, 0.5) *
    gaussian_mf(sepal_width, 2.8, 0.3) *
    gaussian_mf(petal_length, 4.2, 0.4) *
    gaussian_mf(petal_width, 1.3, 0.2)
)
```

######  Virginica

```python
virginica_score = (
    gaussian_mf(sepal_length, 6.5, 0.6) *
    gaussian_mf(sepal_width, 3.0, 0.4) *
    gaussian_mf(petal_length, 5.5, 0.5) *
    gaussian_mf(petal_width, 2.0, 0.3)
)
```

---

### 3. **Normalisasi dan Defuzzifikasi**
```python
# Normalisasi skor
total = setosa_score + versicolor_score + virginica_score
setosa_prob = setosa_score / total
versicolor_prob = versicolor_score / total
virginica_prob = virginica_score / total
```
Normalisasi skor fuzzy:
- Menghitung **jumlah total skor fuzzy** dari semua kelas.
- Setiap skor dibagi dengan total → hasilnya adalah **probabilitas relatif** (antara 0 dan 1)
- Probabilitas ini menunjukkan **seberapa dominan** masing-masing kelas terhadap input

---

### 4. **Klasifikasi Final**
```python
# Fungsi untuk mengklasifikasikan satu atau banyak sample sekaligus
def classify(*samples):
    for i, sample in enumerate(samples, 1):
        sepal_length, sepal_width, petal_length, petal_width = sample
        result, setosa_prob, versicolor_prob, virginica_prob = classify_iris(sepal_length, sepal_width, petal_length, petal_width)
        print(f"Sample {i}:")
        print(f"  Hasil klasifikasi: {result}")
        print(f"  Probabilitas: Setosa={setosa_prob:.2f}, Versicolor={versicolor_prob:.2f}, Virginica={virginica_prob:.2f}\n")
```

Fungsi ini memungkinkan:
- Mengklasifikasikan **satu atau banyak sample sekaligus**
- Menampilkan hasil klasifikasi dan probabilitasnya dengan format yang rapi

### Cara Kerja

#### 1. **Parameter `*samples`**

```python
def classify(*samples):
```

- Tanda `*` berarti bisa memasukkan **banyak tuple** sebagai argumen
- Contoh:

```python
classify(test1, test2, test3)
```

akan dianggap sebagai:

```python
samples = [(...), (...), (...)]
```

#### 2. **Loop dan Enumerasi**

```python
for i, sample in enumerate(samples, 1):
```

- `enumerate()` memberi nomor urut mulai dari 1
- `sample` adalah tuple berisi 4 nilai: sepal_length, sepal_width, petal_length, petal_width

#### 3. **Pemanggilan Fungsi Klasifikasi**

```python
result, setosa_prob, versicolor_prob, virginica_prob = classify_iris(...)
```

- Fungsi `classify_iris()` dipanggil untuk setiap sample
- Hasil klasifikasi dan probabilitas dikembalikan

#### 4. **Output Format**

```python
print(f"Sample {i}:")
print(f"  Hasil klasifikasi: {result}")
print(f"  Probabilitas: Setosa={setosa_prob:.2f}, Versicolor={versicolor_prob:.2f}, Virginica={virginica_prob:.2f}\n")
```

- Menampilkan hasil dengan dua angka di belakang koma
- Formatnya rapi dan cocok untuk laporan atau debugging

---

##  Contoh Output

Jika menjalankan:

```python
test1 = (5.1, 3.5, 1.4, 0.2)
test2 = (6.3, 3.3, 6.0, 2.5)
classify(test1, test2)
```

Output-nya bisa seperti:

```
Sample 1:
  Hasil klasifikasi: Setosa
  Probabilitas: Setosa=0.98, Versicolor=0.02, Virginica=0.00

Sample 2:
  Hasil klasifikasi: Virginica
  Probabilitas: Setosa=0.01, Versicolor=0.10, Virginica=0.89
```

---