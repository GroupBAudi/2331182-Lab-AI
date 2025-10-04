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
- Semakin dekat nilai `x` ke `mean`, semakin tinggi nilai keanggotaannya (mendekati 1).
- `sigma` menentukan seberapa lebar toleransi kategori tersebut.

---

### 2. **Rule-Based Fuzzy Classification**
```python
setosa_score = gaussian_mf(...) * gaussian_mf(...) * ...
```
- Kamu membuat **tiga aturan fuzzy**:
  - Rule 1 → Setosa
  - Rule 2 → Versicolor
  - Rule 3 → Virginica
- Setiap rule menggabungkan empat fitur (sepal & petal length/width) dan menghitung skor total untuk masing-masing kelas.

---

### 3. **Normalisasi dan Defuzzifikasi**
```python
total = setosa_score + versicolor_score + virginica_score
setosa_prob = setosa_score / total
```
- Skor dari tiap rule dijumlahkan → lalu dibagi untuk dapatkan **probabilitas relatif**
- Ini adalah proses **defuzzifikasi** → mengubah nilai fuzzy menjadi keputusan final

---

### 4. **Klasifikasi Final**
```python
if max(...) == setosa_prob:
    result = "Setosa"
```
- Sistem memilih kelas dengan probabilitas tertinggi
- Output berupa nama bunga dan probabilitas tiap kelas

---

##  Contoh Output

Untuk `test = (5.1, 3.5, 1.4, 0.2)`  
Output-nya akan seperti ini:
```
Sample 1:
  Hasil klasifikasi: Setosa
  Probabilitas: Setosa=0.99, Versicolor=0.01, Virginica=0.00
```




