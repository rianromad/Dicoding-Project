# Laporan Proyek Machine Learning - Subkhan Rian Romadhon

<img src="https://cdn.corporate.walmart.com/dims4/WMT/0b04aa6/2147483647/strip/true/crop/2400x1260+0+0/resize/1200x630!/quality/90/?url=https%3A%2F%2Fcdn.corporate.walmart.com%2F6f%2Fd3%2Ff3f5a16f44a88d88b8059defd0a9%2Foption-signage.jpg" width=500>
<br><br>

**Contents:**
**1. [Project Domain](#project-domain)**
**2. [Business Understanding](#business-understanding)**
**3. [Data Preparation](#data-preparation)**
**4. [Modeling](#modeling)**
**5. [Evaluation](#evaluation)**

## Project Domain

Perusahaan retail merupakan perusahaan yang menyimpan dan melakukan proses jual beli barang jadi kepada konsumen untuk mendapatkan profit. Kebutuhan harian, pakaian hingga barang elektronik disediakan oleh perusahaan retail demi memenuhi keinginan dan permintaan konsumen. Seiring berjalannya waktu, permintaan maupun persepsi konsumen dapat berubah-ubah sehingga mempengaruhi tingkat penjualan. 

Dengan berkembangnya teknologi saat ini, machine learning dapat dimanfaatkan untuk memprediksi penjualan suatu perusahaan retail. Machine learning merupakan cabang dari kecerdasan buatan yang mempelajari bagaimana suatu mesin dapat meniru perilaku manusia menggunakan data sebagai bahan pelatihan. Pada tahun 2008 pun model machine learning sudah mulai memanfaatkan jaringan saraf tiruan yang lebih kompleks dibandingkan model tradisional. Misalnya, penelitian dari Au *et al.* (2008) mengembangkan algoritma *Evolutionary Neural Network* (ENN) untuk memprediksi penjualan di toko pakaian. Algoritma tersebut menggabungkan jaringan saraf tiruan dengan algoritma genetik untuk memilih parameter yang sesuai sehingga bisa mendapatkan akurasi yang optimal. Dari hasil penelitiannya model dengan algoritma ENN mampu mengalahkan model dengan algoritma *Seasonal Autoregressive Integrated Moving Average* (SARIMA) ketika data memiliki koefisien variasi permintaan harian yang rendah serta memiliki trend dan seasonal yang tidak tetap.

## Business Understanding
#### 1. Problem Statements
Walmart Inc merupakan perusahaan retail multinasional Amerika yang mengoperasikan bisnis hypermarket, toko serba diskon, serta toko kebutuhan sehari-hari. Perusahaan ini memiliki kantor pusat di daerah Bentonville, Arkansas dan memiliki banyak toko yang tersebar di seluruh belahan dunia. Dikarenakan permintaan konsumen yang sering berubah-ubah, perusahaan bisa saja mengalami *stockout* karena tidak memenuhi permintaan maupun *overstock* karena *overestimate* terhadap permintaan konsumen. Bagaimana supaya perusahaan bisa melakukan estimasi total penjualan mingguan yang optimal berdasarkan data penjualan sebelumnya. 

#### 2. Goals
Dengan mengetahui perkiraan penjualan mingguan, perusahaan diharapkan dapat mengantisipasi adanya fluktuasi permintaan dengan melakukan strategi pengadaan yang tepat sehingga bisa mengoptimalkan profit yang didapatkan serta mengurangi risiko *stockout* maupun *overstock*.

#### 3. Solutions
Solusi yang ditawarkan yakni membuat model prediksi menggunakan algoritma yang cocok untuk data time series seperti Moving Average, Holt Winter, ARIMA, SARIMA dan lain-lain. Menurut Beheshti-Kashi *et al*. (2014), aspek penting dalam memilih model prediksi adalah menemukan model yang akurat, juga mudah diaplikasikan dalam kehidupan sehari-hari. Algoritma yang berbasis jaringan saraf tiruan maupun model hybrid bisa memberikan hasil yang lebih akurat jika dibandingkan dengan model tradisional. Akan tetapi, hal tersebut juga berdampak pada sulitnya memahami bagaimana model prediksi bekerja, khususnya bagi perusahaan yang ingin mengaplikasikan model tersebut. Tidak masalah menggunakan model yang lebih sederhana, asalkan memiliki skor performa yang sesuai kriteria.

Selain itu, dalam memilih algoritma harus disesuaikan dengan karakteristik data time series, misalnya apakah terdapat seasonality maupun tren pada data. Hal tersebut akan dibahas pada bagian **[modeling](#modeling)**. Model terbaik dipilih berdasarkan metrik Mean Absolute Percentage Error (MAPE) terkecil. Metrik tersebut dipilih karena memiliki skala dari 1 sampai 100 persen sehingga mudah untuk diinterpretasikan. Metrik seperti Mean Absolute Error (MSE) dan Mean Squared Error (MSE) juga digunakan untuk melihat apakah model yang paling baik memiliki tiga skor yang sama-sama memiliki nilai paling rendah atau tidak. 

## Data Understanding
Tahapan ini dilakukan untuk memahami data yang akan digunakan dalam pemodelan, meliputi:

- Melakukan penggabungan data (jika diperlukan)
- Mengetahui tipe data yang terdapat pada masing-masing atribut
- Mengecek *missing value*
- Deskripsi statistik

Terdapat data historis 45 toko Walmart dari beberapa daerah. Setiap toko memiliki beberapa departemen. Atribut harga bahan bakar dalam seminggu, hari libur, dan kondisi temperatur juga ditampilkan di data ini. Walmart juga menjalankan strategi markdown (diskon) sepanjang tahun. Markdown tersebut dilakukan sebelum hari libur besar seperti Super Bowl, hari Buruh, Thanksgiving, dan Natal.

Data yang digunakan bersumber dari <a href="https://www.kaggle.com/datasets/divyajeetthakur/walmart-sales-prediction">link dataset</a>, berjumlah 421,570 record penjualan mingguan mulai tanggal 05 Februari 2010 hingga 26 Oktober 2012 (143 minggu). Terdapat 16 atribut, diantaranya:

- Store = nomor toko
- Size = ukuran toko
- Dept = departemen toko
- Date = menunjukkan waktu mingguan (hari jumat dalam seminggu)
- Temperatur = menunjukkan rata-rata temperatur di daerah tersebut dalam satuan Fahrenheit
- FuelPrice = biaya bahan bakar di daerah tersebut
- Markdown1-5 = data yang berkaitan dengan kegiatan promosi yang dilakukan perusahaan. Hanya tersedia setelah bulan November 2011 dan tidak tersedia di semua toko pada waktu yang sama. Nilai yang kosong ditandai dengan tulisan *Null*
- CPI = indeks harga konsumen
- Unemployment = tingkat pengangguran
- IsHoliday = menunjukkan apakah tanggal tersebut merupakan hari libur spesial
- Type = jenis toko (A, B, C)
- Weekly Sales = penjualan mingguan

Berikut merupakan atribut beserta tipe data yang diperoleh melalui penggabungan data dari berbagai file. Terlihat bahwa masih terdapat missing value dan beberapa atribut yang masih berupa kategorik. Atribut markdown menunjukkan promosi yang dilakukan oleh perusahaan dan hanya tersedia setelah bulan November 2011 serta tidak tersedia di seluruh stores. Diasumsikan bahwa adanya nilai null menandakan bahwa markdown tidak dilakukan sehingga kita akan menggantinya dengan nilai 0 pada tahapan [data preparation](#data-preparation).

<img src="https://user-images.githubusercontent.com/61647791/163713353-65a5aac4-ce1c-44fd-9f97-34208b1d6656.png">

<img src="https://user-images.githubusercontent.com/61647791/163713375-80812c8f-8de5-497d-b8a5-41866f65f2fc.png">

Selanjutnya kita juga dapat mengetahui deskripsi statistik dari data tersebut yang ditunjukkan pada gambar di bawah ini:

<img src="https://user-images.githubusercontent.com/61647791/163713501-5e86d776-d443-4ad6-a823-138be207f103.png">

Melalui statistik deskriptif kita dapat mengetahui gambaran kasar dari data numerikal, baik dari rata-rata, median, kuartil maupun nilai minimum dan maksimum. Misalnya pada atribut penjualan mingguan (weekly sales) yang akan diprediksi. Atribut ini memiliki rata-rata sebesar 15,981.26 dengan nilai minimum -4,988.940 dan maksimum 693,099.36. Artinya di toko tersebut selain mendapatkan revenue juga mengalami kerugian yang ditandai dengan nilai yang negatif. 

Selain dengan statistik deskriptif, visualisasi juga dapat dilakukan untuk memahami karakter data dengan lebih mudah. Di bawah ini merupakan sebaran penjualan mingguan yang digambarkan menggunakan histogram. Terlihat bahwa grafik lebih condong ke kiri, menunjukkan bahwa penjualan mingguan dari setiap toko kebanyakan berada pada interval yang lebih rendah. Selain itu, toko yang memiliki penjualan mingguan yang tinggi jumlahnya terbilang cukup rendah. 

<img src=https://user-images.githubusercontent.com/61647791/163713567-fcce42f6-e663-4834-9f13-a8f9404a16c8.png>

## Data Preparation
Tahapan ini dilakukan untuk mempersiapkan data sebelum nantinya digunakan dalam pemodelan. Tahap persiapan data meliputi:
- Mengatasi *missing value*
- Transformasi data kategorikal
- Pemilihan atribut

Pada model prediksi seperti regresi dan peramalan, semua atribut yang digunakan harus bertipe numerikal sehingga atribut yang bersifat kategorikal harus diubah terlebih dahulu. Atribut IsHoliday masih bersifat kategorikal (True False) sehingga akan diubah menjadi bernilai 1 = True dan 0 = False. Selain atribut IsHoliday, atribut Type juga perlu diubah menjadi bernilai numerik melalui one hot encoding.

<img src="https://user-images.githubusercontent.com/61647791/163713642-e401a841-2881-47bd-80e9-360f55cf488e.png">

Umumnya dalam membangun model prediksi (peramalan) menggunakan algoritma seperti Moving Average (MA) maupun SARIMA, kita hanya memerlukan atribut tanggal dan targetnya saja. Sedangkan apabila ingin membangun model prediksi (regresi) atribut lainnya bisa dimasukkan ke dalam model. Tujuan dari permasalahan ini adalah memprediksi total penjualan mingguan, sehingga dilakukan grouping data berdasarkan tanggal.

<img src="https://user-images.githubusercontent.com/61647791/163713679-3349dec3-36da-4278-8da5-5c345cdb238b.png">

## Modeling

Dikarenakan dataset dikumpulkan menurut urutan waktu (time series) serta nilai yang akan diprediksi berupa numerik maka model yang layak untuk diimplementasikan pada kasus di atas adalah model regresi maupun model peramalan seperti Moving Average (MA) maupun Seasonal Autoregressive Moving Average (SARIMA). Untuk menentukan model apa yang layak digunakan pada data ini, dilakukan proses dekomposisi untuk melihat komponen time series. Komponen tersebut diantaranya:

- Level = rata-rata nilai pada sebuah data time series
- Trend = pola yang menunjukkan kecenderungan naik atau turunnya sebuah nilai pada data time series
- Seasonality = merupakan pola yang berulang dengan teratur
- Noise = variasi random pada data time series

Berikut merupakan illustrasi dari keempat komponen time series di atas:

<img src="https://www.researchgate.net/profile/Joanna-Michalowska-2/publication/330169483/figure/fig2/AS:734124910329857@1552040426538/General-form-of-time-series-and-its-components-in-time-series-components-random.ppm">

<img src="https://user-images.githubusercontent.com/61647791/163713731-c84a2b71-ef36-4edb-a9f6-cb10a153b948.png">

Berdasarkan plot time series, tampak bahwa total penjualan mingguan memiliki pola musiman yang berulang setiap tahun dengan trend yang meningkat. Di bawah ini merupakan panduan dalam memilih model time series berdasarkan Hanke & Wichern (2014). Model yang akan saya gunakan yakni SARIMA (Box-Jenkins), serta algoritma peramalan dari Meta bernama Prophet.

<img src="https://user-images.githubusercontent.com/61647791/163517008-d66de14a-79bb-4dc1-9cbb-4cff393d40a7.png" >

#### 1. FB Prophet

Berdasarkan situs prophet, FB Prophet merupakan algoritma peramalan yang dikembangkan oleh Meta berbasis model aditif dengan fungsi trend non-linear yang menyesuaikan musiman tahunan, mingguan, dan harian, ditambah dengan pengaruh hari libur. Prophet sangat canggih dalam menangani missing data dan perubahan trend serta dapat menangani outlier. Di bawah ini merupakan model FB Prophet berdasarkan persamaan regresi aditif menurut Kumar Jha & Pande (2021):

<img src="https://user-images.githubusercontent.com/61647791/163697962-9cadbf22-46d5-474d-9bcd-79dab8ed9033.png" >

Di mana:

- y(t) = model regresi aditif
- g(t) = komponen tren
- h(t) = komponen hari libur
- s(t) = komponen musiman

Komponen trend g(t) dapat dimodelkan dengan dua cara, yakni:

- Logistic Growth Model

    Model merepresentasikan kenaikan secara pelan-pelan kemudian menuju pada kondisi jenuh. Berikut merupakan persamaan dari model tersebut:
    
    <img src="https://user-images.githubusercontent.com/61647791/163698182-2c23cc94-7a78-47a4-bda6-b6a66a3dfb91.png" >
   
- Piece-wise Linear Model
    Merupakan modifikasi dari model linear, di mana setiap rentang dari nilai x memiliki hubungan linear yang berbeda. Diformulasikan dengan persamaan sebagai berikut:
<img src="https://user-images.githubusercontent.com/61647791/163698252-54647996-238c-45f7-b71e-de85a51f3240.png" >

Untuk menggunakan model FB Prophet, nama atribut tanggal dan target perlu diubah terlebih dahulu menjadi "ds" untuk tanggal dan "y" untuk target. 

<img src="https://user-images.githubusercontent.com/61647791/163713788-cb67b63e-2280-44cd-b9a3-a933ddd3507b.png">

#### 2. Seasonal Autoregressive Integrated Moving Average (SARIMA)
Seasonal ARIMA merupakan pengembangan dari model ARIMA (Box-Jenkins) yang dikhususkan untuk menangani data time series yang memiliki pola musiman. Pada model SARIMA terdapat, enam buah komponen yang digunakan dan biasanya dinotasikan dalam bentuk ARIMA(p,d,q)(P,D,Q). 
Keenam komponen tersebut dapat kita ketahui melalui plot Autocorrelation Function (ACF) dan Partial Autocorrelation Function (PACF). Berikut adalah penjelasan keenam komponen:
- p = orde model AR
- d = pembeda non musiman
- q = orde model MA
- P = orde model AR musiman
- D = pembeda musiman
- Q = orde model MA musiman



Di bawah ini merupakan persamaan yang digunakan untuk membuat model peramalan menggunakan SARIMA:

<img src="https://user-images.githubusercontent.com/61647791/163704674-5b5d4d78-5f79-47e0-8561-fefaf40af6be.png" >

Kita akan menentukan keenam komponen melalui serangkaian eksplorasi data time series menggunakan plot ACF dan PACF.

<img src="https://user-images.githubusercontent.com/61647791/163713834-9b8c8f3e-7f56-4d77-883a-a24b3fa2ece9.png">

Terlihat pada grafik time series masih terdapat pola musiman, sehingga perlu dilakukan proses pembedaan musiman (seasonal differencing) untuk menghilangkan pola musiman sehingga bentuknya menjadi lebih stasioner. Kemudian pada plot acf, terdapat 3 buah lag (lag 1,2,5) yang melebihi batas signifikan sehingga untuk parameter q dan Q bernilai 3. Pada plot pacf, terdapat dua buah lag (lag 1, 5) yang melebihi batas signifikan sehingga untuk parameter p, dan P bernilai 2.

<img src="https://user-images.githubusercontent.com/61647791/163713870-1cd482bb-f946-47a3-a3e5-44dfceeab3ea.png">

Dari grafik time series, pola musiman sudah menghilang. Akan tetapi masih terdapat pola trend, sehingga perlu dilakukan differencing untuk menghilangkan pola tren.

<img src="https://user-images.githubusercontent.com/61647791/163713912-0d82e034-9fda-4402-8f83-dc04141480f9.png">

Berdasarkan grafik time series terlihat bahwa data sudah memiliki pola stasioner serta tidak berpola tren maupun musiman. Oleh karena itu, ketika akan menggunakan model SARIMA maka parameter pembeda non musiman (d), dan pembeda musiman (D) bernilai 1. 

<img src="https://user-images.githubusercontent.com/61647791/163713962-86b4d9d4-a0bb-4835-89a1-535adaba1914.png">

<img src="https://user-images.githubusercontent.com/61647791/163714009-382b2469-b4b6-4b0c-9c94-f072b10aba00.png">

Dari model summary maupun grafik residu, kita bisa mendapatkan beberapa informasi terkait kelayakan model:

- Berdasarkan uji statistik Ljung-Box nilai P-value kurang dari 0.05 sehingga menolak hipotesa awal. Artinya tidak terdapat korelasi serial pada model sehingga dapat digunakan untuk memprediksi total penjualan mingguan.
- Hal tersebut juga dibuktikan melalui plot ACF residu model bahwa semua lag cenderung berada dalam batas signifikan.

## Evaluation
Untuk memastikan bahwa model memiliki performa yang baik dalam memprediksi total penjualan mingguan, diperlukan sebuah metrik penilaian. Mean Absolute Percentage Error (MAPE) merupakan salah satu metrik yang biasa digunakan untuk mengevaluasi performa model regresi. Skor MAPE memiliki rentang 1 sampai 100 persen sehingga lebih mudah diinterpretasikan dibandingkan metrik seperti Mean Squared Error (MSE) maupun Mean Absolute Error (MAE). Model yang paling baik dipilih berdasarkan MAPE terkecil. Dibawah ini merupakan formula MAPE beserta kriteria skor berdasarkan Lewis (1982). 

**Rumus MAPE**

<img src="https://1.bp.blogspot.com/-0995oy9kwNo/Xfh7nJ9tL_I/AAAAAAAABbY/yMXNlUJqzRw5fqWTyYZYGsk9SOpEIWMCwCLcBGAsYHQ/s1600/MAAAA.JPG" width=300>

Di mana:
- y = nilai aktual
- $\hat{y}$ = nilai hasil prediksi

**Kriteria Skor MAPE**

<br>
<img src="https://www.researchgate.net/profile/Chao-Hung-Wang-3/publication/27219891/figure/tbl1/AS:394224022376454@1471001739320/MAPE-CRITERIA-FOR-MODEL-EVALUATION.png" width=400 >

<br>

Selain menggunakan MAPE sebagai metrik untuk mengevaluasi model peramalan, saya juga menggunakan metrik Mean Absolute Error (MAE) dan Mean Squared Error untuk menilai konsistensi model peramalan terbaik. Berikut merupakan formula untuk menghitung skor MAE dan MSE:

**Rumus MAE:**

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRaF1XcwUzJt3314Fn0dVCK7HLZKgDcrHOO5sSuWcjrJSJEdzlZEPO5qsqJG8xbKMeRUoo&usqp=CAU" width=200>

**Rumus MSE:**

<img src="https://i0.wp.com/statisticsbyjim.com/wp-content/uploads/2021/11/mse_formula.png?resize=246%2C78&ssl=1g" width=150>
<br><br>

**Plot Prediksi menggunakan FB Prophet**

<img src="https://user-images.githubusercontent.com/61647791/163714416-891e77c0-72e6-4fc2-b0d4-d4e94112d6a3.png">

**Plot Prediksi menggunakan FB Prophet**

<img src="https://user-images.githubusercontent.com/61647791/163714365-8949cb8f-1827-48c7-bd14-418e40c7807f.png">

Di bawah ini merupakan rekap skor performa dari kedua model peramalan.

<img src="https://user-images.githubusercontent.com/61647791/163714072-feac24b8-9d1b-4479-8cdb-f3e309cba0eb.png" width=400>

Berdasarkan tabel di atas, kedua model memiliki skor MAPE dibawah 10% sehingga akurat dalam memprediksi total penjualan mingguan. Akan tetapi, model yang dibangun menggunakan SARIMA memberikan skor MAE, MSE dan MAPE yang lebih rendah jika dibandingkan dengan model yang dibuat menggunakan FB Prophet. Oleh karena itu, SARIMA dapat digunakan untuk memprediksi total penjualan mingguan perusahaan Walmart. Di bawah ini merupakan grafik prediksi total penjualan mingguan perusahaan Walmart selama 26 minggu ke depan.

<img src="https://user-images.githubusercontent.com/61647791/163714382-116c3c23-a275-4ecd-a367-eabfd91758ba.png">

**Daftar Pustaka:**

1. Paper

    - Au, K., Choi, T. and Yu, Y. (2008), "Fashion retail forecasting by evolutionary neural networks", International Journal of Production Economics, Vol. 114 No. 2, pp. 615-630.
    
    - Beheshti-Kashi, S., Karimi, H., Thoben, K., Lütjen, M. and Teucke, M. (2014), "A survey on retail sales forecasting and prediction in fashion markets", Systems Science &amp; Control Engineering, Vol. 3 No. 1, pp. 154-161.

    - Kumar Jha, B. and Pande, S. (2021), "Time Series Forecasting Model for Supermarket Sales using FB-Prophet", 2021 5th International Conference on Computing Methodologies and Communication (ICCMC), doi:10.1109/iccmc51019.2021.9418033.
    
2. Buku

    - Hanke, J. and Wichern, D. (2014), Business forecasting, Pearson, Harlow, 9thed.
    
    - Lewis, C. (1982). Industrial and business forecasting methods. London: Butterworth Scientific
    
3. Situs web

    https://facebook.github.io/prophet/

4. Dataset

    https://www.kaggle.com/datasets/divyajeetthakur/walmart-sales-prediction