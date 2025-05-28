# Finantza-aholkularitza aplikazioa (GrAL)  

Aplikazio honek **Flask** erabiltzen du Backend bezala.  
**RAG** sistema eta **agente adimendunak** erabiliz, finantza-aholkularitza arloko kontsultak onartzen ditu.  

<p align="center">
  <a href="https://skillicons.dev">
    <img src="https://skillicons.dev/icons?i=py,docker,git" />
  </a>
</p>

---

## 🚀 Beharrezko baldintzak  

Ziurtatu honakoak instalatuta dituzula:  

- **Python 3.10+** 
- **Docker eta Docker Compose** (instalazioa errazteko, hautazkoa)  

---

## ⚙️ Ingurune aldagaien konfigurazioa (.env)

Sortu `.env` fitxategia proiektuaren erroan eta gehitu honako aldagaiak:

```env
TOGETHER_API_KEY=
LLAMA_CLOUD_API_KEY=
MONGODB_URI=
```

---

## 🏗 Instalazioa eta exekuzioa  

### 🔹 Aukera 1: Docker bidez  

1. Irudia sortu:  
   ```sh
   docker-compose build
   ```

2. Aplikazioa abiatu:  
   ```sh
   docker-compose up -d
   ```

3. Aplikazioa gelditu:  
   ```sh
   docker-compose down
   ```

### 🔹 Aukera 2: Python inguruneen bidez

1. Ingurune birtuala sortu:  
   ```sh
   python3 -m venv venv
   ```

2. Ingurunea aktibatu:  
   Windows:
   ```sh
   .\venv\Scripts\activate
   ```
   Linux/Mac:
   ```sh
   source venv/bin/activate
   ```

3. Paketeak instalatu:  
   ```sh
   pip install -r requirements.txt
   ```

4. Aplikazioa exekutatu:  
   ```sh
   python app.py
   ```

---

## 📦 Funtzionalitateak

- Finantzei buruzko kontsultak
- Merkatuaren analisia
- ETF-en jarraipena
- Makroekonomia azterketa
- Finantza-berrien jarraipena

---

## 🌎 Erabilpena

1. Nabigatzailean sartu: http://localhost:5000
2. Erabili interfaze grafikoa finantza-kontsultak egiteko
3. Sisteman integratutako agente adimenduek zure kontsultak prozesatuko dituzte

---

## 🔧 Dependentziak

Proiektu honek honako pakete nagusiak erabiltzen ditu:

- Flask - Web framework-a
- Together AI - Hizkuntza ereduetarako
- Qdrant - Bektore biltegia
- MongoDB - Datu-basea
- yfinance - Finantza datuen lorpena
- GNews - Berrien bilaketa
- Gradio - Interfaze grafikoa

Dependentzia guztiak `requirements.txt` fitxategian daude zehaztuta.