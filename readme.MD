# Finantza-aholkularitza aplikazioa (GrAL)  

Aplikazio honek **Flask** erabiltzen du BackEnd bezala.  
**RAG** eta **agente adimentsuak** erabiliz, finantza-aholkularitza arloko kontsultak onartzen ditu.  

---

## 🚀 Beharrezko baldintzak  

Ziurtatu honakoak instalatuta dituzula:  

- **Python 3.10+** 
- **Docker eta Docker Compose** (instalazioa errazteko, hautazkoa)  

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

## 🌎 Erabilpena

    Sartu nabigatzailean: http://localhost:5000
