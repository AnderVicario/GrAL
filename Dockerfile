# Erabili Python oinarrizko irudi bat
FROM python:3.10

# Edukiontzi barruko lan-direktorioa konfiguratu
WORKDIR /app

# Kopiatu behar diren fitxategiak
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Flask ataka erakutsi
EXPOSE 5000

# Aplikazioa exekutatzeko komandoa
CMD ["python", "app.py"]