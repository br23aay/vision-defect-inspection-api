FROM python:3.11-slim

WORKDIR /app

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY model/ model/
COPY db/ db/

# Create directory for SQLite database
RUN mkdir -p /data
ENV DB_PATH=/data/predictions.db

EXPOSE 5000

CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "--timeout", "120", "app:app"]
