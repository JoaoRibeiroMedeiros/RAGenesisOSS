# Use the official lightweight Python image.
FROM python:3.9-slim

# Set the working directory in the container.
WORKDIR /app

# Upgrade pip and setuptools.
RUN pip install --upgrade pip setuptools

# Copy the requirements file into the container.
COPY requirements_st.txt requirements.txt

# Install the Python dependencies.
RUN pip install -r requirements.txt

# Create the .streamlit directory
RUN mkdir -p /root/.streamlit

# Copy the Streamlit config file into the container
COPY config.toml /root/.streamlit/config.toml

# Copy the rest of the working directory contents into the container at /app.
COPY . .

# Expose the port that Streamlit uses.
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app_streamlit.py"]
