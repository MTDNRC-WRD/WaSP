Self-Potential Processing Module
Python tools for processing, filtering, and plotting self-potential (SP), drift, and temperature/conductivity survey data from CSV inputs configured through a TOML file.
google.github

Overview
This repository contains three Python scripts plus a config.toml file that work together as a small processing pipeline for hydrogeophysical survey data.
google.github
 The codebase follows a Google-style approach to Python documentation, with module and function docstrings, explicit configuration, and command-line execution for plotting outputs.
google.github
+1

The workflow is:

Read project settings and file paths from config.toml.

Process raw self-potential, drift, and temperature/conductivity data into cleaned CSV outputs.

Generate standard figures from the processed CSV files.
peps.python
+1

Repository Structure

text
.
├── config.toml
├── process_code.py
├── plot_outputs.py
├── G.py
└── test_data/
Suggested purpose for each file:

File	Purpose
GNSS_Logger.py	Logs GNSS/NMEA output from an Arrow receiver to CSV for field data capture.
data_processing.py	Loads survey data, applies filtering and corrections, and writes processed output tables.
plotting.py	Reads processed output tables and writes standard PNG figures.
config.toml	Defines input/output paths, file names, and processing parameters.

Configuration
The project uses a TOML configuration file with paths, files, and processing sections.
google.github
 A typical example looks like this:


text
[paths]
input_dir = "test_data"
processed_dir = "test_data/processed"
figures_dir = "test_data/figures"

[files]
sp_data = "Self_Potential_Data_Rio_Grande.csv"
drift_data = "Self_Potential_Electrode_Drift_Data_Rio_Grande.csv"
temp_cond_data = "Temperature_Conductivity_Data_Rio_Grande.csv"
hfem_resistivity = "HFEM_resistivity.csv"

[processing]
dipole_length_m = 0.5588
gaussian_sigma = 30
boxcar_m = 5
boxcar_iterations = 1
Config fields
Section	Key	Description
paths	input_dir	Directory containing raw input CSV files.
paths	processed_dir	Directory where processed CSV outputs are written.
paths	figures_dir	Directory where generated figures are saved.
files	sp_data	Raw self-potential survey data file.
files	drift_data	Electrode drift data file.
files	temp_cond_data	Temperature and conductivity survey data file.
files	hfem_resistivity	Reserved HFEM resistivity file; currently wired in but not used.
processing	dipole_length_m	Dipole length used when integrating SP gradient to electric potential.
processing	gaussian_sigma	Gaussian filter half-width parameter in samples.
processing	boxcar_m	Half-width of the boxcar window in samples.
processing	boxcar_iterations	Number of boxcar filter passes.
Installation
Use Python 3.11 or newer so tomllib is available in the standard library.
peps.python
 Install the required packages before running the scripts:


Limitations
The processing code expects specific CSV column positions rather than named schema validation.

Segment handling is hard-coded for IDs 1 through 4.

Some helper inputs, such as hfem_resistivity, are configured but not yet used.

Plotting assumes the processed output files already exist and contain the expected columns.

Requirements: 
numpy
pandas
matplotlib
pyserial
pynmea2
