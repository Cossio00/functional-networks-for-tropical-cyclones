# Tropical Cyclone Functional Network Pipeline

Automated and parameterizable pipeline for the generation and analysis of climate functional networks associated with tropical cyclones using mean sea-level pressure (MSLP) data from the ERA5 reanalysis.

## Overview

This project provides an automated computational pipeline for generating and analyzing functional climate networks associated with tropical cyclones.

The pipeline processes ERA5 mean sea-level pressure (MSLP) data and integrates the main steps required for the analysis, including:

- ERA5 data acquisition through the Copernicus Climate Data Store API;
- Calculation of daily MSLP climatology;
- Calculation of MSLP anomalies;
- Land-sea mask application;
- Construction of sliding temporal windows;
- Functional network construction based on Kendall's rank correlation;
- Calculation of network metrics;
- Correction of spatial boundary effects using Spatially Embedded Random Networks (SERN);
- Generation of visualizations.

Any temporal window contain 10 consecutive days of 3-hourly observations, resulting in 80 time steps per window. Consecutive windows are shifted by one day, allowing the evolution of the network structure to be analyzed throughout the cyclone period. !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

The project was developed to facilitate reproducible and systematic climate-network analyses and to allow the same computational workflow to be applied to different cyclones, regions, and time periods.

### Methodological overview

The methodology uses a spatial resolution of `0.75° × 0.75°` and considers only oceanic grid points as network nodes.

For each temporal window, statistically significant Kendall correlations (p ≤ 0.05) are considered, and the upper 5% of significant positive correlations are used to construct the adjacency matrix.

### Project structure

The project is organized into modules according to their role in the processing pipeline:

| Module |	Description|
| --- | ---|
|`src/data`|	ERA5 and land-sea mask data acquisition|
|`src/processing`	|Data preprocessing and temporal-window generation
|`src/network`	|Functional-network construction and network metrics
|`src/correction`	|Boundary-effect correction using SERN
|`src/visualization`	|Generation of spatial network-metric plots
|`config`	|Cyclone and region configuration
|`Dataset/`	|Downloaded and processed input datasets
|`Metrics/`	|Intermediate and final numerical results
|`Plots/`	|Generated visualizations

### Requirements

The project requires Python 3 and the scientific Python ecosystem used by the processing pipeline.
The main dependencies include:

- NumPy
- SciPy
- pandas
- xarray
- netCDF4
- Matplotlib
- NetworkX
- joblib
- tqdm
- CDS API


Install the dependencies with:

`pip install numpy scipy pandas xarray netCDF4 matplotlib networkx joblib tqdm cdsapi` !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

The exact dependency list can be moved to a requirements.txt file for easier installation. !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


### ERA5 API configuration

The data acquisition modules use the Copernicus Climate Data Store (CDS) API.

Before using the automatic data acquisition commands, configure your CDS API credentials according to the official CDS API documentation.

The project expects the credentials to be available to the CDS API client.

Data acquisition can then be performed directly through the command-line interface.


### Configuration

The pipeline is parameterized through two dictionaries:
```
config/dictionary.py
config/dictionary_regions.py
```

This allows new cyclones and study regions to be added without changing the processing pipeline itself.

### Cyclone configuration

Cyclones are defined in `config/dictionary.py.`

For example:
```
CYCLONES = {

    "Gaja": {
        "start": "2018-10-10 00:00",
        "end": "2018-12-22 21:00",
        "folder": "Gaja",
        "region": "Bengal_Bay"
    },

    "Irma": {
        "start": "2017-07-30 00:00",
        "end": "2017-10-12 21:00",
        "folder": "Irma",
        "region": "North_Atlantic_Ocean"
    }
}
```

Each cyclone requires:

|Parameter|	Description|
|---|---|
|`start`|	Beginning of the period processed by the pipeline|
|`end`	|End of the period processed by the pipeline|
|`folder`|	Output folder associated with the cyclone|
|`region`|	Region in which the cyclone is analyzed|

The `start` and `end` values define the complete temporal interval from which the sliding windows are generated.

For example, with a 10-day window and a one-day displacement:

```
Window 1 → Day 1 – Day 10
Window 2 → Day 2 – Day 11
Window 3 → Day 3 – Day 12
...
```

This produces overlapping temporal windows that allow the evolution of the network to be followed throughout the analyzed period.

### Region configuration

Study regions are defined in:

`config/dictionary_regions.py`

For example:

```
REGIONS = {

    "Bengal_Bay": {
        "area": [34.5, 49.5, 4.5, 100],
        "folder": "Bengal_Bay",
    },

    "North_Atlantic_Ocean": {
        "area": [42, -100, 10, -42],
        "folder": "North_Atlantic_Ocean",
    }

}
```
The area parameter defines the geographic extent of the region:

`[north, west, south, east]`

For example:

`"area": [34.5, 49.5, 4.5, 100]`

corresponds to:
```
North latitude:  34.5°
West longitude:  49.5°
South latitude:   4.5°
East longitude: 100°
```
The region associated with each cyclone is defined through the `region` field in `CYCLONES`.

### Usage

The main entry point of the project is:

`python main.py`

Running the command without arguments displays the available commands and usage examples.

### List available cyclones

To list all configured cyclones:

`python main.py cyclones`

The command displays the cyclone name, associated region, and configured processing period. This is useful for checking which events are currently configured in the project.

### Running the complete pipeline

Once the required ERA5 data are available, the complete processing workflow can be executed with:

`python main.py processing Gaja`

The pipeline automatically determines the region associated with the selected cyclone. The user does not need to execute each processing module individually.

### Downloading ERA5 MSLP data

To request the ERA5 MSLP data associated with a configured cyclone:

`python main.py mslp Gaja`

The pipeline uses the region associated with Gaja to determine the spatial domain of the request. The data are downloaded and stored under the corresponding region directory.

### Downloading the land-sea mask

The land-sea mask can be obtained with:

`python main.py landsea Gaja`

The region is automatically obtained from the cyclone configuration.

## Processing stages
### 1. Daily climatology

The pipeline calculates a daily MSLP climatology for the period 1979–2018.

For each grid point and calendar day, the eight 3-hourly observations available for each day are considered. The resulting climatology contains 365 daily spatial fields.

The climatology is calculated before applying the land-sea mask.

### 2. MSLP anomalies

MSLP anomalies are calculated as:

`Anomaly = Observed MSLP − Climatological MSLP`

This removes the seasonal component from the time series and provides the anomaly fields used in subsequent network construction.

### 3. Land-sea mask

The land-sea mask removes continental grid points from the analysis.

Only oceanic grid points are retained as network nodes.

### 4. Sliding temporal windows

The anomaly time series are divided into overlapping 10-day windows.

Each window contains:

`10 days × 8 observations/day = 80 time steps`

The displacement between consecutive windows is one day:
```
Window 1 → Days 1–10
Window 2 → Days 2–11
Window 3 → Days 3–12
...
```
This produces an evolutionary sequence of functional networks.

## Functional network construction

For each temporal window, the pipeline calculates Kendall's rank correlation coefficient between the anomaly time series associated with every pair of oceanic grid points. Only statistically significant correlations with: `p ≤ 0.05` are retained.

The adjacency matrix is then constructed using the upper 5% of significant positive Kendall correlation values.

Therefore: `Aij = 1` when a connection is established between nodes i and j, and: `Aij = 0` otherwise.

## Network metrics

Three network metrics are calculated for each temporal window.

### Degree

The degree of a node corresponds to the number of connections associated with that node. Higher values indicate nodes with a larger number of statistical connections in the functional network.

### Mean geographical distance

For each node, the mean geographical distance of its connections is calculated using the Haversine distance. The metric represents the average spatial extent of the connections associated with a node.

### Local clustering coefficient

The local clustering coefficient measures the degree of interconnection between the neighbors of a node. It characterizes the local organization of the functional network and the formation of triangular structures between connected nodes.

## Boundary-effect correction

Spatially embedded networks can be affected by artificial boundaries introduced by the study domain and by the land-sea mask.

Nodes near boundaries may have fewer possible connections simply because nodes outside the selected domain do not exist in the network.

To reduce this effect, the pipeline uses Spatially Embedded Random Networks (SERN).

For each temporal window, 1000 surrogate networks are generated while preserving the geographic positions of the nodes and the observed relationship between connection probability and geographical distance. The same three network metrics are calculated for the surrogate networks.

The corrected metric is calculated as:

```math
M_{\mathrm{corr}} = \frac{M_{\mathrm{obs}}} {\langle M_{\mathrm{SERN}}\rangle}
```
where:

$$\M_{\mathrm{obs}}\$$ is the metric calculated from the original functional network;
$$\(\langle M_{\mathrm{SERN}}\rangle\)$$ is the mean metric obtained from the 1000 surrogate networks.

The correction is applied independently to degree, mean geographical distance, and clustering coefficient.

## Reproducibility

The project is designed around parameterized cyclone and region dictionaries.

To analyze a new cyclone, the main configuration requirements are:

1. Add the cyclone to `config/dictionary.py`;
2. Associate it with an existing region or create a new region;
3. Provide the corresponding `start` and `end` dates;
4. Run the data acquisition commands;
5. Execute the processing pipeline.

For example:

```
CYCLONES = {
    "NewCyclone": {
        "start": "YYYY-MM-DD HH:MM",
        "end": "YYYY-MM-DD HH:MM",
        "folder": "NewCyclone",
        "region": "Bengal_Bay"
    }
}
```
If a new geographic domain is required, add it to `dictionary_regions.py`:
```
REGIONS = {
    "NewRegion": {
        "area": [North, West, South, East],
        "folder": "NewRegion"
    }
}
```

The remaining processing stages can then be executed through the same command:

`python main.py processing NewCyclone`

This parameterization is one of the main characteristics of the project, allowing the same workflow to be applied to different events and geographic domains.

## Command reference
|Command|	Description|
|---|---|
|`python main.py`	| Display command-line help|
|`python main.py cyclones`	| List configured cyclones|
|`python main.py mslp <cyclone>`|	Download ERA5 MSLP data|
|`python main.py landsea <cyclone>`|	Download the land-sea mask|
|`python main.py processing <cyclone>`|	Execute the complete processing pipeline|