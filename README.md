# The codes and models for the paper: 
*"Water clarity mapping of global lakes using a novel hybrid deep-learning-based recurrent model with Landsat OLI images"*

## **Citation**: 

Y. He, Z. Lu, W. J. Wang, D. Zhang, Y. L. Zhang, B. Q. Qin, K. Shi*, & X. F. Yang. 2022. Water clarity mapping of global lakes using a novel hybrid deep-learning-based recurrent model with Landsat OLI images. *Water Research*, 215, 118241.
<https://doi.org/10.1016/j.watres.2022.118241>

*Notes:* This web only provides the codes used in this ciation. The in-situ measurements may open access in the future.

## Data pre-processing (almost working in *GEE* platform)
### 1. The global water bodies extraction:
This code extracted the pixels with water occurrence ≥ 25% based on the **JRC Global Surface Water Mapping Layers, v1.3**, which is provided by GEE. Detailed information was described in the above-mentioned citation.

The online accessed link is shown below:

<https://code.earthengine.goo-gle.com/?scriptPath=users%2Fheyuan9874%2Fextract_inland_water%3Aextract_inland_water1>

### 2. The surface reflectance of Landsat 8 OLI extraction:
This code extracted the lake-average of surface reflectance for B1, B2, B3, B4, B5, and B7 of Landsat 8 OLI based on global water bodies and lake boundary (provided by *HydroLAKES* [<https://www.hydrosheds.org/products/hydrolakes>]). 
Detailed information was described in the above-mentioned citation.

The online accessed link is shown below:

<https://code.earthengine.google.com/529cda3218497357af0799cda42e4d39>

## Deep-learning-based models (working in *Python* platform)
### 1. DRGN_SDD_Landsat8 (also upload to releases)
This folder save the Deep Gated Recurrent Network (DGRN) model, which input with normalized surface reflectance and output the log-transformed SDD. DGRN model running with Keras 2.6.0. The model architecture and parameterization are shown in the citation.

## Data post-processing
### 1. draw_temporal_trend.py
The *draw_temporal_trend.py* is used to calculate the lake-average of SDD from surface reflectance based on DGRN model and extract the global lakes without ice-covered regions. 

### 2. Fig1.drawio & Graphical_Abstract.drawio & FigS2.drawio

The *Fig1.drawio*, *Graphical_Abstract.drawio*, and *FigS2.mxd* are running in Drawio software (also work in VScode), which are used to draw the Fig. 1 (model architecture), the graphical abstract, and the working flow of Landsat imagery in the citation.

### 3. optical_properties.py
The *optical_properties.py* is used to describe the optical properties distribution of surface reflectance, as shown as Fig. 2.

### 4. sdd_DGRN.py
The *sdd_DGRN.py* is used to describe the model accuracy of the DGRN model in training and testing sets, as shown as Fig. 3.

### 5. Fig4.ipynb
The *Fig4.ipynb* draws the heatmap to depict the comparison among multiple empirical models, as shown as Fig. 4.

### 6. Fig5.py
The *Fig5.py* draws the Taylor diagram to depict the comparison among multiple empirical models, as shown as Fig. 5.

### 7. basemap_mapping_backups.ipynb
The *basemap_mapping_backups.ipynb* draws the spatio-temporal results of lake-average of SDD, as shown as Fig. 6, 7, 8, 9, 11, 12, S6, S7, S8. Detailed information can be accessed in the citation.

### 8. Fig10.ipynb
The *Fig10.ipynb* draws the histogram to depict the contributions of lake-specific characteristics to SDD, which is shown in Fig. 10.

### 9. normalization_sdd_l8.py
The *normalization_sdd_l8.py* draws the scatter plots to depict the model accuracy of multiple empirical models, which is shown in Fig. S3.

### 10. case_lakes.mxd & FigS1.mxd
The *case_lakes.mxd*, and *FigS1.mxd* are working in ArcGIS 10.3, which were used to describle the spatial variability of four lake cases and the distribution of in-situ measurements.

### 11. FigS5.ipynb
The *FigS5.ipynb* draws the probability distribution density of SDD in four lake cases, which is shown in Fig. S5.
