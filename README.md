# Facial Image Segmentation and Clustering

## Project Overview

This project was developed for the INT20H'24 Hackathon and focuses on image segmentation using unsupervised learning methods and classical computer vision techniques. The primary goal is to segment and cluster images of faces into specific categories, leveraging pre-trained neural networks for detection and processing.

## Dataset

We utilized the WIKI dataset from the following source: [IMDB-WIKI dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/). The dataset contains images of well-known individuals, which we filtered to include only face images in a frontal view. We selected and processed 9,000 high-quality face images for our project.

## Solution Breakdown

### Data Preparation (`int20h_data_preparation.ipynb`)

This notebook handles the initial processing of the dataset:
1. **Data Download**: Download the WIKI dataset containing images.
2. **Face Detection**: Apply a face detection algorithm to identify and crop face regions from the images.
3. **Image Filtering**: Filter and save the best 9,000 face images in a separate directory, ensuring only clean, frontal face images are used.

### Clustering (`clusterization.ipynb`)

This notebook implements the clustering of the prepared face images:
1. **Feature Extraction**: Use pre-trained neural networks to extract features from the face images, converting them into a more compact representation.
2. **Clustering Algorithm**: Experiment with several classical clustering algorithms (e.g., K-Means, DBSCAN) to segment the faces into distinct groups.
3. **Cluster Analysis**: Determine the optimal number of clusters (no more than 10) based on visual and logical consistency, ensuring a well-justified segmentation approach.

### Average Face Calculation (`face_avarage.ipynb`)

This notebook is responsible for calculating and visualizing the average face for each cluster:
1. **Image Normalization**: Align and normalize face images within each cluster.
2. **Average Calculation**: Compute the average face for each cluster by averaging pixel values.
3. **Visualization**: Display the average face images, providing a visual representation of the typical face for each cluster.

## How to Run the Program

1. **Clone the Repository**: 
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. **Install Dependencies**:
    Ensure you have `pip` and `virtualenv` installed. Create and activate a virtual environment:
    ```bash
    virtualenv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3. **Download Dataset**:
    Follow the instructions in the `int20h_data_preparation.ipynb` notebook to download and prepare the WIKI dataset.

4. **Run Data Preparation**:
    Execute the `int20h_data_preparation.ipynb` notebook to process and filter the face images:
    ```bash
    jupyter notebook int20h_data_preparation.ipynb
    ```

5. **Run Clustering**:
    Execute the `clusterization.ipynb` notebook to cluster the processed face images:
    ```bash
    jupyter notebook clusterization.ipynb
    ```

6. **Calculate Average Faces**:
    Execute the `face_avarage.ipynb` notebook to calculate and visualize the average faces for each cluster:
    ```bash
    jupyter notebook face_avarage.ipynb
    ```

## Conclusion

This project demonstrates a creative and technically sound approach to face image segmentation and clustering, using a combination of classical machine learning techniques and pre-trained neural networks. The result is a set of clearly defined face clusters and average face representations that highlight the diversity and common features within each group.

For further details, please refer to the provided Jupyter notebooks.
