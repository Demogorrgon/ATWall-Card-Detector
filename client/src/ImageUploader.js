import React, { useState, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';

import styles from './ImageUploader.module.css';

const ImageUploader = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [boundingBoxes, setBoundingBoxes] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const handleImageUpload = async (acceptedFiles) => {
    const file = acceptedFiles[0];
    setSelectedImage(URL.createObjectURL(file));

    setIsLoading(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${process.env.REACT_APP_API_ENDPOINT}/recognize_text`, {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const data = await response.json();
        setBoundingBoxes(data["bounding_boxes"]);
      } else {
        console.error('Error:', response.statusText);
      }
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const {getRootProps, getInputProps} = useDropzone({
    accept: 'image/*',
    onDrop: handleImageUpload,
    multiple: false
  });

  useEffect(() => {
    const imageElement = document.getElementById('uploaded-image');
    if (imageElement) {
      const canvas = document.createElement('canvas');
      canvas.width = imageElement.width;
      canvas.height = imageElement.height;
      const context = canvas.getContext('2d');
      context.drawImage(imageElement, 0, 0);
      boundingBoxes.forEach((box) => {
        const [ymin, xmin, ymax, xmax] = box.box;
        const x = xmin * imageElement.width;
        const y = ymin * imageElement.height;
        const width = (xmax - xmin) * imageElement.width;
        const height = (ymax - ymin) * imageElement.height;
        context.strokeStyle = 'rgba(171, 37, 250, 0.8)';
        context.lineWidth = 2;
        context.beginPath();
        context.rect(x, y, width, height);
        context.stroke();
      });
      imageElement.src = canvas.toDataURL();
    }
  }, [boundingBoxes]);

  const handleClickImage = () => {
    if (selectedImage) {
      setSelectedImage(null);
    }
  };

  return (
    <div className={styles.container}>
      <h1 className={styles.header}>ATWALL</h1>
      {!selectedImage ? (
        <div className={styles.dropzoneContainer} {...getRootProps()}>
          <input {...getInputProps()} />
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="48"
            height="48"
            viewBox="0 0 48 48"
            fill="none"
            stroke="#AB25FA"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className={styles.dropzoneIcon}
          >
            <path d="M12 22l6 6 6-6M12 8h12"/>
          </svg>
          <p className={styles.dropzoneText}>Drag and drop an image here or click to select</p>
        </div>
      ) : (
        <div className={styles.imageContainer} onClick={handleClickImage}>
          {isLoading && <div className={styles.loader}></div>}
          <img
            id="uploaded-image"
            src={selectedImage}
            alt="Uploaded"
            className={`${styles.uploadedImage} ${isLoading ? styles.grayedOut : ''}`}
          />
        </div>
      )}
    </div>
  );
};

export default ImageUploader;