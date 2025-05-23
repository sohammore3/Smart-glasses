# Smart Glasses for Visually Impaired People

## Overview

This project develops smart glasses to assist visually impaired individuals by providing real-time environmental awareness through audio feedback. The system uses a camera attached to glasses (e.g., Raspberry Pi-based) and implements three core features:

- **Object Detection:** Identifies objects like chairs or doors using YOLOv5.
- **Face Recognition:** Recognizes known faces using the `face_recognition` library.
- **Text Recognition:** Reads text from signs or labels using Tesseract OCR.

The system processes visual input and delivers audio output via text-to-speech, enabling users to navigate and interact with their surroundings.

---

## Features

- **Object Detection:** Real-time detection of common objects using YOLOv5.
- **Face Recognition:** Identifies pre-registered individuals from a known faces database.
- **Text Recognition:** Extracts and reads text from images using OCR.
- **Audio Output:** Converts results to audio using gTTS and `pygame`.

---

## Prerequisites

### Hardware

- Raspberry Pi 4 with camera module or USB webcam
- Audio output device (earphones or speakers)

### Software

- Python 3.8+
- Tesseract OCR

Install Tesseract:
```bash
sudo apt install tesseract-ocr libtesseract-dev

