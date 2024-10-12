# Sight beyond Seeing

This application is an advanced assistive tool designed to support visually impaired individuals by leveraging state-of-the-art AI technology. Specifically, it uses multi-modal large language models (LLMs) that integrate visual and linguistic data to interpret and communicate complex visual context through spoken language. The app captures images from a user's surroundings and processes the visuals using multimodal LLMS to deliver detailed, near-real-time feedback in audio form. This app enables users to better understand their immediate environment by providing a practical and intuitive way to navigate daily life independently.

### Projects Structure

    .
    ├── audio                   # Contains Pre rendered common phrases in mp3 format
    ├── diagrams_and_images     # SUpporting diagrams and images like solution diagrams, results, etc.
    ├── training                # Tinyllama finetune traning data and code
    ├── install.txt             # Some setup info for Raspberry Pi
    ├── README.md
    ├── requirements.txt        # Tools and utilities
    └── sight_beyond_seeing.py  # Main application
