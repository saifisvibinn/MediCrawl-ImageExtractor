# PDF Figure Extractor

A Streamlit web application that automatically extracts figures and captions from PDF files using computer vision and OCR.

## Features

- 📄 **Batch Processing**: Upload and process multiple PDF files at once
- 🖼️ **Smart Detection**: Automatically detects figures, charts, and diagrams
- 📝 **Caption Recognition**: Uses OCR to identify and match captions with figures
- 🔄 **Multi-page Stitching**: Combines figures split across multiple pages
- 💾 **Easy Download**: Download all results as a ZIP file or individually
- 🎨 **Modern UI**: Clean, responsive web interface

## Live Demo

[Deploy on Streamlit Cloud](https://streamlit.io/cloud)

## Installation

### Local Development

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR (optional, for caption detection):
   - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - **Mac**: `brew install tesseract`
   - **Linux**: `sudo apt-get install tesseract-ocr`

4. Run the app:
```bash
streamlit run pdf_extractor_gui.py
```

### Streamlit Cloud Deployment

1. Fork this repository
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app and select your forked repository
4. Set the main file path to: `pdf_extractor_gui.py`
5. Deploy!

**Note**: Tesseract OCR may not be available on Streamlit Cloud. The app will still extract figures but caption detection will be limited.

## Usage

1. **Upload PDFs**: Drag and drop or browse to select PDF files
2. **Adjust Settings** (optional):
   - Image Quality (DPI): 150-600 (default: 400)
   - Auto-rotate: Automatically rotate tall images
3. **Extract**: Click the "Extract Figures" button
4. **Download**: Preview results and download as ZIP or individually

## How It Works

1. **Page Rendering**: Each PDF page is rendered as a high-resolution image
2. **Figure Detection**: Uses OpenCV to detect rectangular regions (figures)
3. **Caption Recognition**: OCR scans for text matching caption patterns (Figure, Fig., Table, etc.)
4. **Smart Matching**: Captions are matched with nearby figures
5. **Stitching**: Multi-page figures are automatically combined
6. **Export**: Figures are saved with their captions

## Configuration

Adjust these settings in the sidebar:

- **DPI**: Higher values = better quality but slower processing
  - 200: Fast preview
  - 400: Recommended (default)
  - 600: High quality for small details

- **Auto-rotate**: Rotates images that are taller than wide (useful for landscape papers)

## Output Structure

Extracted figures are organized by PDF filename:

```
extracted_figures.zip
├── document1/
│   ├── figure_with_caption_f001.png
│   ├── figure_with_caption_f002.png
│   └── figure_f003.png
└── document2/
    ├── figure_with_caption_f001.png
    └── standalone_caption_001.png
```

## Requirements

- Python 3.8+
- Streamlit
- OpenCV
- PyMuPDF (fitz)
- pytesseract (optional)
- NumPy
- Pillow

## Limitations

- Works best with PDFs containing boxed/bordered figures
- Caption detection requires Tesseract OCR
- Very large PDFs may take time to process
- Complex layouts may require manual review

## Tips for Best Results

- Use clean, high-quality PDFs (not scanned images)
- Ensure figures have clear borders
- Standard caption formats work best (Figure 1, Fig. 2, etc.)
- Try different DPI settings if results aren't optimal

## Troubleshooting

**No figures detected?**
- Try adjusting DPI settings
- Check if PDF contains actual figures (not just text)
- Ensure figures have clear borders

**Captions not detected?**
- Install Tesseract OCR
- Check caption format (should start with "Figure", "Fig.", or "Table")
- Try higher DPI for better OCR accuracy

**Slow processing?**
- Lower the DPI setting
- Process fewer PDFs at once
- Large PDFs naturally take longer

## License

MIT License - feel free to use and modify as needed.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues or questions, please open an issue on GitHub.

---

Made with ❤️ using Streamlit
