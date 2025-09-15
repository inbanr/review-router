# Router Review Project

The **Router Review Project** is an AI-powered customer feedback classifier. It analyzes Amazon product reviews and automatically routes negative complaints into relevant categories: Delivery, Product, or Other. Positive reviews are identified and preserved for potential marketing insights.

---

## Features

- Uses OpenAI's `gpt-3.5-turbo` to semantically classify customer complaints.
- Loads and parses Amazon review data from the Kaggle dataset (`bittlingmayer/amazonreviews`).
- Separates positive and negative feedback.
- Routes negative reviews into meaningful categories for operational response.
- Designed for future enhancements like tagging praise for marketing or training fine-tuned classifiers.

---

## Technologies Used

- Python 3.12
- LangChain
- OpenAI API
- Pandas
- dotenv
- Kaggle API

---

## Dataset

- Source: bittlingmayer/amazonreviews (via Kaggle)
- Format: `.bz2` compressed text files with sentiment-labeled reviews.

---

## Installation & Setup

1. **Clone the repo**  
   `git clone https://github.com/YOUR_USERNAME/review-router.git`  
   `cd review-router`

2. **Create a virtual environment**  
   `python3 -m venv venv`  
   `source venv/bin/activate`

3. **Install dependencies**  
   `pip install -r requirements.txt`

4. **Set up environment variables**  
   Create a `.env` file in the root directory with:  
   `OPENAI_API_KEY=your_openai_api_key_here`

5. **Download the Kaggle dataset**  
   - Move `kaggle.json` to `~/.kaggle/`  
   - Run the following in Python:

   ```python
   import kagglehub
   kagglehub.dataset_download("bittlingmayer/amazonreviews")
   ```

---

## Running the Classifier

To run the classifier on a sample of 10 negative reviews:  
`python main.py`

This script:
- Loads 10,000 Amazon reviews
- Filters out only negative reviews (label = 1)
- Uses OpenAI to classify a sample into: Delivery, Product, or Other
- Prints out the results for inspection

---

## Next Steps

- Expand classification to full dataset
- Save output to CSV or a database
- Fine-tune a model or train a supervised classifier
- Build a dashboard for visualization and routing
- Identify & tag positive reviews for marketing

---

## Author

**Inban Rajamani**  
Built with ❤️ using OpenAI + LangChain  
GitHub: https://github.com/inbanr

---

## License

MIT License