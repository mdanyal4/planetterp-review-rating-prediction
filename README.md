# PlanetTerp Review Rating Prediction

An NLP notebook that predicts 1–5-star ratings from student review text and compares them with the ratings students actually submitted on PlanetTerp.

## How it works

- Collects reviews for eight professors through the PlanetTerp API.
- Builds a pandas dataset and removes incomplete and duplicate records.
- Uses the pretrained `nlptown/bert-base-multilingual-uncased-sentiment` model to predict a star rating for each review.
- Compares predicted and submitted ratings using a confusion matrix and mean absolute error (MAE).
- Plots the distribution of submitted star ratings.

The model is used for inference as provided; this project does not train or fine-tune BERT.

## Result

The committed notebook reports **0.67 stars mean absolute error**. In that saved run, predictions differed from submitted ratings by 0.67 stars on average. This is an error measure, not a classification accuracy percentage. Fresh API data and dependency changes can affect rerun results.

## Stack

Python, pandas, NumPy, requests, Hugging Face Transformers, Matplotlib, and Seaborn.

## Run the notebook

1. Download or clone the repository.
2. Install the dependencies in your Python environment:

   ```bash
   pip install pandas numpy requests transformers torch matplotlib seaborn jupyter
   ```

3. Open the notebook:

   ```bash
   jupyter notebook review_rating_prediction.ipynb
   ```

4. Run the cells from top to bottom. Internet access is needed to fetch reviews and download the pretrained model on its first run.

You can also upload [review_rating_prediction.ipynb](review_rating_prediction.ipynb) to Google Colab and install any missing dependencies there.

## Files

- `review_rating_prediction.ipynb` — API collection, cleaning, inference, plots, and saved evaluation output.

## Scope and limitations

This is an exploratory analysis of reviews for eight selected professors, not a representative measure of teaching quality. The current code sends only the first 512 **characters** of each review to the model; that is not token-aware truncation and can omit context. Course lists are manually attached at the professor level, rather than identifying the course for each individual review. Submitted ratings are subjective, and model predictions should not be treated as objective evaluations of professors.
