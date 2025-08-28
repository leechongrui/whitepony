# Raw Data Documentation

This folder contains the original dataset provided for the hackathon.

## Files
- `kaggle_google_restuarant_reviews.csv` – 1,100 Google Maps reviews with the following columns:
  - `business_name`: Name of the business
  - `author_name`: Reviewer name
  - `text`: Review text
  - `photo`: Image path
  - `rating`: Numeric rating (1–5)
  - `rating_category`: Category (e.g., taste, menu)


- `colorado_reviews.json` - contains Google Local review data from Montana.


  ### Self Sourced
- `kaggle_hotel_spam_corpus` - contains 400 truthful positive reviews from TripAdvisor, 400 deceptive positive reviews from Mechanical Turk, 400 truthful negative reviews from Expedia, Hotels.com, Orbitz, Priceline, TripAdvisor and Yelp, 400 deceptive negative reviews from Mechanical Turk.
  - `deceptive`: Label indicating if the review is fake/deceptive (True/False or 1/0)
  - `hotel`: Name or ID of the hotel being reviewed
  - `polarity`: Sentiment of the review (e.g., positive or negative)
  - `source`: Where the review was obtained (e.g., TripAdvisor, Google, Yelp)
  - `text`: The actual review content (main feature for NLP)
