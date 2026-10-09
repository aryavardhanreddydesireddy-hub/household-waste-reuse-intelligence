# Household Waste Reuse Intelligence

A beginner-friendly Python and Streamlit prototype that records household items and suggests reuse, repair, donation, recycling, or responsible-disposal options based on material and condition.

## Features
- Record items with name, material, condition, date, and notes.
- Save item records in a local SQLite database.
- Generate general suggestions using transparent rules.
- Rank general ideas using TF-IDF text similarity.
- Upload and display an item photo.
- View item history and download it as CSV.

## Technology stack
Python, Streamlit, SQLite, Pandas, Scikit-learn, Pillow.

## Run on Windows
1. Install Python 3.11 or newer from https://www.python.org/downloads/.
2. Open Command Prompt in this project folder.
3. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
4. Activate it:
   ```bash
   .venv\Scripts\activate
   ```
5. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
6. Run the application:
   ```bash
   streamlit run app.py
   ```
7. Click **Load sample data** in the sidebar to try fictional examples.

## Database and privacy
SQLite creates `data/waste_reuse.db` when the app runs. The database is excluded from Git because it is local data. Uploaded images are displayed during the session and are not saved to the database.

## Important scope note
This starter version does not automatically classify images; the uploader only displays them. Suggestions are rule-based and TF-IDF text similarity is a lightweight text feature. No trained deep-learning classifier or LLM is integrated yet. Recycling rules vary by location, so verify local guidance before acting.

## Future improvements
- Add a pretrained image-classification model and show its confidence.
- Add a verified local recycling and e-waste directory.
- Add optional LLM-generated reuse ideas.
- Add unit tests and screenshots of the running app.
