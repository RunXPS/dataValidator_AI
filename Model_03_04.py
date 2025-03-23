import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load the dataset
def predict_similarity(csv_file, new_row):
    """
    Predicts the similarity of a new row to the existing rows in a dataset.

    Args:
        csv_file (str): Path to the CSV file containing the dataset.
        new_row (dict): A dictionary representing the new row to be added.
                         The keys should match the column names in the CSV file.

    Returns:
        tuple: A tuple containing:
            - similarities (list): List of similarity scores between the new row and existing rows.
            - most_similar_index (int): Index of the most similar row in the dataset.
    """
    try:
        df = pd.read_csv(csv_file, encoding="unicode_escape")

    except FileNotFoundError:
        return "Error: CSV file not found."
    except Exception as e:
        return f"Error: Failed to read CSV file. {str(e)}"

    # Fill missing values with an empty string
    df = df.fillna("")

    # drop irrelavent column values
    df.drop(
        [
            "personid",
            "Person Full Name",
            "Person Constituent ID",
            "Person First Name",
            "Person Last Name",
            "Person URL",
            "Employment Company Name",
            "Employment End Month",
            "Employment End Year",
            "Company Details Size",
            "Employment Captured Date",
            "Person Email",
            "Employment Title Is Senior",
            "Employment Salary Min",
            "Employment Salary Max",
            "Employment Salary Captured Date",
            "Employment Seniority Level",
            "Company Type Type",
            "_totalcount_",
        ],
        axis=1,
    )
    print(f"{df.columns}")

    # Combine all text columns into a single text feature
    text_columns = df.select_dtypes(include="object").columns
    df["combined_text"] = df[text_columns].apply(
        lambda row: " ".join(row.values.astype(str)), axis=1
    )

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["combined_text"])

    # Create a DataFrame from the new row
    new_row_df = pd.DataFrame([new_row])

    # Fill missing values in the new row with an empty string
    new_row_df = new_row_df.fillna("")

    # Combine text columns in the new row
    new_row_df["combined_text"] = new_row_df[text_columns].apply(
        lambda row: " ".join(row.values.astype(str)), axis=1
    )

    # Transform the new row using the fitted vectorizer
    new_row_vector = vectorizer.transform(new_row_df["combined_text"])

    # Calculate cosine similarity between the new row and existing dataset
    similarities = cosine_similarity(new_row_vector, tfidf_matrix)

    # Find the index of the most similar row
    most_similar_index = np.argmax(similarities)

    return [similarities[0], most_similar_index]


# Example usage
csv_file = "/workspaces/Modeling/LiveAlumni founder owner entrepreneur 08292024(Table).csv"  # Replace 'your_file.csv' with the actual path to your CSV file

# Example of a new row to be added
new_row = {
    "personid": "00000001",
    "Person Full Name": "John Doe",
    "Person Constituent ID": "00000001",
    "Person First Name": "John",
    "Person Last Name": "Doe",
    "Person URL": "https://www.linkedin.com/in/erikadietrick/",
    "Person Headline": "Founder @ Stealth",
    "Employment Title": "Founder",
    "Employment Company Name": "Stealth",
    "Employment Start Month": "April",
    "Employment Start Year": "2022",
    "Employment End Month": "September",
    "Employment End Year": "2024",
    "Company Details Size": "1-10",
    "Industry Name": "Technology",
    "Company Industry Name": "Software Development",
    "Location City": "Los Angeles",
    "Location State": "California",
    "Location Country": "USA",
    "Employment Captured Date": "03/04/2025",
    "Person Email": "",
    "Employment Title Is Senior": "TRUE",
    "Employment Salary Min": "0",
    "Employment Salary Max": "10",
    "Employment Salary Captured Date": "03/04/2025",
    "Employment Seniority Level": "",
    "Company Type Type": "",
    "_totalcount_": "",
}


similarities, most_similar_index = predict_similarity(csv_file, new_row)

if isinstance(similarities, str):
    print(similarities)  # Print the error message
else:
    # Cosine Similarities
    # 1 = same thing
    # 0 = not similar
    # -1 = complete opposites

    print("Similarities:", similarities)
    print(
        f"Most similar index: {most_similar_index} = {similarities[most_similar_index]}"
    )
