
# entrepreneur_model.py
"""Synthetic entrepreneur prediction model using Random Forest."""

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

def generate_synthetic_data(n=1000, random_state=42):
    np.random.seed(random_state)
    age = np.random.randint(18, 71, size=n)
    education = np.random.choice([1, 2, 3], size=n, p=[0.6, 0.3, 0.1])
    work_experience = np.random.randint(0, 41, size=n)

    majors = ['Business', 'Engineering', 'Computer Science', 'Economics', 'Science', 'Arts', 'Other']
    industries = ['Tech', 'Finance', 'Healthcare', 'Education', 'Manufacturing', 'Retail', 'Other']

    prior_startup_involvement = np.random.binomial(1, 0.1, size=n)
    entrepreneurship_education = np.random.poisson(1.5, size=n)

    employment_titles = ['Analyst', 'Manager', 'Engineer', 'Consultant', 'Director', 'Technician', 'Clerk', 'Other']
    locations = ['San Francisco', 'New York', 'Chicago', 'Austin', 'Seattle', 'Remote', 'Other']

    risk_tolerance = np.clip(np.random.normal(0.5, 0.15, size=n), 0, 1)
    income = np.random.lognormal(mean=10.5, sigma=0.5, size=n)
    credit_score = np.clip(np.random.normal(700, 50, size=n), 300, 850)
    family_background = np.random.binomial(1, 0.2, size=n)
    network_size = np.clip(np.random.normal(50, 20, size=n), 0, None)
    savings = np.random.lognormal(mean=10, sigma=0.6, size=n)
    self_efficacy = np.clip(np.random.normal(0.6, 0.1, size=n), 0, 1)
    openness = np.clip(np.random.normal(0.5, 0.1, size=n), 0, 1)

    df = pd.DataFrame({
        'age': age,
        'education': education,
        'work_experience': work_experience,
        'major': np.random.choice(majors, size=n),
        'industry': np.where(work_experience > 0, np.random.choice(industries, size=n), 'Unemployed'),
        'prior_startup_involvement': prior_startup_involvement,
        'entrepreneurship_education': entrepreneurship_education,
        'employment_title': np.where(work_experience > 0, np.random.choice(employment_titles, size=n), 'Unemployed'),
        'location': np.random.choice(locations, size=n),
        'risk_tolerance': risk_tolerance,
        'income': income,
        'credit_score': credit_score,
        'family_background': family_background,
        'network_size': network_size,
        'savings': savings,
        'self_efficacy': self_efficacy,
        'openness': openness
    })

    title_scores = {
        'Analyst':0.2, 'Manager':0.5, 'Engineer':0.3, 'Consultant':0.4,
        'Director':0.6, 'Technician':0.1, 'Clerk':0.0, 'Other':0.1, 'Unemployed':0.0
    }
    location_scores = {
        'San Francisco':0.6, 'New York':0.5, 'Chicago':0.4,
        'Austin':0.4, 'Seattle':0.45, 'Remote':0.3, 'Other':0.2
    }
    df['title_score'] = df['employment_title'].map(title_scores)
    df['location_score'] = df['location'].map(location_scores)

    df_norm = df.copy()
    for col in ['age','education','work_experience','credit_score','network_size',
                'income','savings','entrepreneurship_education','title_score',
                'location_score','self_efficacy','openness']:
        df_norm[col] = (df[col] - df[col].mean()) / df[col].std()

    weights = {
        'age': -0.2, 'education': 0.5, 'work_experience': 0.3,
        'risk_tolerance': 1.0, 'income': 0.4, 'credit_score': 0.2,
        'family_background': 0.8, 'network_size': 0.3, 'savings': 0.4,
        'self_efficacy': 1.2, 'openness': 0.6,
        'prior_startup_involvement': 1.0, 'entrepreneurship_education': 0.5,
        'title_score': 0.7, 'location_score': 0.5
    }

    lin = sum(df_norm[feat] * w for feat, w in weights.items()) - 0.5
    prob = expit(lin)
    df['started_business'] = np.random.binomial(1, prob, size=n)
    return df

def train_and_evaluate(df):
    features = [
        'age', 'education', 'work_experience', 'major', 'industry',
        'prior_startup_involvement', 'entrepreneurship_education',
        'employment_title', 'location', 'risk_tolerance', 'income',
        'credit_score', 'family_background', 'network_size',
        'savings', 'self_efficacy', 'openness'
    ]
    X = pd.get_dummies(df[features], drop_first=True)
    y = df['started_business']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('ROC AUC:', roc_auc_score(y_test, y_proba))
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    df = generate_synthetic_data()
    train_and_evaluate(df)
