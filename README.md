AI-Driven Resume Enhancer

Overview

The AI-Driven Resume Enhancer is a machine learning-based application that analyzes resumes and provides structured feedback to enhance their quality. By leveraging Natural Language Processing (NLP) and machine learning techniques, the system extracts key details such as skills, education, and experience from resumes and provides valuable insights, including resume scoring, skill recommendations, and career suggestions.

Features

Resume Parsing: Extracts key information (skills, experience, education) from PDF resumes using Pyresparser and spaCy.

Job Role Prediction: Suggests suitable job roles based on resume content.

Resume Scoring: Evaluates resumes and assigns a score based on industry standards.

Skill & Certification Recommendations: Identifies missing skills and suggests relevant certification courses.

YouTube Video Recommendations: Provides resources for resume building, ATS optimization, and interview preparation.

Technology Stack

Backend: Python (Flask)

Machine Learning: Scikit-learn, NLP (spaCy, Pyresparser)

Database: CSV-based data storage (roles-based-on-skills.csv)

Frontend: Streamlit

Dependencies: Refer to requirements.txt for a list of required libraries

Installation

Prerequisites

Python 3.7+

Virtual environment (optional but recommended)

Steps

Clone the repository:

git clone <repository-url>
cd AI-Resume-Enhancer

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run App.py

Files & Directories

App.py - Main application file

Courses.py - Module for handling course recommendations

model.py - Contains ML model logic for job role prediction

roles-based-on-skills.csv - Dataset mapping skills to job roles

requirements.txt - List of dependencies required for the project

Usage

Upload your resume in PDF format.

The system extracts relevant details and predicts suitable job roles.

View resume score, skill gaps, and certification suggestions.

Access video resources for improving resume quality and job prospects.

Future Enhancements

Integration with job portals for real-time job recommendations.

More advanced resume parsing with deep learning techniques.

Support for additional document formats (e.g., DOCX).

License

This project is licensed under the MIT License.

Contact

For any inquiries, reach out to Satish Dasu or visit GitHub.
