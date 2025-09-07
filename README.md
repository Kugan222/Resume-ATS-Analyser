# Resume ATS Analyzer

An intelligent Applicant Tracking System (ATS) that analyzes resumes against job descriptions using Natural Language Processing and Machine Learning techniques.

## 🚀 Features

- **PDF Resume Parsing**: Extracts text and structured information from PDF resumes
- **Keyword Extraction**: Identifies skills, experience, education, and projects from resumes
- **Job Description Analysis**: Analyzes job requirements and responsibilities
- **Smart Matching**: Uses NLP to match candidates to job requirements
- **Scoring System**: Provides compatibility scores with detailed breakdowns
- **Skill Gap Analysis**: Identifies missing skills and areas for improvement

## 📋 What You'll Get as Output

The system provides:

1. **Overall Match Score** (0-100%): How well the resume matches the job
2. **Skill Match Analysis**: 
   - Found skills vs Required skills
   - Missing critical skills
   - Skill relevance scores
3. **Experience Analysis**:
   - Years of experience vs requirements
   - Relevant project matches
   - Industry experience alignment
4. **Education Match**: Degree requirements vs candidate qualifications
5. **Detailed Recommendations**: Suggestions for resume improvement

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Step 1: Clone/Navigate to Project
```bash
cd resume-ats-analyzer
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download NLP Models
```bash
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet')"
```

## 📖 Usage

### Basic Usage
```bash
python main.py --resume path/to/resume.pdf --job path/to/job_description.txt
```

### Interactive Mode
```bash
python main.py --interactive
```

### Batch Processing
```bash
python main.py --batch --resume-folder data/sample_resumes --job data/sample_jobs/job1.txt
```

## 📁 Project Structure

```
resume-ats-analyzer/
├── src/
│   ├── __init__.py
│   ├── resume_parser.py      # PDF parsing and text extraction
│   ├── job_analyzer.py       # Job description analysis
│   ├── matcher.py           # Matching algorithm
│   └── utils.py             # Utility functions
├── data/
│   ├── sample_resumes/      # Sample PDF resumes for testing
│   └── sample_jobs/         # Sample job descriptions
├── models/                  # Trained models (if any)
├── tests/                   # Unit tests
├── main.py                  # Main application entry point
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🔍 Example Output

```
=== RESUME ANALYSIS RESULTS ===

Overall Match Score: 78%

SKILL ANALYSIS:
✅ Found Skills: Python, Machine Learning, SQL, Git
❌ Missing Skills: Docker, Kubernetes, AWS
📊 Skill Match: 65%

EXPERIENCE ANALYSIS:
📅 Experience: 3 years (Requirement: 2-4 years) ✅
🎯 Relevant Projects: 2/3 match job requirements
📊 Experience Match: 85%

EDUCATION ANALYSIS:
🎓 Degree: Bachelor's in Computer Science ✅
📊 Education Match: 90%

RECOMMENDATIONS:
1. Add Docker and containerization experience
2. Highlight cloud platform knowledge
3. Include more data science project details
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues:

1. **PDF not reading**: Ensure PDF is not password protected and contains selectable text
2. **Model download fails**: Check internet connection and try downloading models manually
3. **Low accuracy**: The system works best with well-structured resumes and detailed job descriptions

### Support
If you encounter any issues, please check the troubleshooting section or open an issue on GitHub.
