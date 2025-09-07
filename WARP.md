# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

The Resume ATS Analyzer is an intelligent Applicant Tracking System that uses Natural Language Processing and Machine Learning to analyze resumes against job descriptions. It provides compatibility scores, skill gap analysis, and detailed recommendations for candidates.

## Core Architecture

The application follows a modular architecture with four main components:

### Main Components
- **`main.py`**: Application entry point with CLI interface and interactive mode
- **`src/resume_parser.py`**: PDF parsing, text extraction, and resume data structuring
- **`src/job_analyzer.py`**: Job description parsing and requirements extraction
- **`src/matcher.py`**: Scoring algorithms that match resumes to jobs using weighted metrics
- **`src/utils.py`**: Utility functions for file handling, validation, and data processing

### Data Flow
1. Resume PDF → ResumeParser → Structured resume data (skills, experience, education)
2. Job description text → JobAnalyzer → Requirements data (skills, experience, education)
3. Both datasets → ResumeJobMatcher → Weighted compatibility score + detailed analysis

### Scoring System
The matcher uses weighted scoring across five dimensions:
- **Skills (35%)**: Exact and fuzzy matching of technical skills
- **Experience (25%)**: Years of experience vs requirements
- **Education (15%)**: Degree requirements matching
- **Semantic Similarity (15%)**: TF-IDF cosine similarity between texts
- **Keywords (10%)**: Keyword density matching

## Common Development Commands

### Initial Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate

# Activate virtual environment (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLP models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet')"
```

### Running the Application
```bash
# Interactive mode (recommended for testing)
python main.py --interactive

# Single resume analysis
python main.py --resume path/to/resume.pdf --job path/to/job_description.txt --title "Job Title"

# Batch processing
python main.py --batch --resume-folder data/sample_resumes --job data/sample_jobs/software_engineer_ml.txt --title "Software Engineer"

# With verbose logging
python main.py --interactive --verbose
```

### Testing and Development
```bash
# Run system test with sample data
python test_system.py

# Test individual components (example workflow)
python -c "from src.job_analyzer import JobAnalyzer; ja = JobAnalyzer(); print('JobAnalyzer working')"
```

### Working with Sample Data
The `data/` directory contains sample job descriptions. To test with real resumes, place PDF files in `data/sample_resumes/` (directory may need to be created).

## Key Technical Details

### NLP Dependencies
- **spaCy**: Used for text processing and semantic analysis (requires `en_core_web_sm` model)
- **NLTK**: Used for tokenization, stopwords, and lemmatization
- **scikit-learn**: TF-IDF vectorization and cosine similarity
- **fuzzywuzzy**: Fuzzy string matching for skills

### PDF Processing
The system uses a two-tier approach for PDF text extraction:
1. **Primary**: `pdfplumber` for complex layouts and tables
2. **Fallback**: `PyPDF2` if pdfplumber fails

### Skill Matching Strategy
Skills are categorized into predefined groups (programming languages, web technologies, databases, cloud platforms, data science, tools) and matched using:
- Exact string matching with word boundaries
- Fuzzy matching (80% threshold) for variations
- Skill name normalization for common variations

### File Structure Expectations
```
data/
├── sample_resumes/     # PDF files for testing (not in git)
└── sample_jobs/        # Text files with job descriptions
    ├── software_engineer_ml.txt
    └── data_scientist.txt

results/                # Output directory (created automatically)
├── analysis_results.json
└── batch_analysis_*.json
```

## Development Guidelines

### Adding New Skills
Skills are defined in both `resume_parser.py` and `job_analyzer.py` in the `skill_patterns` dictionary. When adding skills:
- Add to both files consistently
- Use lowercase for pattern matching
- Consider common variations and abbreviations
- Test with both resume and job description matching

### Modifying Scoring Weights
Scoring weights are defined in `matcher.py` in the `weights` dictionary. The current distribution prioritizes:
1. Skills matching (35%) - most critical
2. Experience level (25%) - years of experience
3. Education (15%) - degree requirements
4. Semantic similarity (15%) - overall text similarity
5. Keywords (10%) - keyword density

### Error Handling
The application uses extensive try-catch blocks and logging. Key patterns:
- Log errors at appropriate levels (INFO, WARNING, ERROR)
- Provide user-friendly error messages
- Graceful degradation (e.g., if one PDF fails in batch mode, continue with others)
- File validation before processing

### Output Formats
The system generates:
- **Console output**: Formatted analysis results with scores and recommendations
- **JSON files**: Detailed results for programmatic processing
- **Interactive feedback**: Progress updates and guided prompts

## Testing Strategy

Use `test_system.py` to validate:
- Job description analysis functionality
- Complete workflow simulation with mock resume data
- Expected output format demonstration

The test script demonstrates all core capabilities without requiring actual PDF files, making it useful for development and CI/CD validation.
