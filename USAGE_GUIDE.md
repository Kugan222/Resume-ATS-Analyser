# Resume ATS Analyzer - Usage Guide

## Quick Start

### Method 1: Interactive Mode (Recommended for beginners)
```bash
python main.py --interactive
```
This will launch an interactive menu where you can choose options and be guided through the process.

### Method 2: Command Line (Single Resume)
```bash
python main.py --resume path/to/resume.pdf --job path/to/job_description.txt --title "Job Title"
```

### Method 3: Batch Processing
```bash
python main.py --batch --resume-folder path/to/resumes/ --job path/to/job_description.txt --title "Job Title"
```

## Detailed Usage

### Prerequisites

1. **Python Environment**: Ensure you have Python 3.8+ installed
2. **Dependencies**: Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. **NLP Models**: Download required language models:
   ```bash
   python -m spacy download en_core_web_sm
   ```

### Input Requirements

#### Resume Files
- **Format**: PDF files only
- **Content**: Must contain selectable text (not scanned images)
- **Structure**: Works best with standard resume sections:
  - Contact Information
  - Experience/Work History
  - Education
  - Skills
  - Projects (optional)

#### Job Description Files
- **Format**: Plain text files (.txt)
- **Content**: Should include:
  - Job requirements
  - Required skills
  - Experience requirements
  - Education requirements
  - Job responsibilities

### Running the System

#### Interactive Mode
```bash
python main.py --interactive
```

**What you'll see:**
1. Welcome screen with options
2. Choose between single resume or batch analysis
3. Guided prompts for file paths and job information
4. Real-time progress updates
5. Detailed results display
6. Option to save results

#### Single Resume Analysis
```bash
python main.py --resume resume.pdf --job job.txt --title "Software Engineer"
```

**Parameters:**
- `--resume`: Path to the resume PDF file
- `--job`: Path to the job description text file
- `--title`: Job title (optional, but recommended)
- `--output`: Output filename for results (optional)

#### Batch Analysis
```bash
python main.py --batch --resume-folder resumes/ --job job.txt --title "Data Scientist"
```

**Parameters:**
- `--batch`: Enable batch mode
- `--resume-folder`: Directory containing resume PDF files
- `--job`: Path to job description text file
- `--title`: Job title (optional)
- `--output`: Output filename for results (optional)

### Understanding the Output

#### Overall Match Score
- **85-100%**: Excellent Match - Highly qualified candidate
- **70-84%**: Good Match - Well-qualified candidate
- **55-69%**: Fair Match - Potentially suitable with some gaps
- **0-54%**: Poor Match - Significant skill/requirement gaps

#### Detailed Analysis Sections

1. **Skill Analysis**
   - ✅ Found Skills: Skills present in both resume and job
   - ❌ Missing Skills: Required skills not found in resume
   - 📊 Skill Match Percentage: Overall skill compatibility

2. **Experience Analysis**
   - Years of experience vs. requirements
   - Experience level matching
   - Industry relevance

3. **Education Analysis**
   - Degree requirements vs. candidate qualifications
   - Field of study alignment
   - Education level matching

4. **Additional Metrics**
   - 📝 Text Similarity: Semantic similarity between resume and job
   - 🔍 Keyword Match: Keyword density matching

5. **Recommendations**
   - Specific skills to add or highlight
   - Areas for resume improvement
   - Gap analysis suggestions

### Sample Output

```
==================================================
RESUME ANALYSIS RESULTS
==================================================

Overall Match Score: 78.5% (Good Match)
Job Title: Software Engineer - Machine Learning

SKILL ANALYSIS:
✅ Found Skills: Python, Java, TensorFlow, Git, AWS
❌ Missing Skills: PyTorch, Kubernetes, React
📊 Skill Match: 71.4%

EXPERIENCE ANALYSIS:
✅ Resume: 4 years, Required: 3+ years
📊 Experience Match: 100.0%

EDUCATION ANALYSIS:
✅ Found 1 degree matches out of 1 requirements
📊 Education Match: 100.0%

ADDITIONAL METRICS:
📝 Text Similarity: 65.2%
🔍 Keyword Match: 58.7%

RECOMMENDATIONS:
1. Consider adding these skills: PyTorch, Kubernetes, React
2. Highlight relevant projects and accomplishments
```

### File Organization

```
resume-ats-analyzer/
├── data/
│   ├── sample_resumes/     # Place your resume PDFs here
│   └── sample_jobs/        # Sample job descriptions
├── main.py                 # Main application
├── requirements.txt        # Dependencies
└── results/               # Output files (created automatically)
```

### Tips for Best Results

#### Resume Optimization
1. **Clear Structure**: Use standard section headings
2. **Keywords**: Include industry-specific terms and technologies
3. **Quantify**: Use numbers and metrics where possible
4. **Format**: Ensure PDF is text-searchable (not scanned image)
5. **Completeness**: Include all relevant experience and skills

#### Job Description Preparation
1. **Detailed Requirements**: Include specific skills and tools
2. **Clear Structure**: Separate required vs. preferred qualifications
3. **Experience Levels**: Specify years of experience needed
4. **Education**: Include degree requirements if applicable

### Troubleshooting

#### Common Issues

1. **PDF Not Reading**
   - Ensure PDF contains selectable text
   - Try saving as a new PDF from Word/Google Docs
   - Check file permissions

2. **Low Match Scores**
   - Review job description for specific requirements
   - Ensure resume includes relevant keywords
   - Check for skill naming variations (e.g., "JavaScript" vs "JS")

3. **Missing Skills Detection**
   - Use standard skill names (e.g., "Python" not "python programming")
   - Include skills in a dedicated Skills section
   - Use industry-standard terminology

4. **Installation Issues**
   - Ensure Python 3.8+ is installed
   - Install dependencies one by one if batch install fails
   - Check internet connection for model downloads

#### Getting Help

- Run the test script: `python test_system.py`
- Check logs for detailed error messages
- Ensure all files are in the correct format and location

### Advanced Usage

#### Custom Configuration
- Modify skill patterns in `src/resume_parser.py` and `src/job_analyzer.py`
- Adjust matching weights in `src/matcher.py`
- Add new file formats by extending parser modules

#### Integration
- Import classes directly for custom applications
- Use JSON output for further processing
- Extend with additional NLP models or APIs

### Performance Notes

- **Single Resume**: ~10-30 seconds
- **Batch Processing**: ~1-2 minutes per resume
- **Memory Usage**: ~200-500MB depending on document size
- **Accuracy**: Best with well-structured documents and clear requirements
