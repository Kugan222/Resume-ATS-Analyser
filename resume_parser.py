"""
Resume Parser Module
Extracts text and structured information from PDF resumes
"""

import os
import re
import PyPDF2
import pdfplumber
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResumeParser:
    def __init__(self):
        """Initialize the resume parser with NLP models and tools"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            
            # Common skill keywords categorized
            self.skill_patterns = {
                'programming_languages': [
                    'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift',
                    'kotlin', 'scala', 'r', 'matlab', 'perl', 'go', 'rust', 'typescript'
                ],
                'web_technologies': [
                    'html', 'css', 'react', 'angular', 'vue', 'nodejs', 'express',
                    'django', 'flask', 'spring', 'laravel', 'bootstrap', 'jquery'
                ],
                'databases': [
                    'mysql', 'postgresql', 'mongodb', 'sqlite', 'oracle', 'sql server',
                    'redis', 'elasticsearch', 'cassandra', 'dynamodb'
                ],
                'cloud_platforms': [
                    'aws', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes',
                    'jenkins', 'terraform', 'ansible'
                ],
                'data_science': [
                    'machine learning', 'deep learning', 'tensorflow', 'pytorch',
                    'scikit-learn', 'pandas', 'numpy', 'matplotlib', 'seaborn',
                    'tableau', 'power bi', 'data visualization'
                ],
                'tools': [
                    'git', 'github', 'gitlab', 'jira', 'confluence', 'slack',
                    'visual studio', 'eclipse', 'intellij', 'vim', 'emacs'
                ]
            }
            
            # Education patterns
            self.education_patterns = [
                r'bachelor.*?(?:degree|of|in)',
                r'master.*?(?:degree|of|in)', 
                r'phd.*?(?:degree|of|in)',
                r'doctorate.*?(?:degree|of|in)',
                r'b\.?(?:sc|a|tech|e)\.?',
                r'm\.?(?:sc|a|tech|ba|s)\.?',
                r'ph\.?d\.?'
            ]
            
        except Exception as e:
            logger.error(f"Error initializing ResumeParser: {e}")
            raise

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using multiple methods for robustness"""
        text = ""
        
        # Method 1: Try pdfplumber first (better for complex layouts)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            if text.strip():
                logger.info(f"Successfully extracted text using pdfplumber from {pdf_path}")
                return text.strip()
                
        except Exception as e:
            logger.warning(f"pdfplumber failed for {pdf_path}: {e}")

        # Method 2: Fall back to PyPDF2
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            
            if text.strip():
                logger.info(f"Successfully extracted text using PyPDF2 from {pdf_path}")
                return text.strip()
                
        except Exception as e:
            logger.error(f"Both PDF extraction methods failed for {pdf_path}: {e}")
            
        return ""

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace and special characters
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s@.-]', ' ', text)
        return text.strip()

    def extract_contact_info(self, text: str) -> Dict[str, Optional[str]]:
        """Extract contact information from resume text"""
        contact_info = {
            'email': None,
            'phone': None,
            'linkedin': None,
            'github': None
        }
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, text)
        if email_match:
            contact_info['email'] = email_match.group()

        # Phone pattern (various formats)
        phone_pattern = r'(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
        phone_match = re.search(phone_pattern, text)
        if phone_match:
            contact_info['phone'] = phone_match.group()

        # LinkedIn pattern
        linkedin_pattern = r'linkedin\.com/in/[\w-]+'
        linkedin_match = re.search(linkedin_pattern, text, re.IGNORECASE)
        if linkedin_match:
            contact_info['linkedin'] = linkedin_match.group()

        # GitHub pattern
        github_pattern = r'github\.com/[\w-]+'
        github_match = re.search(github_pattern, text, re.IGNORECASE)
        if github_match:
            contact_info['github'] = github_match.group()

        return contact_info

    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """Extract skills from resume text"""
        text_lower = text.lower()
        found_skills = {}
        
        for category, skills in self.skill_patterns.items():
            found_skills[category] = []
            for skill in skills:
                # Use word boundary matching for better accuracy
                pattern = r'\b' + re.escape(skill.lower()) + r'\b'
                if re.search(pattern, text_lower):
                    found_skills[category].append(skill.title())
        
        # Remove empty categories
        found_skills = {k: v for k, v in found_skills.items() if v}
        
        return found_skills

    def extract_experience(self, text: str) -> Dict[str, any]:
        """Extract work experience information"""
        experience_info = {
            'total_years': 0,
            'companies': [],
            'positions': [],
            'experience_text': ""
        }
        
        # Look for experience section
        experience_section_patterns = [
            r'(?:work\s+)?experience\s*:?\s*(.*?)(?=education|skills|projects|$)',
            r'professional\s+experience\s*:?\s*(.*?)(?=education|skills|projects|$)',
            r'employment\s+history\s*:?\s*(.*?)(?=education|skills|projects|$)'
        ]
        
        for pattern in experience_section_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                experience_info['experience_text'] = match.group(1)
                break
        
        # Extract years of experience
        years_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\s*-\s*(\d+)\s*years?\s*(?:of\s*)?experience'
        ]
        
        for pattern in years_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if len(match.groups()) == 1:
                    experience_info['total_years'] = int(match.group(1))
                else:
                    experience_info['total_years'] = int(match.group(2))
                break
        
        # Extract company names (basic pattern matching)
        company_patterns = [
            r'(?:at\s+|@\s+)([A-Z][A-Za-z\s&.,]+(?:Inc|LLC|Corp|Company|Ltd))',
            r'\b([A-Z][A-Za-z\s&.,]{2,15}(?:Inc|LLC|Corp|Company|Ltd))\b'
        ]
        
        for pattern in company_patterns:
            matches = re.findall(pattern, experience_info['experience_text'])
            experience_info['companies'].extend([match.strip() for match in matches])
        
        # Remove duplicates
        experience_info['companies'] = list(set(experience_info['companies']))
        
        return experience_info

    def extract_education(self, text: str) -> Dict[str, List[str]]:
        """Extract education information"""
        education_info = {
            'degrees': [],
            'institutions': [],
            'fields_of_study': []
        }
        
        text_lower = text.lower()
        
        # Extract degrees
        for pattern in self.education_patterns:
            matches = re.findall(pattern, text_lower)
            education_info['degrees'].extend(matches)
        
        # Extract university/college names (basic pattern)
        institution_patterns = [
            r'\b(?:university|college|institute|school)\s+of\s+[\w\s]+',
            r'\b[\w\s]+\s+(?:university|college|institute)\b'
        ]
        
        for pattern in institution_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            education_info['institutions'].extend([match.strip() for match in matches])
        
        # Clean and deduplicate
        education_info['degrees'] = list(set([deg.strip() for deg in education_info['degrees'] if deg.strip()]))
        education_info['institutions'] = list(set([inst.strip() for inst in education_info['institutions'] if inst.strip()]))
        
        return education_info

    def extract_projects(self, text: str) -> List[str]:
        """Extract project information"""
        projects = []
        
        # Look for projects section
        project_patterns = [
            r'projects?\s*:?\s*(.*?)(?=experience|education|skills|$)',
            r'personal\s+projects?\s*:?\s*(.*?)(?=experience|education|skills|$)',
            r'key\s+projects?\s*:?\s*(.*?)(?=experience|education|skills|$)'
        ]
        
        for pattern in project_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                project_text = match.group(1)
                # Split by bullet points or line breaks
                project_lines = re.split(r'[•\-\*]|\n', project_text)
                projects.extend([line.strip() for line in project_lines if line.strip() and len(line.strip()) > 20])
                break
        
        return projects[:5]  # Return top 5 projects

    def parse_resume(self, pdf_path: str) -> Dict[str, any]:
        """Main method to parse resume and extract all information"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Resume file not found: {pdf_path}")
        
        logger.info(f"Parsing resume: {pdf_path}")
        
        # Extract text
        raw_text = self.extract_text_from_pdf(pdf_path)
        if not raw_text:
            raise ValueError(f"Could not extract text from PDF: {pdf_path}")
        
        # Clean text
        cleaned_text = self.clean_text(raw_text)
        
        # Extract all information
        resume_data = {
            'file_path': pdf_path,
            'raw_text': raw_text,
            'cleaned_text': cleaned_text,
            'contact_info': self.extract_contact_info(cleaned_text),
            'skills': self.extract_skills(cleaned_text),
            'experience': self.extract_experience(cleaned_text),
            'education': self.extract_education(cleaned_text),
            'projects': self.extract_projects(cleaned_text),
            'word_count': len(cleaned_text.split()),
            'parsed_successfully': True
        }
        
        logger.info(f"Successfully parsed resume: {pdf_path}")
        return resume_data

    def get_all_skills_flat(self, skills_dict: Dict[str, List[str]]) -> List[str]:
        """Flatten skills dictionary to a single list"""
        all_skills = []
        for category_skills in skills_dict.values():
            all_skills.extend(category_skills)
        return all_skills


def main():
    """Test function for the resume parser"""
    parser = ResumeParser()
    
    # Test with a sample PDF (you'll need to provide one)
    try:
        sample_pdf = "data/sample_resumes/sample_resume.pdf"
        if os.path.exists(sample_pdf):
            result = parser.parse_resume(sample_pdf)
            print("Resume parsed successfully!")
            print(f"Contact Info: {result['contact_info']}")
            print(f"Skills Found: {result['skills']}")
            print(f"Experience: {result['experience']['total_years']} years")
            print(f"Education: {result['education']['degrees']}")
        else:
            print(f"Sample PDF not found at {sample_pdf}")
            print("Please add a sample resume PDF to test the parser")
    
    except Exception as e:
        print(f"Error testing resume parser: {e}")


if __name__ == "__main__":
    main()
