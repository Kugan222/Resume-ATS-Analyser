"""
Job Description Analyzer Module
Analyzes job descriptions and extracts requirements, skills, and responsibilities
"""

import re
import spacy
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import Dict, List, Optional, Set
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JobAnalyzer:
    def __init__(self):
        """Initialize the job analyzer with NLP models and patterns"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            
            # Skill keywords (same as resume parser for consistency)
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
            
            # Experience level patterns
            self.experience_patterns = [
                r'(\d+)\+?\s*(?:to|-)\s*(\d+)\s*years?\s*(?:of\s*)?experience',
                r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
                r'minimum\s*(?:of\s*)?(\d+)\s*years?',
                r'at\s*least\s*(\d+)\s*years?',
                r'(\d+)\s*(?:to|-)\s*(\d+)\s*years?'
            ]
            
            # Education requirement patterns
            self.education_patterns = [
                r'bachelor.*?(?:degree|in|of)',
                r'master.*?(?:degree|in|of)',
                r'phd.*?(?:degree|in|of)',
                r'doctorate.*?(?:degree|in|of)',
                r'b\.?(?:sc|a|tech|e)\.?',
                r'm\.?(?:sc|a|tech|ba|s)\.?',
                r'ph\.?d\.?'
            ]
            
            # Requirement keywords
            self.requirement_keywords = [
                'required', 'must have', 'essential', 'mandatory', 'necessary',
                'prerequisite', 'minimum', 'should have', 'preferred', 'desired',
                'nice to have', 'plus', 'advantage', 'bonus'
            ]
            
        except Exception as e:
            logger.error(f"Error initializing JobAnalyzer: {e}")
            raise

    def clean_job_description(self, text: str) -> str:
        """Clean and normalize job description text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?()-]', ' ', text)
        return text.strip()

    def extract_skills_from_job(self, text: str) -> Dict[str, List[str]]:
        """Extract required skills from job description"""
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

    def extract_experience_requirements(self, text: str) -> Dict[str, any]:
        """Extract experience requirements from job description"""
        experience_req = {
            'min_years': 0,
            'max_years': 0,
            'experience_text': '',
            'level': 'not_specified'
        }
        
        # Look for experience patterns
        for pattern in self.experience_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple) and len(match) == 2:
                    # Range: "3-5 years"
                    experience_req['min_years'] = max(experience_req['min_years'], int(match[0]))
                    experience_req['max_years'] = max(experience_req['max_years'], int(match[1]))
                elif isinstance(match, str) and match.isdigit():
                    # Single number: "3+ years"
                    years = int(match)
                    if experience_req['min_years'] == 0:
                        experience_req['min_years'] = years
                    else:
                        experience_req['max_years'] = max(experience_req['max_years'], years)
        
        # Determine experience level
        min_years = experience_req['min_years']
        if min_years == 0:
            experience_req['level'] = 'entry_level'
        elif min_years <= 2:
            experience_req['level'] = 'junior'
        elif min_years <= 5:
            experience_req['level'] = 'mid_level'
        elif min_years <= 8:
            experience_req['level'] = 'senior'
        else:
            experience_req['level'] = 'expert'
        
        return experience_req

    def extract_education_requirements(self, text: str) -> Dict[str, List[str]]:
        """Extract education requirements from job description"""
        education_req = {
            'required_degrees': [],
            'preferred_degrees': [],
            'fields_of_study': []
        }
        
        text_lower = text.lower()
        
        # Look for education patterns
        for pattern in self.education_patterns:
            matches = re.findall(pattern, text_lower)
            education_req['required_degrees'].extend([match.strip() for match in matches if match.strip()])
        
        # Look for field of study patterns
        field_patterns = [
            r'(?:in|of)\s+([A-Za-z\s]+)\s+(?:engineering|science|studies|management)',
            r'computer\s+science', r'software\s+engineering', r'data\s+science',
            r'information\s+technology', r'business\s+administration'
        ]
        
        for pattern in field_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            education_req['fields_of_study'].extend([match.strip() for match in matches if match.strip()])
        
        # Remove duplicates
        education_req['required_degrees'] = list(set(education_req['required_degrees']))
        education_req['fields_of_study'] = list(set(education_req['fields_of_study']))
        
        return education_req

    def extract_responsibilities(self, text: str) -> List[str]:
        """Extract key responsibilities from job description"""
        responsibilities = []
        
        # Look for responsibility sections
        responsibility_patterns = [
            r'responsibilities\s*:?\s*(.*?)(?=requirements|qualifications|skills|$)',
            r'duties\s*:?\s*(.*?)(?=requirements|qualifications|skills|$)',
            r'you\s+will\s*:?\s*(.*?)(?=requirements|qualifications|skills|$)',
            r'role\s+involves\s*:?\s*(.*?)(?=requirements|qualifications|skills|$)'
        ]
        
        for pattern in responsibility_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                resp_text = match.group(1)
                # Split by bullet points or line breaks
                resp_lines = re.split(r'[•\-\*]|\n|\.|;', resp_text)
                responsibilities.extend([
                    line.strip() for line in resp_lines 
                    if line.strip() and len(line.strip()) > 10
                ])
                break
        
        return responsibilities[:10]  # Return top 10 responsibilities

    def categorize_requirements(self, text: str) -> Dict[str, List[str]]:
        """Categorize requirements into must-have and nice-to-have"""
        requirements = {
            'must_have': [],
            'nice_to_have': [],
            'preferred': []
        }
        
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Check for must-have indicators
            must_have_indicators = ['required', 'must have', 'essential', 'mandatory', 'necessary']
            nice_to_have_indicators = ['preferred', 'desired', 'nice to have', 'plus', 'advantage', 'bonus']
            
            is_must_have = any(indicator in sentence_lower for indicator in must_have_indicators)
            is_nice_to_have = any(indicator in sentence_lower for indicator in nice_to_have_indicators)
            
            if is_must_have and len(sentence.strip()) > 20:
                requirements['must_have'].append(sentence.strip())
            elif is_nice_to_have and len(sentence.strip()) > 20:
                requirements['nice_to_have'].append(sentence.strip())
            elif any(skill_cat for skill_cat in self.skill_patterns.values() 
                    for skill in skill_cat if skill.lower() in sentence_lower):
                requirements['preferred'].append(sentence.strip())
        
        return requirements

    def extract_company_info(self, text: str) -> Dict[str, Optional[str]]:
        """Extract basic company information from job description"""
        company_info = {
            'company_name': None,
            'industry': None,
            'company_size': None,
            'location': None
        }
        
        # Company name patterns (basic)
        company_patterns = [
            r'at\s+([A-Z][A-Za-z\s&.,]+(?:Inc|LLC|Corp|Company|Ltd))',
            r'join\s+([A-Z][A-Za-z\s&.,]{2,20})',
            r'([A-Z][A-Za-z\s&.,]{2,20})\s+is\s+(?:looking|seeking)'
        ]
        
        for pattern in company_patterns:
            match = re.search(pattern, text)
            if match:
                company_info['company_name'] = match.group(1).strip()
                break
        
        # Location patterns
        location_patterns = [
            r'located\s+in\s+([A-Za-z\s,]+)',
            r'based\s+in\s+([A-Za-z\s,]+)',
            r'([A-Z][a-z]+,\s+[A-Z]{2})',  # City, State
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)(?:\s+area|\s+region)?'  # City State
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, text)
            if match:
                company_info['location'] = match.group(1).strip()
                break
        
        return company_info

    def analyze_job_description(self, job_text: str, job_title: str = "Not Specified") -> Dict[str, any]:
        """Main method to analyze job description and extract all information"""
        if not job_text or not job_text.strip():
            raise ValueError("Job description text cannot be empty")
        
        logger.info(f"Analyzing job description for: {job_title}")
        
        # Clean the text
        cleaned_text = self.clean_job_description(job_text)
        
        # Extract all information
        job_analysis = {
            'job_title': job_title,
            'raw_text': job_text,
            'cleaned_text': cleaned_text,
            'required_skills': self.extract_skills_from_job(cleaned_text),
            'experience_requirements': self.extract_experience_requirements(cleaned_text),
            'education_requirements': self.extract_education_requirements(cleaned_text),
            'responsibilities': self.extract_responsibilities(cleaned_text),
            'requirements_categorized': self.categorize_requirements(cleaned_text),
            'company_info': self.extract_company_info(cleaned_text),
            'word_count': len(cleaned_text.split()),
            'analyzed_successfully': True
        }
        
        logger.info(f"Successfully analyzed job description for: {job_title}")
        return job_analysis

    def get_all_required_skills_flat(self, skills_dict: Dict[str, List[str]]) -> List[str]:
        """Flatten skills dictionary to a single list"""
        all_skills = []
        for category_skills in skills_dict.values():
            all_skills.extend(category_skills)
        return all_skills

    def get_priority_skills(self, job_analysis: Dict[str, any]) -> Dict[str, List[str]]:
        """Get skills prioritized by importance (must-have vs nice-to-have)"""
        must_have_skills = []
        nice_to_have_skills = []
        
        # Extract skills from categorized requirements
        must_have_text = ' '.join(job_analysis['requirements_categorized']['must_have']).lower()
        nice_to_have_text = ' '.join(job_analysis['requirements_categorized']['nice_to_have']).lower()
        
        # Check which skills appear in must-have vs nice-to-have sections
        all_skills = self.get_all_required_skills_flat(job_analysis['required_skills'])
        
        for skill in all_skills:
            if skill.lower() in must_have_text:
                must_have_skills.append(skill)
            elif skill.lower() in nice_to_have_text:
                nice_to_have_skills.append(skill)
            else:
                # Default to must-have if not clearly categorized
                must_have_skills.append(skill)
        
        return {
            'must_have': must_have_skills,
            'nice_to_have': nice_to_have_skills
        }


def main():
    """Test function for the job analyzer"""
    analyzer = JobAnalyzer()
    
    # Sample job description for testing
    sample_job = """
    Software Engineer - Machine Learning
    
    We are looking for a skilled Software Engineer with expertise in Machine Learning 
    to join our growing team at TechCorp Inc.
    
    Requirements:
    - Bachelor's degree in Computer Science or related field
    - 3+ years of experience in software development
    - Strong proficiency in Python and Java
    - Experience with machine learning frameworks like TensorFlow or PyTorch
    - Knowledge of SQL databases and cloud platforms (AWS preferred)
    - Must have experience with Git and version control
    
    Nice to have:
    - Master's degree in Machine Learning or Data Science
    - Experience with Docker and Kubernetes
    - Knowledge of React for frontend development
    
    Responsibilities:
    - Develop and deploy machine learning models
    - Collaborate with data scientists and engineers
    - Write clean, maintainable code
    - Participate in code reviews
    
    Location: San Francisco, CA
    """
    
    try:
        result = analyzer.analyze_job_description(sample_job, "Software Engineer - ML")
        print("Job description analyzed successfully!")
        print(f"Required Skills: {result['required_skills']}")
        print(f"Experience Required: {result['experience_requirements']['min_years']} years")
        print(f"Education: {result['education_requirements']['required_degrees']}")
        print(f"Company: {result['company_info']['company_name']}")
        
    except Exception as e:
        print(f"Error testing job analyzer: {e}")


if __name__ == "__main__":
    main()
