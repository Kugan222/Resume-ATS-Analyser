"""
Utility Functions for Resume ATS Analyzer
Common helper functions used across the application
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

def validate_file_path(file_path: str, expected_extension: Optional[str] = None) -> bool:
    """Validate if a file path exists and optionally check extension"""
    
    if not file_path or not isinstance(file_path, str):
        return False
    
    path = Path(file_path)
    
    if not path.exists():
        return False
    
    if not path.is_file():
        return False
    
    if expected_extension:
        if not path.suffix.lower() == expected_extension.lower():
            return False
    
    return True

def clean_filename(filename: str) -> str:
    """Clean filename by removing invalid characters"""
    
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = re.sub(r'\s+', '_', filename)
    filename = filename.strip('.')
    
    # Ensure filename is not too long
    if len(filename) > 200:
        filename = filename[:200]
    
    return filename

def create_directory_if_not_exists(directory_path: str) -> bool:
    """Create directory if it doesn't exist"""
    
    try:
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {directory_path}: {e}")
        return False

def save_json_file(data: Any, file_path: str, indent: int = 2) -> bool:
    """Save data to JSON file safely"""
    
    try:
        # Create directory if needed
        directory = Path(file_path).parent
        create_directory_if_not_exists(str(directory))
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
        
        logger.info(f"Data saved to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save JSON file {file_path}: {e}")
        return False

def load_json_file(file_path: str) -> Optional[Any]:
    """Load data from JSON file safely"""
    
    try:
        if not validate_file_path(file_path, '.json'):
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Data loaded from {file_path}")
        return data
        
    except Exception as e:
        logger.error(f"Failed to load JSON file {file_path}: {e}")
        return None

def load_text_file(file_path: str) -> Optional[str]:
    """Load text content from file safely"""
    
    try:
        if not validate_file_path(file_path):
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return content.strip()
        
    except Exception as e:
        logger.error(f"Failed to load text file {file_path}: {e}")
        return None

def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes"""
    
    try:
        if not validate_file_path(file_path):
            return 0.0
        
        size_bytes = Path(file_path).stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        return round(size_mb, 2)
        
    except Exception as e:
        logger.error(f"Failed to get file size for {file_path}: {e}")
        return 0.0

def find_pdf_files(directory: str, max_files: int = 100) -> List[str]:
    """Find all PDF files in a directory"""
    
    pdf_files = []
    
    try:
        if not os.path.isdir(directory):
            return pdf_files
        
        directory_path = Path(directory)
        
        for pdf_file in directory_path.glob("*.pdf"):
            if len(pdf_files) >= max_files:
                break
            pdf_files.append(str(pdf_file))
        
        # Sort files alphabetically
        pdf_files.sort()
        
        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
        return pdf_files
        
    except Exception as e:
        logger.error(f"Error finding PDF files in {directory}: {e}")
        return pdf_files

def calculate_percentage(part: float, total: float) -> float:
    """Calculate percentage safely"""
    
    if total == 0:
        return 0.0
    
    percentage = (part / total) * 100
    return round(percentage, 2)

def normalize_skill_name(skill: str) -> str:
    """Normalize skill name for better matching"""
    
    if not skill or not isinstance(skill, str):
        return ""
    
    # Convert to lowercase and strip whitespace
    normalized = skill.lower().strip()
    
    # Handle common variations
    skill_mappings = {
        'c++': 'cpp',
        'c#': 'csharp',
        'node.js': 'nodejs',
        'react.js': 'react',
        'vue.js': 'vue',
        'angular.js': 'angular',
        'scikit-learn': 'sklearn',
        'sci-kit learn': 'sklearn',
        'tensor flow': 'tensorflow',
        'py torch': 'pytorch',
        'amazon web services': 'aws',
        'google cloud platform': 'gcp',
        'microsoft azure': 'azure'
    }
    
    return skill_mappings.get(normalized, normalized)

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string"""
    
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to maximum length"""
    
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def extract_email_from_text(text: str) -> Optional[str]:
    """Extract email address from text"""
    
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    match = re.search(email_pattern, text)
    
    return match.group() if match else None

def extract_phone_from_text(text: str) -> Optional[str]:
    """Extract phone number from text"""
    
    # Pattern for various phone number formats
    phone_pattern = r'(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
    match = re.search(phone_pattern, text)
    
    return match.group() if match else None

def get_project_info() -> Dict[str, str]:
    """Get project information"""
    
    return {
        'name': 'Resume ATS Analyzer',
        'version': '1.0.0',
        'description': 'AI-powered resume screening system',
        'author': 'ATS Team',
        'license': 'MIT'
    }

def create_sample_resume_instructions() -> str:
    """Create instructions for adding sample resumes"""
    
    instructions = """
    📄 Adding Sample Resumes for Testing
    =====================================
    
    To test the system with your own resumes:
    
    1. Save your resume as a PDF file
    2. Place it in the 'data/sample_resumes/' folder
    3. Ensure the PDF contains selectable text (not just images)
    4. Run the system using:
       - Interactive mode: python main.py --interactive
       - Command line: python main.py --resume path/to/resume.pdf --job job_description.txt
    
    📝 Resume Format Tips:
    - Use standard section headings (Experience, Education, Skills, Projects)
    - Include contact information (email, phone)
    - List skills clearly and use industry-standard terms
    - Quantify achievements where possible
    - Ensure the PDF is text-searchable
    
    🎯 Best Results:
    - Well-structured resumes with clear sections
    - Industry-specific keywords and terminology
    - Detailed job descriptions with specific requirements
    - Matching the resume content to the job requirements
    """
    
    return instructions

def main():
    """Test utility functions"""
    
    print("🧰 Testing Utility Functions")
    print("=" * 40)
    
    # Test file validation
    print(f"✓ Current directory exists: {validate_file_path('.')}")
    print(f"✓ Test file validation works: {validate_file_path('nonexistent.pdf', '.pdf')}")
    
    # Test skill normalization
    test_skills = ['C++', 'Node.js', 'TensorFlow', 'AWS']
    print("\n🏷️ Skill Normalization:")
    for skill in test_skills:
        normalized = normalize_skill_name(skill)
        print(f"  {skill} -> {normalized}")
    
    # Test text utilities
    long_text = "This is a very long text that needs to be truncated for display purposes."
    print(f"\n✂️ Text truncation: {truncate_text(long_text, 30)}")
    
    # Test project info
    info = get_project_info()
    print(f"\n📊 Project: {info['name']} v{info['version']}")
    
    print("\n✅ All utility functions working correctly!")

if __name__ == "__main__":
    main()
